"""Optuna hyperparameter optimiser for Gold Regime X.

Key design decisions:
- SQLite study.db with load_if_exists=True: crash-safe resume at any point
- Scores on OOS Sharpe ONLY to prevent IS data leakage / overfitting
- Per-trial gc.collect() to prevent RAM accumulation over long M5 runs
- Optional n_jobs>1: Optuna uses threads; note that XGBoost independently
  uses multiple cores, so n_jobs>2 typically yields diminishing returns
  unless you have 16+ cores.  Default n_jobs=1 is safest.
- psutil RAM guard: if RAM usage exceeds 90%, new trials pause 30s to let
  the OS breathe before dispatching more work.
- Progress dashboard every 5 trials with ETA.
- Telegram heartbeat every 10% of requested trials (if credentials set).
"""

import gc
import os
import time

import optuna

from src.processor import process_pipeline
from src.engine_hmm import fit_hmm
from src.engine_xgb import prepare_features, train_xgb_ensemble, get_predictions_ensemble
from src.backtester import vectorized_backtest
from src.risk_manager import SMALL_ACCOUNT_THRESHOLD
from src.logger import setup_logger

logger = setup_logger(__name__)

# Optional psutil for RAM guard — degrades gracefully if not installed
try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    logger.debug("psutil not installed — RAM guard disabled.")

def _study_db(broker: str) -> str:
    """Return the SQLite storage URL for a given broker, e.g. sqlite:///models/study_headway_cent.db."""
    return f"sqlite:///models/study_{broker}.db"
DD_HARD_LIMIT   = 0.15
# Minimum OOS trades before a trial is considered scoreable.
# M5 has 288 bars/day — the OOS window (~2yr) allows up to ~870 trades at
# 2/day cap.  Setting 300 forces at least 0.69 trades/day on average,
# preventing the optimizer from rewarding ultra-infrequent cherry-picked wins.
# Both M15 and H1 have ~4.1 years of OOS data (125K H1 / 493K M15 bars, 20% split),
# so 200 trades = ~49/year = ~12/quarter — active enough to validate on 3m live data.
MIN_OOS_TRADES_BY_TF: dict[str, int] = {"M5": 300, "M15": 200, "H1": 200}
MIN_OOS_TRADES  = 50   # fallback for unknown TFs
RAM_HIGH_PCT    = 90    # pause new trials when used RAM exceeds this %
RAM_PAUSE_SEC   = 30    # seconds to sleep when RAM is low

# Study tier objectives --------------------------------------------------------
# Small accounts ($15–$50): extreme drawdown protection — Sharpe - DD×5.0
# Growth accounts (>$50):   balanced with higher frequency — Sharpe - DD×3.0
# Both tiers now use 15% DD cap — BUY+SELL doubles trade frequency so the
# previous 10% cap rejected every trial even when OOS Sharpe was acceptable.
TIER_CONFIGS = {
    "small":  {"dd_penalty": 5.0, "dd_limit": 0.15},
    "growth": {"dd_penalty": 3.0, "dd_limit": 0.15},
}


def _get_tier(balance: float) -> str:
    return "small" if balance <= SMALL_ACCOUNT_THRESHOLD else "growth"


def _score_result(result: dict, tier: str, broker: str) -> float:
    cfg  = TIER_CONFIGS[tier]
    dd   = result["max_drawdown"]
    base = (result["sharpe_ratio"] - dd * cfg["dd_penalty"]
            if broker == "headway_cent"
            else result["sharpe_ratio"])
    if dd > cfg["dd_limit"]:
        # Sliding penalty: -2.0 per 1% of DD over the limit.
        # Replaces the hard -10.0 so trials with, e.g., 16% DD on a 15% limit
        # can still score positively instead of being discarded outright.
        overshoot = (dd - cfg["dd_limit"]) * 200   # 0.01 excess → -2.0
        return base - overshoot
    return base


def make_objective(balance: float = 15.0, broker: str = "standard", tf: str = "H1"):
    """Return an Optuna objective function for the given account / TF context."""
    tier = _get_tier(balance)
    min_oos_trades = MIN_OOS_TRADES_BY_TF.get(tf.upper(), MIN_OOS_TRADES)

    def objective(trial: optuna.Trial) -> float:
        # M5: obs_cov floor raised to 1.0 — values below ~0.5 produce a
        # degenerate Kalman filter where Bull/Chop become identical states
        # and the HMM fires 500K+ transitions (every 5-min bar).
        # H1/M15 keep the wide range since their bars carry more information.
        if tf.upper() == "M5":
            obs_cov = trial.suggest_float("obs_cov", 1.0, 5.0, log=True)
        else:
            obs_cov = trial.suggest_float("obs_cov", 0.1, 5.0, log=True)
        # M5 Kalman is calibrated for 5-min noise; H1/M15 have smoother signals
        # so trans_cov above 0.03 tends to produce chaotic Bull/Chop oscillation
        # (49K+ transitions, identical state means) that the persistence guard
        # below must then reject.  Cap it early to save wasted trials.
        if tf.upper() == "M5":
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.1,  log=True)
        else:
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.03, log=True)
        # M5: n_states=3 is always degenerate for 5-min bars — Bull and Chop
        # collapse to identical means (0.000016 return, 0.000340 vol) producing
        # 500K+ HMM transitions and a non-positive-definite covariance matrix.
        # Restrict M5 to {2, 4} which are both stable.
        #
        # H1/M15: n_states=2 has no Chop state.  With the regime-aligned filter
        # (BUY only in Bull, SELL only in Bear) every bar is forced into one
        # direction — in a strong trending market the HMM mislabels many bars as
        # the wrong regime and the model fires signals counter to the trend.
        # Require at least 3 states so ambiguous bars enter Chop (no signal) and
        # only clean Bull/Bear periods generate trades.
        if tf.upper() == "M5":
            n_states = trial.suggest_categorical("n_states", [2, 4])
        else:
            n_states   = trial.suggest_int("n_states", 3, 4)
        # M5 probs cluster below 0.56 in live — narrow range forces the
        # optimizer to find high-frequency signals in the 0.50–0.55 window
        # for cent accounts.  Standard accounts have higher per-trade costs
        # so a more conservative range (0.55–0.60) ensures spread is covered.
        # H1/M15: ceiling cut to 0.58 (was 0.65) — prevents ultra-conservative
        # solutions that produce <5 live trades/quarter; short floor raised to
        # 0.42 (was 0.35) for symmetric buy/sell sensitivity.
        # short_threshold must stay below prob_threshold (no-trade zone must exist).
        if tf.upper() == "M5":
            if broker == "standard":
                prob_threshold  = trial.suggest_float("prob_threshold",  0.55, 0.60)
                short_threshold = trial.suggest_float("short_threshold", 0.40, 0.45)
            else:
                prob_threshold  = trial.suggest_float("prob_threshold",  0.50, 0.53)
                short_threshold = trial.suggest_float("short_threshold", 0.44, 0.50)
        else:
            prob_threshold  = trial.suggest_float("prob_threshold",  0.50, 0.58)
            short_threshold = trial.suggest_float("short_threshold", 0.42, 0.50)

        # Guard: thresholds must not overlap — a crossover means every bar gets
        # both a BUY and SELL signal simultaneously, which is nonsensical.
        if short_threshold >= prob_threshold:
            return -10.0

        # M5 uses shallower trees (2-3) to prevent IS memorisation across the
        # large bar count; heavier L1 reg (1-20) to sparsify feature weights.
        if tf.upper() == "M5":
            max_depth = trial.suggest_int("max_depth", 2, 3)
            reg_alpha = trial.suggest_float("reg_alpha", 1.0, 20.0, log=True)
        else:
            max_depth = trial.suggest_int("max_depth", 3, 5)
            reg_alpha = trial.suggest_float("reg_alpha", 0.01, 10.0, log=True)
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=50)
        subsample        = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 15)
        gamma            = trial.suggest_float("gamma",     0.01, 5.0,  log=True)

        try:
            df = process_pipeline(
                obs_cov=obs_cov, trans_cov=trans_cov, save=False, tf=tf
            )
            _hmm, states, _ = fit_hmm(df, n_states=n_states)

            # ── HMM quality gate ─────────────────────────────────────────────
            # Reject trials where any state has low self-transition probability.
            # Persistence < 0.80 means the state flips nearly every bar (e.g.
            # Bull↔Chop oscillating with identical means, 49K+ transitions).
            # Those labels are noise and will corrupt backtester P&L.
            _min_persist = min(_hmm.transmat_[i, i] for i in range(n_states))
            if _min_persist < 0.80:
                logger.warning(
                    "Trial %d: degenerate HMM (min state persistence=%.4f < 0.80) "
                    "— penalising.",
                    trial.number, _min_persist,
                )
                return -100.0

            X, y, df_aligned     = prepare_features(df, states)
            models, thresholds, metrics = train_xgb_ensemble(
                X, y,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
            )
            _, probabilities = get_predictions_ensemble(models, thresholds, X)

            states_aligned = states[df.index.isin(df_aligned.index)]
            split_idx      = metrics.get("split_idx")
            result = vectorized_backtest(
                df_aligned, probabilities, states_aligned,
                split_idx=split_idx,
                account_size=balance,
                broker=broker,
                tf=tf,
                prob_threshold=prob_threshold,
                short_threshold=short_threshold,
            )

            # Score on OOS only to prevent IS data leakage
            if split_idx and "oos_sharpe_ratio" in result:
                if result.get("oos_n_trades", 0) < min_oos_trades:
                    return -10.0
                oos_result = {
                    "sharpe_ratio": result["oos_sharpe_ratio"],
                    "max_drawdown": result["oos_max_drawdown"],
                }
                score = _score_result(oos_result, tier, broker)
            else:
                score = _score_result(result, tier, broker)

            logger.info(
                "Trial %d [%s/%s tier=%s $%.0f]: score=%.3f  "
                "IS=%.3f  OOS=%.3f  OOS_DD=%.1f%%  trades=%d",
                trial.number, tf, broker, tier, balance,
                score,
                result.get("is_sharpe_ratio",  result["sharpe_ratio"]),
                result.get("oos_sharpe_ratio", result["sharpe_ratio"]),
                result.get("oos_max_drawdown", result["max_drawdown"]) * 100,
                result.get("oos_n_trades",     result["n_trades"]),
            )
            return score

        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return -100.0

        finally:
            # Release HMM/XGB objects and aligned arrays to prevent RAM creep
            # over long M5 runs (~750K bars × 500 trials)
            gc.collect()

    return objective


def _make_callbacks(total_target: int, study_name: str, already_done: int = 0) -> list:
    """Build Optuna callbacks for progress reporting, RAM guard, and Telegram.

    Args:
        total_target: Total trial target for the full study (not just this session).
        study_name:   Optuna study name (for Telegram messages).
        already_done: Completed trials in the DB before this session started.
                      Progress is reported as total_done / total_target so a
                      resumed run correctly shows e.g. [401/500] not [1/100].
    """
    from src.notifier import send_telegram_msg

    start_time    = [time.time()]   # mutable list for closure mutation
    heartbeat_pct = set()           # tracks which 10% milestones were pinged

    def _callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        total_done = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        session_done = max(0, total_done - already_done)

        if session_done <= 0:
            return

        # ── RAM guard: pause before dispatching the next trial ───────────────
        if _PSUTIL_OK:
            mem = psutil.virtual_memory()
            if mem.percent >= RAM_HIGH_PCT:
                logger.warning(
                    "RAM at %.0f%% (>= %d%%) — pausing %ds.",
                    mem.percent, RAM_HIGH_PCT, RAM_PAUSE_SEC,
                )
                time.sleep(RAM_PAUSE_SEC)

        # ── Progress dashboard every 5 session trials ─────────────────────────
        if session_done % 5 == 0:
            elapsed   = time.time() - start_time[0]
            rate      = elapsed / session_done          # seconds per trial
            remaining = max(0, total_target - total_done)
            eta_sec   = remaining * rate
            if eta_sec >= 3600:
                eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            else:
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
            best    = study.best_value if study.best_trial else float("-inf")
            ram_str = (
                f"  RAM: {psutil.virtual_memory().percent:.0f}%"
                if _PSUTIL_OK else ""
            )
            print(
                f"  [{total_done:>4}/{total_target}]  "
                f"Best Sharpe: {best:+.3f}  |  "
                f"ETA: {eta_str}{ram_str}"
            )

        # ── Telegram heartbeat every 10% of total progress ───────────────────
        if total_target > 0:
            milestone = (int(total_done / total_target * 100) // 10) * 10
            if milestone > 0 and milestone not in heartbeat_pct:
                heartbeat_pct.add(milestone)
                best = study.best_value if study.best_trial else float("-inf")
                send_telegram_msg(
                    f"Optimization <b>{milestone}%</b> complete\n"
                    f"Study: <code>{study_name}</code>\n"
                    f"Best Sharpe: <b>{best:.3f}</b>  |  "
                    f"Trials: {total_done}/{total_target}"
                )

    return [_callback]


def _study_name(
    base: str    = "gold_regime_x",
    broker: str  = "standard",
    tier: str    = "small",
    tf: str      = "H1",
) -> str:
    """Deterministic study name — must be identical across sessions to resume."""
    parts = [base, tier, broker]
    if tf.upper() != "H1":
        parts.append(tf.upper())
    return "_".join(parts)


def run_optimization(
    n_trials: int   = 250,
    balance: float  = 15.0,
    broker: str     = "standard",
    tf: str         = "H1",
    n_jobs: int     = 1,
) -> optuna.Study:
    """Run (or resume) an Optuna study and return it when complete.

    The SQLite study.db persists every completed trial to disk.  If the run
    is interrupted at any point (Ctrl+C, crash, power cut), simply re-run the
    same command — Optuna will load_if_exists=True and pick up exactly where
    it left off.

    To start fresh: delete  models/study.db  manually.

    Args:
        n_trials: Additional trials to run THIS session.  Optuna adds them
                  on top of any already-completed trials in the DB.
        n_jobs:   Parallel trial workers (Python threads).  Default 1.
                  XGBoost already uses all CPU cores internally, so n_jobs>2
                  can cause core contention and actually slow things down.
                  For true multi-process parallelism, open multiple terminals
                  and run the same command simultaneously — they all share the
                  same study.db safely via SQLite locking.
    """
    tier = _get_tier(balance)
    name = _study_name(broker=broker, tier=tier, tf=tf)
    storage = _study_db(broker)

    os.makedirs("models", exist_ok=True)

    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,       # crash-safe resume
        pruner=optuna.pruners.MedianPruner(),
    )

    already_done = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    # n_trials is the TOTAL study target — compute how many remain to run
    remaining = max(0, n_trials - already_done)
    if already_done > 0:
        pct = already_done / n_trials * 100
        print(
            f"\nFailsafe: {already_done}/{n_trials} trials already in study_{broker}.db "
            f"({pct:.0f}% complete). "
            f"Resuming from trial #{already_done + 1} — {remaining} remaining.\n"
        )
    else:
        print(f"\nStarting new study '{name}' — target {n_trials} trials.\n")

    if remaining == 0:
        print("Target already reached — no new trials needed. Use a higher --trials value to continue.\n")
        return study

    logger.info(
        "Optimization: tf=%s  broker=%s  tier=%s  balance=$%.0f  "
        "target=%d  already_done=%d  remaining=%d  n_jobs=%d  study=%s",
        tf, broker, tier, balance, n_trials, already_done, remaining, n_jobs, name,
    )

    # Suppress Optuna's verbose per-trial INFO logs; our callback handles output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        make_objective(balance=balance, broker=broker, tf=tf),
        n_trials=remaining,
        n_jobs=n_jobs,
        show_progress_bar=(n_jobs == 1),   # tqdm bar misleads with threads
        callbacks=_make_callbacks(n_trials, name, already_done=already_done),
    )

    best       = study.best_value
    total_done = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    logger.info(
        "Optimization done: best_score=%.3f  total_trials=%d  params=%s",
        best, total_done, study.best_params,
    )

    # ── Completion banner ─────────────────────────────────────────────────────
    pad = 46
    print(
        "\n"
        "************************************************************\n"
        "*                                                          *\n"
        "*              OPTIMIZATION COMPLETE                       *\n"
        f"*   Study : {name:<{pad}}*\n"
        f"*   Trials: {total_done:<{pad}}*\n"
        f"*   Best  : {best:<+{pad}.4f}*\n"
        "*                                                          *\n"
        "************************************************************\n"
    )

    from src.notifier import send_telegram_msg
    send_telegram_msg(
        f"<b>Optimization 100% Complete!</b>\n"
        f"Study: <code>{name}</code>\n"
        f"Trials: <b>{total_done}</b>\n"
        f"Best OOS Sharpe: <b>{best:.3f}</b>\n"
        + "\n".join(
            f"  <code>{k}</code>: {v}"
            for k, v in study.best_params.items()
        )
    )

    return study


def get_best_params(
    balance: float = 15.0,
    broker: str    = "standard",
    tf: str        = "H1",
) -> dict:
    """Load the best hyperparameters from the persisted SQLite study.

    Tries the tier matching *balance* first.  If that study doesn't exist yet
    (e.g. growth tier has never been optimized) it falls back to the small tier
    so that Optuna thresholds are always used rather than hardcoded defaults.
    """
    tier = _get_tier(balance)
    name = _study_name(broker=broker, tier=tier, tf=tf)
    storage = _study_db(broker)

    def _try_load(db: str) -> dict | None:
        try:
            study = optuna.load_study(study_name=name, storage=db)
            return study.best_params
        except Exception:
            return None

    params = _try_load(storage)
    if params is not None:
        return params

    # Legacy fallback: studies optimised before per-broker DB split were saved
    # to models/study.db.  Try it so users don't lose their optimisation work.
    _legacy = "sqlite:///models/study.db"
    if storage != _legacy and os.path.exists("models/study.db"):
        params = _try_load(_legacy)
        if params is not None:
            logger.info(
                "Loaded params for study '%s' from legacy models/study.db. "
                "Re-run --mode optimize to persist to study_%s.db.",
                name, broker,
            )
            return params

    if tier == "small":
        raise KeyError(f"No Optuna study '{name}' found in {storage} or study.db")
    # Growth (or future tiers) — reuse the small-tier study params so live
    # trading always benefits from Optuna thresholds.
    fallback = _study_name(broker=broker, tier="small", tf=tf)
    logger.info(
        "Growth study '%s' not found — using small-tier params from '%s'.",
        name, fallback,
    )
    study = optuna.load_study(study_name=fallback, storage=storage)
    return study.best_params
