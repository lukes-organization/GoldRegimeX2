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

STUDY_DB        = "sqlite:///models/study.db"
DD_HARD_LIMIT   = 0.15
MIN_OOS_TRADES  = 50
RAM_HIGH_PCT    = 90    # pause new trials when used RAM exceeds this %
RAM_PAUSE_SEC   = 30    # seconds to sleep when RAM is low

# Study tier objectives --------------------------------------------------------
# Small accounts ($15–$50): extreme drawdown protection — Sharpe - DD×5.0
# Growth accounts (>$50):   balanced with higher frequency — Sharpe - DD×3.0
TIER_CONFIGS = {
    "small":  {"dd_penalty": 5.0, "dd_limit": 0.10},
    "growth": {"dd_penalty": 3.0, "dd_limit": 0.15},
}


def _get_tier(balance: float) -> str:
    return "small" if balance <= SMALL_ACCOUNT_THRESHOLD else "growth"


def _score_result(result: dict, tier: str, broker: str) -> float:
    cfg = TIER_CONFIGS[tier]
    if result["max_drawdown"] > cfg["dd_limit"]:
        return -10.0
    if broker == "headway_cent":
        return result["sharpe_ratio"] - result["max_drawdown"] * cfg["dd_penalty"]
    return result["sharpe_ratio"]


def make_objective(balance: float = 15.0, broker: str = "standard", tf: str = "H1"):
    """Return an Optuna objective function for the given account / TF context."""
    tier = _get_tier(balance)

    def objective(trial: optuna.Trial) -> float:
        obs_cov        = trial.suggest_float("obs_cov",        0.1,  5.0,  log=True)
        trans_cov      = trial.suggest_float("trans_cov",      0.001, 0.1, log=True)
        n_states       = trial.suggest_int("n_states", 2, 4)
        prob_threshold = trial.suggest_float("prob_threshold", 0.52, 0.68)

        max_depth        = trial.suggest_int("max_depth", 3, 5)
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=50)
        subsample        = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 15)
        gamma            = trial.suggest_float("gamma",     0.01, 5.0,  log=True)
        reg_alpha        = trial.suggest_float("reg_alpha", 0.01, 10.0, log=True)

        try:
            df = process_pipeline(
                obs_cov=obs_cov, trans_cov=trans_cov, save=False, tf=tf
            )
            _hmm, states, _ = fit_hmm(df, n_states=n_states)
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
            )

            # Score on OOS only to prevent IS data leakage
            if split_idx and "oos_sharpe_ratio" in result:
                if result.get("oos_n_trades", 0) < MIN_OOS_TRADES:
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
            return -10.0

        finally:
            # Release HMM/XGB objects and aligned arrays to prevent RAM creep
            # over long M5 runs (~750K bars × 500 trials)
            gc.collect()

    return objective


def _make_callbacks(n_trials: int, study_name: str, already_done: int = 0) -> list:
    """Build Optuna callbacks for progress reporting, RAM guard, and Telegram.

    Args:
        n_trials:     Number of trials requested for THIS session.
        study_name:   Optuna study name (for Telegram messages).
        already_done: Completed trials in the DB before this session started.
                      All progress percentages are relative to session trials only,
                      so a resumed run shows [1/500] not [373/500].
    """
    from src.notifier import send_telegram_msg

    start_time    = [time.time()]   # mutable list for closure mutation
    heartbeat_pct = set()           # tracks which 10% milestones were pinged

    def _callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        total_done = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        # session_done = trials completed in THIS session only
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
            rate      = elapsed / session_done
            remaining = max(0, n_trials - session_done)
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
                f"  [{session_done:>4}/{n_trials}]  "
                f"Best Sharpe: {best:+.3f}  |  "
                f"ETA: {eta_str}{ram_str}"
            )

        # ── Telegram heartbeat every 10% of SESSION trials ───────────────────
        if n_trials > 0:
            milestone = (int(session_done / n_trials * 100) // 10) * 10
            if milestone > 0 and milestone not in heartbeat_pct:
                heartbeat_pct.add(milestone)
                best = study.best_value if study.best_trial else float("-inf")
                send_telegram_msg(
                    f"Optimization <b>{milestone}%</b> complete\n"
                    f"Study: <code>{study_name}</code>\n"
                    f"Best Sharpe: <b>{best:.3f}</b>  |  "
                    f"Trials: {session_done}/{n_trials}"
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

    os.makedirs("models", exist_ok=True)

    study = optuna.create_study(
        study_name=name,
        storage=STUDY_DB,
        direction="maximize",
        load_if_exists=True,       # crash-safe resume
        pruner=optuna.pruners.MedianPruner(),
    )

    already_done = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    if already_done > 0:
        pct = already_done / (already_done + n_trials) * 100
        print(
            f"\nFailsafe: {already_done} completed trials found in study.db "
            f"({pct:.0f}% of total {already_done + n_trials}). "
            f"Resuming from trial #{already_done + 1}...\n"
        )
    else:
        print(f"\nStarting new study '{name}' — {n_trials} trials.\n")

    logger.info(
        "Optimization: tf=%s  broker=%s  tier=%s  balance=$%.0f  "
        "n_trials=%d  n_jobs=%d  study=%s  already_done=%d",
        tf, broker, tier, balance, n_trials, n_jobs, name, already_done,
    )

    # Suppress Optuna's verbose per-trial INFO logs; our callback handles output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        make_objective(balance=balance, broker=broker, tf=tf),
        n_trials=n_trials,
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
    """Load the best hyperparameters from the persisted SQLite study."""
    tier = _get_tier(balance)
    name = _study_name(broker=broker, tier=tier, tf=tf)
    study = optuna.load_study(study_name=name, storage=STUDY_DB)
    return study.best_params
