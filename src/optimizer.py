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
from src.engine_xgb import (
    prepare_features, train_xgb_ensemble, get_predictions_ensemble, compute_regime_stats,
)
from src.backtester import vectorized_backtest, format_payout
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

# Hard floors applied before scoring — prevent degenerate or blow-up trials.
# TF-specific hard floor: trials below these counts return -50.0 immediately.
# They produce statistically meaningless RF/PF ratios (10 trades → RF=20 by chance)
# and pollute the surrogate model, biasing future sampling toward "tiny trade" configs.
MIN_OOS_TRADES_HARD = {"M5": 60, "M15": 30, "H1": 15}
MAX_FLOAT_DD    = 0.20   # 20% floating drawdown hard cap — terminal for $15 account
PAYOFF_FLOOR_USD = 0.035 # $0.035 minimum average edge per trade — covers spread + gives real alpha
RAM_HIGH_PCT    = 90     # pause new trials when used RAM exceeds this %
RAM_PAUSE_SEC   = 30     # seconds to sleep when RAM is low
# TF-specific progressive penalty thresholds — trades below these earn score × 0.1
TF_MIN_OOS_TRADES = {"H1": 15, "M15": 140, "M5": 350}


def _get_tier(balance: float) -> str:
    return "small" if balance <= SMALL_ACCOUNT_THRESHOLD else "growth"


def _score_result(result: dict, tier: str = None, broker: str = None, tf: str = "H1") -> float:
    """Complex Criterion: (RF × 0.4) + (PF × 0.3) + (Sharpe × 0.3) + Consistency bonus (M5/M15).

    RF  = OOS Net Profit / OOS Max Floating Drawdown   (capped at 50)
    PF  = Gross Profit / |Gross Loss|                  (capped at 10)

    Weighting rationale:
    - RF (40 %): primary capital-preservation metric; rewards profit relative to risk
    - PF (30 %): trade quality; filters inconsistent winners that inflate Sharpe
    - Sharpe (30 %): risk-adjusted consistency; prevents high-RF/noisy trajectories
    - Consistency bonus (M5/M15 only, max +0.5): rewards steady week-to-week income
      over models that owe their RF/PF to 1-2 outlier streaks.  Not applied to H1
      (swing TF with only 1-2 trades/week — weekly bucketing adds no signal there).
    """
    net_profit  = result.get("total_return", 0.0)
    floating_dd = result.get("floating_max_drawdown", result.get("max_drawdown", 0.0))
    sharpe      = result.get("sharpe_ratio", 0.0)
    pf          = result.get("profit_factor", 1.0)

    if floating_dd <= 0:
        rf = 5.0 if net_profit > 0 else 0.0
    else:
        rf = min(net_profit / floating_dd, 5.0)   # cap at 5 — max contribution 2.0
    pf = min(pf, 3.0)                              # cap at 3 — max contribution 0.9

    score = float((rf * 0.4) + (pf * 0.3) + (sharpe * 0.3))

    if tf.upper() in ("M5", "M15"):
        consistency = result.get("return_consistency", 0.0)
        score += consistency * 0.5   # max +0.5 for perfectly steady weekly income

    return score


def make_objective(balance: float = 15.0, broker: str = "standard", tf: str = "H1"):
    """Return an Optuna objective function for the given account / TF context."""
    tier = _get_tier(balance)

    def objective(trial: optuna.Trial) -> float:
        # obs_cov controls Kalman filter responsiveness (lower = trusts measurements
        # more = faster regime tracking).  M5 uses a wider lower range (0.05) so
        # Optuna can find highly-responsive configs that catch 5-min micro-regime
        # shifts before they reverse.  The 0.5 floor on H1/M15 prevents degenerate
        # covariance errors that occur at low obs_cov on smoother hourly bars.
        # The persistence (<0.65) and vol-ratio (>10) guards catch any remaining
        # degenerate HMMs that slip through at low obs_cov.
        if tf.upper() == "M5":
            obs_cov = trial.suggest_float("obs_cov", 0.05, 5.0, log=True)
        else:
            obs_cov = trial.suggest_float("obs_cov", 0.5,  5.0, log=True)
        # M5 Kalman is calibrated for 5-min noise; H1/M15 have smoother signals
        # so trans_cov above 0.03 tends to produce chaotic Bull/Chop oscillation
        # (49K+ transitions, identical state means) that the persistence guard
        # below must then reject.  Cap it early to save wasted trials.
        if tf.upper() == "M5":
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.1,  log=True)
        else:
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.03, log=True)
        # n_states=3 is always degenerate for M5 — Bull and Chop collapse to
        # identical means (0.000016 return, 0.000340 vol) producing 500K+ HMM
        # transitions and a non-positive-definite covariance matrix.
        # n_states=2 is HMM-stable (~6K transitions) but has no Chop state:
        # every bar is forced Bull or Bear, the strategy is always in the
        # market, OOS DD balloons to 50-74%, and the score formula makes
        # convergence mathematically impossible.  n_states=4 is the only
        # option that is both stable and has a Chop state to absorb ambiguous
        # bars.  Same reasoning applies to H1/M15.
        if tf.upper() == "M5":
            n_states = trial.suggest_categorical("n_states", [4])
        elif tf.upper() == "H1":
            # H1: require [3,4].  n_states=3 with obs_cov < ~1.0 collapses Bull
            # and Chop to identical means (49K+ transitions) — the same degenerate
            # pattern seen on M5 with 3 states.  Allowing n_states=4 gives Optuna a
            # stable fallback (Chop_Low / Chop_High micro-regimes) while still
            # producing clean Bull/Bear signals.  n_states=2 excluded: no Chop
            # state causes counter-trend signals via regime-aligned filter.
            n_states = trial.suggest_int("n_states", 3, 4)
        elif tf.upper() == "M15":
            # M15: n_states=3 degenerates identically to M5 — Bull and Chop
            # collapse to identical means (~return 0.000022, vol ~0.000595) producing
            # 177K–194K transitions and non-positive-definite covariance errors.
            # Restrict to n_states=4 only.  n_states=2 excluded: no Chop state
            # causes counter-trend signals via regime-aligned filter.
            n_states = trial.suggest_categorical("n_states", [4])
        else:
            n_states   = trial.suggest_int("n_states", 3, 4)
        # M5 uses shallower trees (2-3) to prevent IS memorisation across the
        # large bar count; heavier L1 reg (1-20) to sparsify feature weights.
        # H1/M15: max_depth 3-6 allows the model to map complex HMM-state ×
        # dollar-correlation interactions; reg_alpha capped at 1.2 (was 10.0)
        # so the model can express real alpha without being choked by L1 sparsity.
        if tf.upper() == "M5":
            max_depth        = trial.suggest_int("max_depth", 2, 3)
            reg_alpha        = trial.suggest_float("reg_alpha", 1.0, 20.0, log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 15)
        elif tf.upper() == "H1":
            max_depth        = trial.suggest_int("max_depth", 3, 6)
            reg_alpha        = trial.suggest_float("reg_alpha", 0.01, 1.2, log=True)
            # 5–25: enough regularisation to prevent H1 overfitting without
            # collapsing XGB test accuracy to ~50% (random) as 20–50 did.
            min_child_weight = trial.suggest_int("min_child_weight", 5, 25)
        else:
            max_depth        = trial.suggest_int("max_depth", 3, 6)
            reg_alpha        = trial.suggest_float("reg_alpha", 0.01, 1.2, log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 15)
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        # H1: cap n_estimators at 150 — more trees memorise per-bar noise on hourly data.
        n_estimators     = (trial.suggest_int("n_estimators",  50, 150, step=50)
                            if tf.upper() == "H1"
                            else trial.suggest_int("n_estimators", 100, 500, step=50))
        subsample        = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        gamma            = trial.suggest_float("gamma",     0.01, 0.5,  log=True)

        try:
            df = process_pipeline(
                obs_cov=obs_cov, trans_cov=trans_cov, save=False, tf=tf
            )
            _hmm, states, _ = fit_hmm(df, n_states=n_states, tf=tf)

            # ── HMM quality gate ─────────────────────────────────────────────
            # Reject trials where any state has low self-transition probability.
            # Persistence < 0.65 means the state flips very frequently (e.g.
            # Bull↔Chop oscillating with identical means, 49K+ transitions).
            # Threshold relaxed from 0.72 → 0.65 so the HMM can react faster
            # to new directional data while still blocking degenerate configs.
            _min_persist = min(_hmm.transmat_[i, i] for i in range(n_states))
            if _min_persist < 0.65:
                logger.warning(
                    "Trial %d: degenerate HMM (min state persistence=%.4f < 0.65) "
                    "— penalising.",
                    trial.number, _min_persist,
                )
                return -100.0

            # Spike-catcher Bull guard: one state with vol >> all others passes
            # the persistence check (self-transition=0.9500 > 0.80) but captures
            # only extreme outlier bars, leaving the other states mislabelled.
            # Pattern: max_vol/median_vol > 10 (e.g. 0.030731 / 0.000480 = 64).
            _vols = sorted(
                _hmm.covars_[i][0, 0] ** 0.5 for i in range(n_states)
            )
            _vol_ratio = _vols[-1] / _vols[len(_vols) // 2]  # max / median
            if _vol_ratio > 10:
                logger.warning(
                    "Trial %d: degenerate HMM (max/median vol ratio=%.1f > 10) "
                    "— spike-catcher state detected, penalising.",
                    trial.number, _vol_ratio,
                )
                return -100.0

            X, y, df_aligned, _    = prepare_features(df, states, tf=tf)
            _train_ratio = {"H1": 0.70, "M15": 0.65, "M5": 0.65}.get(tf.upper(), 0.70)
            models, thresholds, metrics = train_xgb_ensemble(
                X, y,
                train_ratio=_train_ratio,
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

            # Compute IS regime statistics for Z-Score signal calibration
            split_idx = metrics.get("split_idx")
            X_is      = X.iloc[:split_idx] if split_idx else X
            states_aligned = states[df.index.isin(df_aligned.index)]
            hmm_states_is  = states_aligned[:len(X_is)]
            regime_stats   = compute_regime_stats(models, thresholds, X_is, hmm_states_is)
            metrics["regime_stats"] = regime_stats

            result = vectorized_backtest(
                df_aligned, probabilities, states_aligned,
                split_idx=split_idx,
                account_size=balance,
                broker=broker,
                tf=tf,
                regime_stats=regime_stats,
            )

            # Score on OOS only to prevent IS data leakage
            if split_idx and "oos_sharpe_ratio" in result:
                oos_n   = result.get("oos_n_trades", 0)
                oos_fdd = result.get("oos_floating_max_drawdown",
                                     result.get("oos_max_drawdown", 0.0))

                # Absolute hard floor — statistically meaningless trials with too few
                # OOS trades produce artefact RF/PF ratios (e.g. 8 trades → RF=20)
                # that permanently bias the surrogate toward "tiny trade" configs.
                # Return -50.0 (not -10.0) so these trials are clearly dead in the
                # surrogate's landscape rather than marginally below average.
                _hard_floor = MIN_OOS_TRADES_HARD.get(tf.upper(), 10)
                if oos_n < _hard_floor:
                    return -50.0

                # Safety floor — 20% floating DD is terminal for a $15 account
                if oos_fdd > MAX_FLOAT_DD:
                    logger.warning(
                        "Trial %d: safety floor (OOS floating DD=%.3f > %.0f%%)",
                        trial.number, oos_fdd, MAX_FLOAT_DD * 100,
                    )
                    return -50.0

                oos_result = {
                    "total_return":          result.get("oos_total_return", 0.0),
                    "floating_max_drawdown": oos_fdd,
                    "sharpe_ratio":          result.get("oos_sharpe_ratio", 0.0),
                    "profit_factor":         result.get("oos_profit_factor", 1.0),
                    "expected_payoff":       result.get("oos_expected_payoff", 0.0),
                    "return_consistency":    result.get("oos_return_consistency", 0.0),
                }
                score = _score_result(oos_result, tier, broker, tf)

                # H1 Drawdown Guard: a 5% floating DD on an hourly model
                # typically means a runaway multi-day trend against a Mean
                # Reversion position — a structural failure, not just noise.
                # Half the score to strongly bias Optuna toward low-DD H1 solutions.
                if tf.upper() == "H1" and oos_fdd > 0.05:
                    score *= 0.5

                # Progressive trade penalty — low-count solutions score at 10%
                # so Optuna learns to explore higher-density regions without
                # the hard cliff of a return -10.0 that poisons the surrogate.
                tf_floor = TF_MIN_OOS_TRADES.get(tf.upper(), 60)
                if oos_n < tf_floor:
                    score *= 0.1

                # Payoff floor — any trial whose average dollar edge < $0.035 per
                # trade (3.5c on a $15 cent account) gets a 90% penalty.  Trades
                # that don't move enough to cover spread are noise, not alpha.
                oos_payoff_usd = oos_result["expected_payoff"] * balance
                if oos_payoff_usd < PAYOFF_FLOOR_USD:
                    score *= 0.1

                # Activity Bonus (M5 only) — penalise solutions that "hide"
                # drawdown by barely trading, and reward genuinely high-frequency
                # configs that prove robustness across many market opportunities.
                # < 150 trades: extra 50% cut on top of the progressive 10% penalty.
                # > 300 trades: 20% bonus — pushes Optuna toward the M5 "heartbeat".
                if tf.upper() == "M5":
                    if oos_n < 150:
                        score *= 0.5
                    elif oos_n > 300:
                        score *= 1.2

                # IS/OOS Sharpe ratio constraint — discard trials where the model
                # generalises poorly (IS Sharpe >> OOS Sharpe).  A ratio < 0.35
                # means the model fitted IS noise; the edge evaporates in OOS.
                # Only applied when IS Sharpe is meaningfully positive (> 0.1) to
                # avoid false-positive rejections on flat IS periods.
                is_sharpe  = result.get("is_sharpe_ratio",  result.get("sharpe_ratio", 0.0))
                oos_sharpe = result.get("oos_sharpe_ratio", 0.0)
                if is_sharpe > 0.1:
                    generalization_ratio = oos_sharpe / is_sharpe
                    if generalization_ratio < 0.35:
                        logger.warning(
                            "Trial %d: IS/OOS Sharpe ratio=%.2f < 0.35 "
                            "(IS=%.3f OOS=%.3f) — overfitting, penalising.",
                            trial.number, generalization_ratio, is_sharpe, oos_sharpe,
                        )
                        return -50.0
            else:
                score = _score_result(result, tier, broker, tf)

            _oos_payoff  = result.get("oos_expected_payoff", result.get("expected_payoff", 0.0))
            _oos_payout  = format_payout(
                result.get("oos_total_return", result.get("total_return", 0.0)),
                balance, broker,
            )
            _oos_eff     = result.get("oos_avg_efficiency", result.get("avg_efficiency", 0.0))
            logger.info(
                "Trial %d [%s/%s $%.0f]: score=%.3f  "
                "RF=%.2f  PF=%.2f  Consist=%.2f  Payoff=$%.4f  Eff=%.2fx  FloatDD=%.1f%%  "
                "trades=%d  payout=%s",
                trial.number, tf, broker, balance,
                score,
                result.get("oos_recovery_factor", result.get("recovery_factor", 0)),
                result.get("oos_profit_factor",   result.get("profit_factor", 1.0)),
                result.get("oos_return_consistency", result.get("return_consistency", 0.0)),
                _oos_payoff * balance,
                _oos_eff,
                result.get("oos_floating_max_drawdown",
                           result.get("oos_max_drawdown", result["max_drawdown"])) * 100,
                result.get("oos_n_trades", result["n_trades"]),
                _oos_payout,
            )
            _oos_mr_n = result.get("oos_mr_trades", 0)
            if _oos_mr_n > 0:
                logger.info(
                    "  MR Trades: %d | MR WR: %.1f%% | MR P&L (log): %.4f",
                    _oos_mr_n,
                    result.get("oos_mr_win_rate", 0.0) * 100,
                    result.get("oos_mr_pnl", 0.0),
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
                f"Best Score: {best:+.3f}  |  "
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
                    f"Best Score: <b>{best:.3f}</b>  |  "
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
        f"Best Score (RF×Sharpe): <b>{best:.3f}</b>\n"
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
