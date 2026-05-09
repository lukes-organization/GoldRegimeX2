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
from itertools import combinations

import numpy as np
import optuna

from src.processor import process_pipeline
from src.engine_hmm import fit_hmm, predict_states
from src.engine_xgb import (
    prepare_features, train_xgb_ensemble, get_predictions_ensemble,
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
MIN_OOS_TRADES_HARD = {"M5": 120, "M15": 30, "H1": 20}
MAX_FLOAT_DD    = 0.20   # 20% floating drawdown hard cap — terminal for $15 account
PAYOFF_FLOOR_USD = 0.035 # $0.035 minimum average edge per trade — covers spread + gives real alpha
RAM_HIGH_PCT    = 90     # pause new trials when used RAM exceeds this %
RAM_PAUSE_SEC   = 30     # seconds to sleep when RAM is low
# TF-specific progressive penalty thresholds — trades below these earn score × 0.1
TF_MIN_OOS_TRADES = {"H1": 25, "M15": 140, "M5": 350}

# ── Combinatorial Purged Cross-Validation (CPCV) ─────────────────────────────
# Data is split into N_BLOCKS equal chronological blocks. Every combination of
# K_TEST blocks is held out as the test set (train on remaining N-K blocks).
# C(6,2) = 15 paths — provides broad coverage across all market regimes without
# hardcoding specific dates.  Purge gap at each train/test boundary removes
# max_hold_bars to prevent feature look-ahead bias (Kalman/ATR lookbacks).
CPCV_N_BLOCKS  = 6
CPCV_K_TEST    = 2
# Fewer trials than static because each trial runs 15 full HMM+XGB fits.
CPCV_TRIALS    = {"H1": 80, "M15": 120, "M5": 200}
# Purge gap at each train/test boundary — matches MAX_HOLD_BARS in signal_engine
CPCV_PURGE_BARS = {"H1": 24, "M15": 32, "M5": 48}
# Minimum trades per CPCV path for the result to count; paths below this are
# penalised so Optuna steers toward configs that trade actively in all regimes.
MIN_TRADES_PER_PATH = {"H1": 30, "M15": 60, "M5": 100}


class CPCVSplitter:
    """Generate train/test boolean mask pairs via Combinatorial Purged CV.

    Splits the timeline into ``n_blocks`` equal chronological blocks and
    yields all C(n_blocks, k_test) = 15 (for 6/2) train/test mask pairs.
    A ``purge_bars``-wide gap is carved out of the training mask at every
    train↔test boundary to prevent feature look-ahead leakage from Kalman
    filter and ATR smoothing lookbacks.

    Usage::
        splitter = CPCVSplitter(n=len(df), purge_bars=24)
        for train_mask, test_mask in splitter.split():
            # train_mask, test_mask: bool arrays of length n
    """

    def __init__(self, n: int, n_blocks: int = CPCV_N_BLOCKS,
                 k_test: int = CPCV_K_TEST, purge_bars: int = 24):
        self.n         = n
        self.n_blocks  = n_blocks
        self.k_test    = k_test
        self.purge_bars = purge_bars
        self._blocks   = np.array_split(np.arange(n), n_blocks)

    def split(self):
        """Yield (train_mask, test_mask) for all C(n_blocks, k_test) combos."""
        for test_combo in combinations(range(self.n_blocks), self.k_test):
            train_mask = np.ones(self.n, dtype=bool)
            test_mask  = np.zeros(self.n, dtype=bool)

            for blk_idx in test_combo:
                blk = self._blocks[blk_idx]
                test_mask[blk] = True
                # Carve purge gap from training set at both boundaries of the block
                start, end = int(blk[0]), int(blk[-1])
                lo = max(0, start - self.purge_bars)
                hi = min(self.n, end + 1 + self.purge_bars)
                train_mask[lo:hi] = False

            train_mask[test_mask] = False  # test bars are never in training
            yield train_mask, test_mask


def _get_tier(balance: float) -> str:
    return "small" if balance <= SMALL_ACCOUNT_THRESHOLD else "growth"


def compute_cpcv_score(
    df_full,
    balance: float,
    broker: str,
    tf: str,
    n_states: int,
    xgb_kwargs: dict,
) -> float:
    """Combinatorial Purged Cross-Validation score — 15 paths, C(6,2).

    For each of the 15 train/test mask pairs produced by CPCVSplitter:
      1. Fit HMM on concatenated training bars (non-contiguous; jump at
         block boundaries is an accepted approximation).
      2. Predict states on ALL bars using the IS model.
      3. Prepare features; scaler fitted on training bars only.
      4. Train XGB on training bars (train_ratio=1.0 within the masked set).
      5. Run bar-by-bar backtest with test_mask — entries only in test bars,
         force-close at test block boundaries.
      6. Collect path score = RF×0.4 + PF×0.3 + Sharpe×0.3 (capped at 2.0).

    Aggregation:
      - Need ≥ 8 valid paths (of 15) else return -50.0.
      - Trade penalty: total_trades < MIN_TRADES_PER_PATH × n_valid_paths → -50.0.
      - Final score = variance-penalised median of valid path scores.
      - Consistency bonus ×1.05 when all valid paths score > 0.30.
    """
    purge = CPCV_PURGE_BARS.get(tf.upper(), 24)
    splitter = CPCVSplitter(n=len(df_full), purge_bars=purge)

    path_scores: list = []
    total_trades: int = 0

    # Predict states on the FULL dataset using a single IS HMM fitted on all data
    # once per trial — expensive but necessary to have coherent state labels.
    # Each path then filters to its training rows for XGB, and uses test_mask
    # for evaluation.  The HMM IS fitted PER PATH so the model only sees
    # training bars during fit.
    for path_idx, (train_mask, test_mask) in enumerate(splitter.split()):
        try:
            df_train = df_full.iloc[train_mask]
            if len(df_train) < 500:
                path_scores.append(None)
                continue

            # Fit HMM on training blocks (concatenated; block-boundary jumps
            # are a standard approximation in CPCV for time-series).
            model_path, train_states, _ = fit_hmm(df_train, n_states=n_states, tf=tf)

            _min_persist = min(model_path.transmat_[i, i] for i in range(n_states))
            if _min_persist < 0.65:
                path_scores.append(None)
                continue

            # Predict states on ALL bars using this path's IS model
            all_states = predict_states(model_path, df_full)

            # Feature preparation — scaler fitted on training rows only
            X_train_raw, y_train_raw, df_train_aligned, scaler = prepare_features(
                df_train, train_states, tf=tf
            )

            # Apply scaler to ALL data (aligned to full df)
            X_all, y_all, df_all_aligned, _ = prepare_features(
                df_full, all_states, feature_scaler=scaler, tf=tf
            )

            # Align train_mask and test_mask to the rows kept after prepare_features
            _kept = df_full.index.isin(df_all_aligned.index)
            train_mask_al = train_mask[_kept]
            test_mask_al  = test_mask[_kept]

            if train_mask_al.sum() < 100:
                path_scores.append(None)
                continue

            # Train XGB on training-block rows of the aligned feature matrix
            X_tr = X_all.iloc[train_mask_al]
            y_tr = y_all.iloc[train_mask_al]
            models_ens, thresholds, _ = train_xgb_ensemble(
                X_tr, y_tr, train_ratio=1.0, **xgb_kwargs
            )

            # Predictions on ALL aligned bars
            _, probs_all = get_predictions_ensemble(models_ens, thresholds, X_all)
            states_all_al = df_all_aligned["hmm_state"].values

            # Backtest with test_mask — only test-block bars are evaluated
            path_result = vectorized_backtest(
                df_all_aligned, probs_all, states_all_al,
                split_idx=None,
                account_size=balance,
                broker=broker,
                tf=tf,
                hmm_transmat=model_path.transmat_,
                test_mask=test_mask_al,
            )

            path_n  = path_result.get("n_trades", 0)
            total_trades += path_n

            rf     = min(path_result.get("recovery_factor", 0.0), 5.0)
            pf     = min(path_result.get("profit_factor", 1.0), 3.0)
            sharpe = path_result.get("sharpe_ratio", 0.0)
            wr     = path_result.get("win_rate", 0.0)
            logger.debug(
                "  CPCV path %d/%d: n=%d WR=%.1f%% Sharpe=%.3f RF=%.2f PF=%.2f",
                path_idx + 1, _n_paths_total(), path_n, wr * 100, sharpe, rf, pf,
            )
            path_score = min(float(rf * 0.4 + pf * 0.3 + sharpe * 0.3), 2.0)
            path_scores.append(path_score)

        except Exception as exc:
            logger.debug("CPCV path %d failed: %s", path_idx + 1, exc)
            path_scores.append(None)

    valid_scores = [s for s in path_scores if s is not None]
    n_valid = len(valid_scores)

    if n_valid < 8:
        logger.warning("CPCV: only %d/%d valid paths — rejecting.", n_valid, len(path_scores))
        return -50.0

    # Hard trade-count floor across all valid paths
    _min_total = MIN_TRADES_PER_PATH.get(tf.upper(), 30) * n_valid
    if total_trades < _min_total:
        logger.warning(
            "CPCV: total_trades=%d < min=%d (%d paths) — rejecting.",
            total_trades, _min_total, n_valid,
        )
        return -50.0

    # Variance-penalised median (more robust to single-path outliers than mean)
    _med   = float(np.median(valid_scores))
    _std   = float(np.std(valid_scores))
    score  = _med - 0.5 * _std

    # Trade-frequency penalty (progressive, not binary) — steers Optuna toward
    # configs that trade actively across all regimes.
    _ideal_total = MIN_TRADES_PER_PATH.get(tf.upper(), 30) * 2 * n_valid
    trade_penalty = min(1.0, total_trades / max(_ideal_total, 1))
    score *= trade_penalty

    # Consistency bonus when every valid path is meaningfully positive
    if all(s > 0.30 for s in valid_scores):
        score *= 1.05

    logger.info(
        "CPCV [%s]: %d/%d paths valid | trades=%d | median=%.3f std=%.3f | score=%.3f",
        tf, n_valid, len(path_scores), total_trades, _med, _std, score,
    )
    return score


def _n_paths_total() -> int:
    """C(CPCV_N_BLOCKS, CPCV_K_TEST) — total number of CPCV paths."""
    from math import comb
    return comb(CPCV_N_BLOCKS, CPCV_K_TEST)


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

            # ── CPCV: 15-path combinatorial purged cross-validation ────────────
            _xgb_kw = dict(
                max_depth=max_depth, learning_rate=learning_rate,
                n_estimators=n_estimators, subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma, reg_alpha=reg_alpha,
            )
            score = compute_cpcv_score(df, balance, broker, tf, n_states, _xgb_kw)
            logger.info(
                "Trial %d [%s/%s $%.0f CPCV]: score=%.3f",
                trial.number, tf, broker, balance, score,
            )
            return score
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
    """Run (or resume) an Optuna study using CPCV and return it when complete.

    Each trial evaluates C(6,2)=15 train/test path combinations.  The study
    persists to SQLite so interrupted runs resume safely from where they left off.

    Args:
        n_trials: Total study target.  Optuna adds new trials on top of those
                  already in the DB — delete models/study_{broker}.db to start fresh.
        n_jobs:   Parallel trial workers.  Default 1 — XGBoost already uses all
                  CPU cores, so n_jobs>1 causes core contention.  For true
                  parallelism, open multiple terminals with the same command.
    """
    tier = _get_tier(balance)
    name = _study_name(broker=broker, tier=tier, tf=tf)
    storage = _study_db(broker)

    # Apply TF-specific CPCV trial count when caller used the default 250
    if n_trials == 250:
        n_trials = CPCV_TRIALS.get(tf.upper(), 80)
    # Force sequential within each trial — 15 HMM+XGB fits per trial already
    # saturates CPU cores; parallelism between trials causes contention.
    n_jobs = 1

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
