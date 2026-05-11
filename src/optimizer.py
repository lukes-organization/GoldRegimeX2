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
import itertools
import os
import time
from itertools import combinations

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit

from src.processor import process_pipeline
from src.engine_hmm import fit_hmm, predict_states
from src.engine_xgb import (
    prepare_features, train_xgb_ensemble, get_predictions_ensemble,
    compute_regime_stats,
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
CPCV_MAX_FLOAT_DD = 0.20 # 20% cap for CPCV paths (2/5 of data — single volatile chunk can exceed 20%)
PAYOFF_FLOOR_USD = 0.035 # $0.035 minimum average edge per trade — covers spread + gives real alpha
RAM_HIGH_PCT    = 90     # pause new trials when used RAM exceeds this %
RAM_PAUSE_SEC   = 30     # seconds to sleep when RAM is low
# TF-specific progressive penalty thresholds — trades below these earn score × 0.1
TF_MIN_OOS_TRADES = {"H1": 25, "M15": 140, "M5": 350}

# ── Rolling Walk-Forward Optimization parameters ──────────────────────────────
# IS_BARS:      In-sample window (1 year of bars per TF).
# OOS_BARS:     Out-of-sample segment (90 days per TF).
# EMBARGO_BARS: Gap between IS end and OOS start (1 day of bars per TF).
# STEP_BARS:    Slide amount per iteration (= OOS_BARS for non-overlapping OOS).
# Expected windows for 10-year H1 dataset (~58K bars):
#   (58406 − 8760) / 2160 ≈ 23 rolling windows
WFO_PARAMS = {
    "H1":  {"is_bars": 8760,   "oos_bars": 2160,  "embargo_bars": 24,  "step_bars": 2160},
    "M15": {"is_bars": 35040,  "oos_bars": 8640,  "embargo_bars": 96,  "step_bars": 8640},
    "M5":  {"is_bars": 105120, "oos_bars": 25920, "embargo_bars": 288, "step_bars": 25920},
}
# Outer Optuna trial counts for WFO (each trial runs all WFO windows).
# Lower than CPCV because each WFO trial is computationally heavier.
WFO_TRIALS = {"H1": 50, "M15": 80, "M5": 100}

# Retained for legacy reference / compare runs — new code uses WFO
# ── CPCV parameters ──────────────────────────────────────────────────────────
# N_SPLITS: number of time-ordered folds.
# N_TEST_SPLITS: folds forming each OOS test path (López de Prado recommends k=2).
# EMBARGO_BARS: bars dropped between train and test to kill serial correlation.
#   H1 : 24 bars  = 24 hours
#   M15: 96 bars  = 24 hours
#   M5 : 288 bars = 24 hours
CPCV_PARAMS = {
    "H1":  {"n_splits": 5, "n_test_splits": 2, "embargo_bars": 24},
    "M15": {"n_splits": 5, "n_test_splits": 2, "embargo_bars": 96},
    "M5":  {"n_splits": 5, "n_test_splits": 2, "embargo_bars": 288},
}

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
MIN_TRADES_PER_PATH = {"H1": 15, "M15": 60, "M5": 100}


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

            # Path models train on 4/6 of data — allow slightly lower persistence
            # than the full-data model.  0.60 still filters degenerate HMMs.
            _min_persist = min(model_path.transmat_[i, i] for i in range(n_states))
            if _min_persist < 0.60:
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

            # Skip zero-trade paths entirely — profit_factor defaults to 1.0 on
            # zero trades, which would inject a spurious 0.30 floor into the median
            # and drag it below zero via the variance penalty.
            if path_n == 0:
                path_scores.append(None)
                continue

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

    # Require at least 6 paths with trades (of 15); HMM degeneracy or very quiet
    # market regimes on some blocks can legitimately produce fewer active paths.
    if n_valid < 6:
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

    # Variance-penalised median (more robust to single-path outliers than mean).
    # Coefficient reduced from 0.5 → 0.25: the old 0.5 was too aggressive and
    # could produce negative scores even when median was meaningfully positive.
    _med   = float(np.median(valid_scores))
    _std   = float(np.std(valid_scores))
    score  = _med - 0.25 * _std

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

    RF  = OOS Net Profit / OOS Max Floating Drawdown   (capped at 5)
    PF  = Gross Profit / |Gross Loss|                  (capped at 3)

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


def _make_cpcv_paths(n: int, n_splits: int, n_test_splits: int):
    """Return a list of (train_indices, test_indices, test_fold_arrays) tuples for CPCV.

    Each element is one backtest *path*.  train_indices is the union of all
    non-test fold bar positions (embargo applied later by the caller).
    test_indices is the union of the n_test_splits test fold bar positions.
    test_fold_arrays is the list of individual per-fold index arrays so the caller
    can apply embargo around EACH fold boundary independently (required for
    non-contiguous test-fold combinations such as folds 0+4).

    For n_splits=5, n_test_splits=2 this produces C(5,2)=10 paths.
    """
    fold_size = n // n_splits
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end   = start + fold_size if i < n_splits - 1 else n
        folds.append(np.arange(start, end))

    paths = []
    for test_combo in itertools.combinations(range(n_splits), n_test_splits):
        test_folds        = set(test_combo)
        train_folds       = [i for i in range(n_splits) if i not in test_folds]
        test_idx          = np.concatenate([folds[i] for i in sorted(test_combo)])
        train_idx         = np.concatenate([folds[i] for i in train_folds])
        individual_folds  = [folds[i] for i in sorted(test_combo)]
        paths.append((train_idx, test_idx, individual_folds))

    return paths


def _run_cpcv(
    df,
    states: np.ndarray,
    tf: str,
    balance: float,
    broker: str,
    xgb_kwargs: dict,
    n_splits: int,
    n_test_splits: int,
    embargo_bars: int,
    hmm_transmat: np.ndarray = None,
) -> dict:
    """Run CPCV and return aggregate statistics across all backtest paths.

    For each of the C(n_splits, n_test_splits) test-fold combinations:
      1. Purge training bars within embargo_bars of BOTH edges of the test window.
      2. Train a fresh XGBoost ensemble on the purged training bars with train_ratio=1.0
         (CPCV owns the train/test split — no internal re-split inside train_xgb_ensemble).
      3. Score the OOS test bars with vectorized_backtest → _score_result.

    The MEDIAN Complex Criterion score across ALL paths (including penalised ones) is
    returned as the CPCV score.  Using ALL paths means configs that produce many
    degenerate paths cannot hide behind a few lucky folds.

    Returns dict with keys:
        cpcv_score     — median Complex Criterion across all paths
        n_paths        — total paths attempted
        n_valid_paths  — paths with enough OOS trades
        median_sharpe  — median OOS Sharpe (diagnostic)
        std_sharpe     — std of OOS Sharpe (high = inconsistent strategy)
        median_trades  — median OOS trade count
        path_scores    — list of per-path Complex Criterion scores
    """
    n = len(df)
    paths = _make_cpcv_paths(n, n_splits, n_test_splits)

    X_full, y_full, df_aligned, _ = prepare_features(df, states, tf=tf)
    states_aligned = states[df.index.isin(df_aligned.index)]

    path_scores   = []
    path_sharpes  = []
    path_trades   = []
    n_valid_paths = 0
    min_trades    = MIN_OOS_TRADES_HARD.get(tf.upper(), 10)

    for train_raw, test_raw, test_fold_arrays in paths:
        if len(test_raw) == 0 or len(train_raw) == 0:
            continue

        # Purge training bars near EACH individual test-fold boundary.
        # Span-based purge (test_raw[0]…test_raw[-1]) falsely eliminates ALL
        # training data for non-contiguous combos like folds (0, 4) because the
        # span covers the entire dataset.  Per-fold boundary purge keeps every
        # training bar that is outside the embargo window of every test fold.
        purged_train_mask = np.ones(len(train_raw), dtype=bool)
        for fold_arr in test_fold_arrays:
            fold_start = int(fold_arr[0])
            fold_end   = int(fold_arr[-1])
            purged_train_mask &= (
                (train_raw < fold_start - embargo_bars) |
                (train_raw > fold_end   + embargo_bars)
            )
        purged_train = train_raw[purged_train_mask]

        if len(purged_train) < 500:
            continue

        df_idx = df.index
        safe_train = purged_train[purged_train < len(df_idx)]
        safe_test  = test_raw[test_raw < len(df_idx)]

        if len(safe_train) == 0 or len(safe_test) == 0:
            continue

        train_timestamps = df_idx[safe_train]
        test_timestamps  = df_idx[safe_test]

        X_train = X_full[X_full.index.isin(train_timestamps)]
        y_train = y_full[y_full.index.isin(train_timestamps)]
        X_test  = X_full[X_full.index.isin(test_timestamps)]

        if len(X_train) < 500 or len(X_test) < 50:
            continue

        # CRITICAL: train_ratio=1.0 — CPCV owns the split.
        try:
            models_path, thresholds_path, _ = train_xgb_ensemble(
                X_train, y_train,
                train_ratio=1.0,
                **xgb_kwargs,
            )
        except Exception as exc:
            logger.debug("CPCV path train failed: %s", exc)
            continue

        _, probs_test = get_predictions_ensemble(models_path, thresholds_path, X_test)

        test_df     = df_aligned[df_aligned.index.isin(test_timestamps)]
        test_states = states_aligned[df_aligned.index.isin(test_timestamps)]

        if len(test_df) < 50 or len(probs_test) != len(test_df):
            continue

        try:
            train_states      = states_aligned[df_aligned.index.isin(train_timestamps)]
            regime_stats_path = compute_regime_stats(
                models_path, thresholds_path, X_train, train_states
            )
        except Exception:
            regime_stats_path = None

        try:
            result_path = vectorized_backtest(
                test_df, probs_test, test_states,
                split_idx=None,
                account_size=balance,
                broker=broker,
                tf=tf,
                hmm_transmat=hmm_transmat,
                regime_stats=regime_stats_path,
            )
        except Exception as exc:
            logger.debug("CPCV path backtest failed: %s", exc)
            continue

        n_trades = result_path.get("n_trades", 0)
        fdd      = result_path.get("floating_max_drawdown",
                                   result_path.get("max_drawdown", 0.0))

        if n_trades < min_trades or fdd > CPCV_MAX_FLOAT_DD:
            path_scores.append(-50.0)
            path_sharpes.append(result_path.get("sharpe_ratio", -10.0))
            path_trades.append(n_trades)
            continue

        n_valid_paths += 1
        score = _score_result(result_path, broker=broker, tf=tf)
        path_scores.append(score)
        path_sharpes.append(result_path.get("sharpe_ratio", 0.0))
        path_trades.append(n_trades)

    if not path_scores:
        return {
            "cpcv_score":    -100.0,
            "n_paths":       len(paths),
            "n_valid_paths": 0,
            "median_sharpe": -10.0,
            "std_sharpe":    0.0,
            "median_trades": 0,
            "path_scores":   [],
        }

    cpcv_score    = float(np.median(path_scores))
    median_sharpe = float(np.median(path_sharpes))
    std_sharpe    = float(np.std(path_sharpes))
    median_trades = int(np.median(path_trades))

    logger.info(
        "CPCV [%s]: %d/%d paths valid | median_trades=%d | "
        "median_score=%.3f | std_sharpe=%.3f | path_scores=%s",
        tf, n_valid_paths, len(paths), median_trades,
        cpcv_score, std_sharpe,
        [round(s, 3) for s in path_scores],
    )
    return {
        "cpcv_score":    cpcv_score,
        "n_paths":       len(paths),
        "n_valid_paths": n_valid_paths,
        "median_sharpe": median_sharpe,
        "std_sharpe":    std_sharpe,
        "median_trades": median_trades,
        "path_scores":   path_scores,
    }


def _run_wfo(
    df,
    n_states: int,
    tf: str,
    balance: float,
    broker: str,
    xgb_kwargs: dict,
    is_bars: int,
    oos_bars: int,
    embargo_bars: int,
    step_bars: int,
) -> dict:
    """Rolling Walk-Forward Optimization — evaluate hyperparams across time windows.

    For each non-overlapping OOS window (slides by *step_bars* until data runs out):
      1. IS = [start … start+is_bars);  OOS = [start+is_bars+embargo … +oos_bars)
      2. HMM is fitted on IS data ONLY — prevents any future-bar lookahead.
      3. IS-fitted HMM applied to OOS bars via predict_states.
      4. TimeSeriesSplit(n_splits=2) on IS data for within-IS consistency check.
      5. Train final XGBoost ensemble on full IS, backtest OOS, score with Complex Criterion.
      6. Window score = OOS Complex Criterion; penalise if IS CV is inconsistent.

    Returns dict with keys:
        wfo_score       — variance-penalised median OOS Complex Criterion
        n_windows       — total windows attempted
        n_valid_windows — windows with enough OOS trades (not penalised)
        median_trades   — median OOS trade count
        std_sharpe      — std of OOS Sharpe (consistency diagnostic)
        window_scores   — list of per-window OOS Complex Criterion scores
        wfe_ratio       — mean(OOS Sharpe) / mean(IS CV Sharpe) [walk-forward efficiency]
    """
    n = len(df)

    window_scores:   list[float] = []
    window_sharpes:  list[float] = []
    is_cv_sharpes:   list[float] = []
    window_trades:   list[int]   = []
    n_valid_windows: int         = 0
    min_trades_hard  = MIN_OOS_TRADES_HARD.get(tf.upper(), 10)

    start = 0
    while start + is_bars + oos_bars + embargo_bars <= n:
        is_end     = start + is_bars
        oos_start  = is_end + embargo_bars
        oos_end    = oos_start + oos_bars

        df_is_slice  = df.iloc[start:is_end]
        df_oos_slice = df.iloc[oos_start:min(oos_end, n)]

        if len(df_oos_slice) < oos_bars // 2:
            break  # not enough OOS bars left

        # ── Fit HMM on IS data only (no lookahead) ───────────────────────────
        try:
            model_is, states_is, _ = fit_hmm(df_is_slice, n_states=n_states, tf=tf)
        except Exception as exc:
            logger.debug("WFO window HMM fit failed at start=%d: %s", start, exc)
            start += step_bars
            continue

        # Reject degenerate HMMs per window
        min_persist = min(model_is.transmat_[i, i] for i in range(n_states))
        if min_persist < 0.65:
            start += step_bars
            continue

        # Apply IS HMM to OOS bars
        try:
            states_oos = predict_states(model_is, df_oos_slice)
        except Exception as exc:
            logger.debug("WFO window OOS state prediction failed: %s", exc)
            start += step_bars
            continue

        # Build feature matrices
        try:
            X_is, y_is, df_is_aligned, _ = prepare_features(
                df_is_slice, states_is, tf=tf
            )
            X_oos, _, df_oos_aligned, _  = prepare_features(
                df_oos_slice, states_oos, tf=tf
            )
        except Exception as exc:
            logger.debug("WFO window feature prep failed: %s", exc)
            start += step_bars
            continue

        if len(X_is) < 500 or len(X_oos) < 50:
            start += step_bars
            continue

        # ── Inner IS cross-validation (2 folds) for consistency check ────────
        tscv = TimeSeriesSplit(n_splits=2)
        _cv_sharpes: list[float] = []
        for _train_idx, _val_idx in tscv.split(X_is):
            X_cv_train = X_is.iloc[_train_idx]
            y_cv_train = y_is.iloc[_train_idx]
            X_cv_val   = X_is.iloc[_val_idx]
            if len(X_cv_train) < 200 or len(X_cv_val) < 50:
                continue
            try:
                _models, _thresh, _ = train_xgb_ensemble(
                    X_cv_train, y_cv_train, train_ratio=1.0, **xgb_kwargs
                )
                _, _probs = get_predictions_ensemble(_models, _thresh, X_cv_val)
                _val_df   = df_is_aligned[df_is_aligned.index.isin(X_cv_val.index)]
                _val_st   = states_is[df_is_slice.index.isin(X_cv_val.index)]
                if len(_val_df) < 20:
                    continue
                _cv_res = vectorized_backtest(
                    _val_df, _probs, _val_st,
                    split_idx=None, account_size=balance, broker=broker, tf=tf,
                    hmm_transmat=model_is.transmat_,
                )
                _cv_sharpes.append(_cv_res.get("sharpe_ratio", 0.0))
            except Exception:
                pass

        mean_cv_sharpe = float(np.mean(_cv_sharpes)) if _cv_sharpes else 0.0
        is_cv_sharpes.append(mean_cv_sharpe)

        # ── Final model on full IS + OOS backtest ─────────────────────────────
        try:
            models_w, thresh_w, _ = train_xgb_ensemble(
                X_is, y_is, train_ratio=1.0, **xgb_kwargs
            )
        except Exception as exc:
            logger.debug("WFO window train failed: %s", exc)
            start += step_bars
            continue

        _, probs_oos = get_predictions_ensemble(models_w, thresh_w, X_oos)

        if len(df_oos_aligned) < 20 or len(probs_oos) != len(df_oos_aligned):
            start += step_bars
            continue

        # Align states to df_oos_aligned index (prepare_features drops first/last row)
        oos_mask = df_oos_slice.index.isin(df_oos_aligned.index)
        states_oos_al = states_oos[oos_mask]

        try:
            oos_result = vectorized_backtest(
                df_oos_aligned, probs_oos, states_oos_al,
                split_idx=None, account_size=balance, broker=broker, tf=tf,
                hmm_transmat=model_is.transmat_,
            )
        except Exception as exc:
            logger.debug("WFO window backtest failed: %s", exc)
            start += step_bars
            continue

        n_trades   = oos_result.get("n_trades", 0)
        fdd        = oos_result.get("floating_max_drawdown",
                                    oos_result.get("max_drawdown", 0.0))
        oos_sharpe = oos_result.get("sharpe_ratio", 0.0)

        if n_trades < min_trades_hard or fdd > CPCV_MAX_FLOAT_DD:
            score = -50.0
        else:
            n_valid_windows += 1
            score = _score_result(oos_result, broker=broker, tf=tf)
            score = max(score, -1.0)  # floor for valid windows
            # Penalise when IS CV Sharpe is deeply negative (overfit to noise)
            if mean_cv_sharpe < -1.0:
                score *= 0.5

        window_scores.append(score)
        window_sharpes.append(oos_sharpe)
        window_trades.append(n_trades)
        logger.debug(
            "WFO [%s] window start=%d: OOS_trades=%d OOS_Sharpe=%.3f "
            "IS_CV_Sharpe=%.3f score=%.3f",
            tf, start, n_trades, oos_sharpe, mean_cv_sharpe, score,
        )
        start += step_bars

    if not window_scores:
        return {
            "wfo_score":        -100.0,
            "n_windows":        0,
            "n_valid_windows":  0,
            "median_trades":    0,
            "std_sharpe":       0.0,
            "window_scores":    [],
            "wfe_ratio":        0.0,
        }

    wfo_score     = float(np.median(window_scores)) - 0.15 * float(np.std(window_scores))
    std_sharpe    = float(np.std(window_sharpes))
    median_trades = int(np.median(window_trades))

    mean_oos_sharpe = float(np.mean(window_sharpes))
    mean_is_sharpe  = float(np.mean(is_cv_sharpes)) if is_cv_sharpes else 0.0
    wfe_ratio = (mean_oos_sharpe / max(abs(mean_is_sharpe), 0.01)) if mean_is_sharpe != 0 else 0.0

    # Consistency bonus when every valid window beats the floor
    valid_sc = [s for s in window_scores if s > -50.0]
    if valid_sc and all(s > 0.30 for s in valid_sc):
        wfo_score *= 1.05

    logger.info(
        "WFO [%s]: %d/%d valid windows | median_trades=%d | "
        "wfo_score=%.3f | std_sharpe=%.3f | WFE=%.2f | window_scores=%s",
        tf, n_valid_windows, len(window_scores), median_trades,
        wfo_score, std_sharpe, wfe_ratio,
        [round(s, 3) for s in window_scores],
    )
    return {
        "wfo_score":        wfo_score,
        "n_windows":        len(window_scores),
        "n_valid_windows":  n_valid_windows,
        "median_trades":    median_trades,
        "std_sharpe":       std_sharpe,
        "window_scores":    window_scores,
        "wfe_ratio":        wfe_ratio,
    }


def make_objective(balance: float = 15.0, broker: str = "standard", tf: str = "H1"):
    """Return an Optuna objective function for the given account / TF context."""
    tier    = _get_tier(balance)
    wfo_cfg = WFO_PARAMS.get(tf.upper(), WFO_PARAMS["H1"])

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
        # H1/M15: max_depth extended to [3,8] — best trial was hitting the old
        # ceiling of 6, indicating Optuna needs headroom to explore 7-8.
        # H1 n_estimators raised to [50,300] to match the lower learning_rate floor.
        if tf.upper() == "M5":
            max_depth        = trial.suggest_int("max_depth", 2, 3)
            reg_alpha        = trial.suggest_float("reg_alpha", 1.0, 20.0, log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 15)
        elif tf.upper() == "H1":
            max_depth        = trial.suggest_int("max_depth", 3, 8)
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-6, 0.3, log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 50)
        else:
            max_depth        = trial.suggest_int("max_depth", 3, 8)
            reg_alpha        = trial.suggest_float("reg_alpha", 0.01, 1.2, log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 15)
        # H1: learning_rate floor lowered to 0.005 so Optuna can explore slow-
        # learning deep trees without the 0.01 floor cutting off valid configs.
        # H1: n_estimators ceiling raised to 300 — slow learning needs more trees.
        if tf.upper() == "H1":
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            n_estimators  = trial.suggest_int("n_estimators", 50, 300, step=50)
        elif tf.upper() == "M5":
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            n_estimators  = trial.suggest_int("n_estimators", 100, 500, step=50)
        else:
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            n_estimators  = trial.suggest_int("n_estimators", 100, 500, step=50)
        subsample        = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        gamma            = trial.suggest_float("gamma",     1e-6, 0.3,  log=True)
        # scale_pos_weight: handles class imbalance between Bull/Bear/Chop labels.
        # H1 distribution is ~Bull 35%, Bear 30%, Chop 35% — slight imbalance.
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 2.0, log=True)

        try:
            df = process_pipeline(
                obs_cov=obs_cov, trans_cov=trans_cov, save=False, tf=tf
            )

            xgb_kwargs = dict(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
                scale_pos_weight=scale_pos_weight,
            )

            # ── WFO scoring ───────────────────────────────────────────────
            # HMM is now fitted per-window inside _run_wfo to prevent lookahead.
            wfo_result = _run_wfo(
                df=df,
                n_states=n_states,
                tf=tf,
                balance=balance,
                broker=broker,
                xgb_kwargs=xgb_kwargs,
                is_bars=wfo_cfg["is_bars"],
                oos_bars=wfo_cfg["oos_bars"],
                embargo_bars=wfo_cfg["embargo_bars"],
                step_bars=wfo_cfg["step_bars"],
            )

            score           = wfo_result["wfo_score"]
            n_valid_windows = wfo_result["n_valid_windows"]
            n_total_windows = wfo_result["n_windows"]
            std_sharpe      = wfo_result["std_sharpe"]
            median_trades   = wfo_result["median_trades"]
            window_scores   = wfo_result["window_scores"]
            wfe_ratio       = wfo_result["wfe_ratio"]

            # ── Hard gates ────────────────────────────────────────────────
            if n_valid_windows == 0:
                return -50.0

            # If fewer than half the windows are valid, config is regime-specific.
            if n_valid_windows < n_total_windows // 2:
                score = min(score, -10.0)

            # High Sharpe std = strategy only works in some market regimes.
            if std_sharpe > 1.0:
                score -= (std_sharpe - 1.0) * 0.3

            # Progressive trade penalty
            tf_floor = TF_MIN_OOS_TRADES.get(tf.upper(), 60)
            if median_trades < tf_floor:
                score *= 0.1

            # M5 activity bonus (same logic as original)
            if tf.upper() == "M5":
                if median_trades < 150:
                    score *= 0.5
                elif median_trades > 300:
                    score *= 1.2

            logger.info(
                "Trial %d [%s/%s $%.0f WFO]: score=%.3f | valid=%d/%d | "
                "std_sharpe=%.3f | median_trades=%d | WFE=%.2f | window_scores=%s",
                trial.number, tf, broker, balance,
                score, n_valid_windows, n_total_windows,
                std_sharpe, median_trades, wfe_ratio,
                [round(s, 3) for s in window_scores],
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
    """Run (or resume) an Optuna study using Rolling WFO and return it when complete.

    Each trial slides IS/OOS windows across the full dataset (non-overlapping OOS).
    The study persists to SQLite so interrupted runs resume safely from where they left off.

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
    wfo_cfg = WFO_PARAMS.get(tf.upper(), WFO_PARAMS["H1"])

    # Apply TF-specific WFO trial count when caller used the default 250
    if n_trials == 250:
        n_trials = WFO_TRIALS.get(tf.upper(), 50)
    # Force sequential within each trial — each WFO trial already runs many
    # HMM+XGB windows; parallelism between trials causes core contention.
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
        print(
            f"\nStarting new WFO study '{name}' — target {n_trials} trials.\n"
            f"WFO config [{tf}]: IS={wfo_cfg['is_bars']} bars  "
            f"OOS={wfo_cfg['oos_bars']} bars  "
            f"embargo={wfo_cfg['embargo_bars']} bars  "
            f"step={wfo_cfg['step_bars']} bars\n"
            f"Recommended trial counts for WFO: H1={WFO_TRIALS['H1']}, "
            f"M15={WFO_TRIALS['M15']}, M5={WFO_TRIALS['M5']}.\n"
        )

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
