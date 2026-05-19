"""Optuna Walk-Forward Optimiser for Gold Regime X — V3.1

Key design:
- Pre-loaded parquet df passed to make_objective; only kalman_smooth called per
  trial (replaces per-trial process_pipeline — 10-50× speedup on long studies)
- Per-window HMM fitting prevents lookahead; IS scaler applied to OOS features
- Config hash guard: study DB invalidated when feature set / TF / version changes
- Consensus params: top-N trial median is more robust than single best trial
- Backward-compatible exports: _score_result, get_best_params, _run_wfo,
  WFO_PARAMS, WFO_PARAMS_FAST, CV_FOLDS, compute_cpcv_score kept for compat
"""

import gc
import hashlib
import itertools
import json
import os
import sys
import time
from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import optuna
import pandas as pd

from src.processor import kalman_smooth
from src.engine_hmm import fit_hmm, predict_states
from src.engine_xgb import (
    prepare_features, train_xgb_ensemble, get_predictions_ensemble,
    compute_regime_stats, get_feature_cols,
)
from src.backtester import vectorized_backtest
from src.risk_manager import SMALL_ACCOUNT_THRESHOLD
from src.logger import setup_logger, append_trial_score

logger = setup_logger(__name__)

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    logger.debug("psutil not installed — RAM guard disabled.")

# ── Versioning ─────────────────────────────────────────────────────────────────
OPTIMIZER_VERSION = "3.8"  # bump when search space or scoring formula changes

# ── Rolling WFO window sizes (bars) ───────────────────────────────────────────
WFO_PARAMS = {
    # H1: step doubled to 2×OOS (4320) so IS overlap between consecutive
    # windows drops from 75% → 50%, giving more independent test periods.
    "H1":  {"is_bars": 8760,   "oos_bars": 2160,  "embargo_bars": 24,  "step_bars": 4320},
    "M15": {"is_bars": 35040,  "oos_bars": 8640,  "embargo_bars": 96,  "step_bars": 8640},
    "M5":  {"is_bars": 105120, "oos_bars": 25920, "embargo_bars": 288, "step_bars": 25920},
}

# Maximum number of WFO windows to evaluate per trial.  When the dataset
# yields more windows than this cap, the OLDEST windows are skipped so that
# only the most-recent MAX_WFO_WINDOWS windows are used.  This keeps trial
# cost predictable and avoids over-weighting stale market regimes.
MAX_WFO_WINDOWS = {"H1": 8, "M15": 8, "M5": 8}
WFO_PARAMS_FAST = {
    "H1":  {"is_bars": 4380,  "oos_bars": 1080, "embargo_bars": 24,  "step_bars": 1080},
    "M15": {"is_bars": 17520, "oos_bars": 4320, "embargo_bars": 96,  "step_bars": 4320},
    "M5":  {"is_bars": 52560, "oos_bars": 8640, "embargo_bars": 288, "step_bars": 8640},
}

# Bars per calendar day per TF
BARS_PER_DAY = {"H1": 24, "M15": 96, "M5": 288}

# Bars per calendar year per TF (for Calmar annualisation)
BARS_PER_YEAR = {"H1": 8760, "M15": 35040, "M5": 105120}

# IS window as fraction of (IS + OOS) bars — used for config hash
IS_OOS_SPLIT = {
    tf: round(WFO_PARAMS[tf]["is_bars"] / (WFO_PARAMS[tf]["is_bars"] + WFO_PARAMS[tf]["oos_bars"]), 4)
    for tf in WFO_PARAMS
}

WFO_TRIALS      = {"H1": 250,  "M15": 100, "M5": 120}
WFO_TRIALS_FAST = {"H1": 60,  "M15": 100, "M5": 120}

# Inner IS cross-validation folds per TF
CV_FOLDS = {"H1": 2, "M15": 3, "M5": 4}

# Hard trade floors — trials below return -50.0 immediately
HARD_TRADE_FLOORS = {"M5": 120, "M15": 60, "H1": 30}
MIN_OOS_TRADES_HARD = HARD_TRADE_FLOORS  # backward-compat alias

# Soft trade floors — trials below earn score × 0.1
SOFT_TRADE_FLOORS = {"H1": 35, "M15": 140, "M5": 350}
TF_MIN_OOS_TRADES = SOFT_TRADE_FLOORS    # backward-compat alias

# TF-keyed search space specs (for config hash)
SEARCH_SPACES = {
    "H1": {
        "obs_cov":           (0.5,   5.0,  "log"),
        "trans_cov":         (0.001, 0.03, "log"),
        "n_states":          (3,     4,    "int"),
        "max_depth":         (4,     8,    "int"),   # raised ceiling: deeper trees for regime features
        "reg_alpha":         (1e-6,  0.5,  "log"),
        "reg_lambda":        (0.01,  2.0,  "log"),   # lowered floor: allow hmm_state signal
        "min_child_weight":  (1,     50,   "int"),   # lowered floor: allow finer splits on small vol buckets
        "learning_rate":     (0.005, 0.15, "log"),
        "n_estimators":      (50,    400,  "int"),
        "subsample":         (0.5,   0.9,  "float"),
        "colsample_bytree":  (0.4,   0.9,  "float"),
        "gamma":             (1e-6,  0.3,  "log"),
        "scale_pos_weight":  (0.5,   2.0,  "log"),
    },
    "M15": {
        "obs_cov":           (0.5,   5.0,  "log"),
        "trans_cov":         (0.001, 0.03, "log"),
        "n_states":          (3,     4,    "int"),
        "max_depth":         (3,     7,    "int"),
        "reg_alpha":         (1e-6,  0.1,  "log"),   # lowered: allow hmm OHE signal through
        "reg_lambda":        (1e-6,  0.1,  "log"),   # lowered: allow hmm OHE signal through
        "min_child_weight":  (3,     30,   "int"),
        "learning_rate":     (0.005, 0.2,  "log"),
        "n_estimators":      (100,   500,  "int"),
        "subsample":         (0.5,   0.9,  "float"),
        "colsample_bytree":  (0.5,   1.0,  "float"),
        "gamma":             (1e-6,  0.3,  "log"),
        "scale_pos_weight":  (0.5,   2.0,  "log"),
    },
    "M5": {
        "obs_cov":           (0.05,  5.0,  "log"),
        "trans_cov":         (0.001, 0.1,  "log"),
        "n_states":          (3,     4,    "int"),
        "max_depth":         (2,     4,    "int"),
        "reg_alpha":         (1e-6,  0.1,  "log"),   # lowered: allow hmm OHE signal through
        "reg_lambda":        (1e-6,  0.1,  "log"),   # lowered: allow hmm OHE signal through
        "min_child_weight":  (5,     25,   "int"),
        "learning_rate":     (0.01,  0.15, "log"),
        "n_estimators":      (200,   600,  "int"),
        "subsample":         (0.55,  0.85, "float"),
        "colsample_bytree":  (0.4,   0.8,  "float"),
        "gamma":             (1e-6,  0.3,  "log"),
        "scale_pos_weight":  (0.5,   2.0,  "log"),
    },
}

MAX_FLOAT_DD      = 0.20
CPCV_MAX_FLOAT_DD = 0.20
PAYOFF_FLOOR_USD  = 0.035
RAM_HIGH_PCT      = 90
RAM_PAUSE_SEC     = 30

# ── CPCV params (retained for legacy / notebook usage) ────────────────────────
CPCV_PARAMS = {
    "H1":  {"n_splits": 5, "n_test_splits": 2, "embargo_bars": 24},
    "M15": {"n_splits": 5, "n_test_splits": 2, "embargo_bars": 96},
    "M5":  {"n_splits": 5, "n_test_splits": 2, "embargo_bars": 288},
}
CPCV_N_BLOCKS       = 4   # C(4,2)=6 paths — faster per trial, less strict than C(6,2)=15
CPCV_K_TEST         = 2
CPCV_TRIALS         = {"H1": 80, "M15": 120, "M5": 200}
CPCV_PURGE_BARS     = {"H1": 24, "M15": 96,  "M5": 288}
MIN_TRADES_PER_PATH = {"H1": 15, "M15": 60,  "M5": 100}


# ── Utility helpers ────────────────────────────────────────────────────────────

def _study_db(broker: str) -> str:
    return f"sqlite:///models/study_{broker}.db"


def _get_tier(balance: float) -> str:
    return "small" if balance <= SMALL_ACCOUNT_THRESHOLD else "growth"


def _study_name(base: str = "gold_regime_x", broker: str = "standard",
                tier: str = "small", tf: str = "H1") -> str:
    parts = [base, tier, broker]
    if tf.upper() != "H1":
        parts.append(tf.upper())
    return "_".join(parts)


def _align_hmm_states(model, states: np.ndarray) -> np.ndarray:
    """Remap a raw hmmlearn state array to canonical ordering (Bull=0, Bear=1, ...).

    fit_hmm + predict_states already handle alignment internally.
    This utility is for external callers (e.g. notebook) that hold a raw model.
    """
    order = np.argsort(model.means_[:, 0])[::-1]
    remap = {int(old): int(new) for new, old in enumerate(order)}
    return np.vectorize(remap.get)(states)


def _make_purged_inner_cv_splits(
    X: pd.DataFrame, n_splits: int, embargo_bars: int = 24
) -> list:
    """Return list of (train_idx, val_idx) with an embargo gap to prevent leakage.

    Standard TimeSeriesSplit has no embargo period, causing serial-correlation
    leakage between adjacent train and validation windows.  This splitter inserts
    a gap of ``embargo_bars`` rows between the end of each training fold and the
    start of its validation fold.
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    splits = []

    for i in range(1, n_splits + 1):
        val_start = i * fold_size
        val_end   = val_start + fold_size if i < n_splits else n

        # Everything before the validation block, minus the embargo gap
        train_end = max(0, val_start - embargo_bars)

        if train_end < 100:  # skip degenerate folds
            continue

        train_idx = np.arange(0, train_end)
        val_idx   = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))

    return splits


def _score_from_backtest(
    result:       dict,
    tf:           str   = "H1",
    account_size: float = 15.0,
    n_bars:       int   = 2160,
) -> float:
    """Compute Calmar-dominant composite score from a vectorized_backtest result dict.

    Formula (industry-aligned):
        Calmar_c × 0.45  +  Sharpe_c × 0.35  +  (PF−1)_c × 0.15  +  Edge_c × 0.05
        + consistency_bonus (M5/M15 only, max +0.30)

    Calmar = Annualised Return / Max Floating DD.  Annualisation makes the reward
    proportional to the deployment horizon rather than the raw OOS window return,
    which avoids favouring parameter sets that got lucky in a long OOS slice.
    All components are symmetrically clamped to prevent single-metric dominance.
    """
    total_return = result.get("total_return", 0.0)
    floating_dd  = result.get("floating_max_drawdown", result.get("max_drawdown", 0.0))
    sharpe       = result.get("sharpe_ratio", 0.0)
    pf           = result.get("profit_factor", 1.0)
    avg_payoff   = result.get("expected_payoff", result.get("avg_trade_pnl", 0.0))

    # Annualise the OOS return so Calmar is comparable across window lengths
    bpy        = BARS_PER_YEAR.get(tf.upper(), 8760)
    ann_return = total_return * (bpy / max(n_bars, 1))
    if floating_dd <= 0:
        calmar = 5.0 if ann_return > 0 else 0.0
    else:
        calmar = ann_return / floating_dd

    calmar_c = float(np.clip(calmar, -5.0, 5.0))
    sharpe_c = float(np.clip(sharpe, -3.0, 3.0))
    pf_norm  = float(np.clip(pf - 1.0, -2.0, 2.0))
    edge_c   = float(np.clip(avg_payoff / max(PAYOFF_FLOOR_USD, 1e-9), 0.0, 2.0))

    score = calmar_c * 0.45 + sharpe_c * 0.35 + pf_norm * 0.15 + edge_c * 0.05

    if tf.upper() in ("M5", "M15"):
        consistency = result.get("return_consistency", 0.0)
        score += float(np.clip(consistency, 0.0, 1.0)) * 0.30

    return score


def _score_result(result: dict, tier: str = None, broker: str = None, tf: str = "H1") -> float:
    """Backward-compatible scoring wrapper — delegates to _score_from_backtest."""
    return _score_from_backtest(result, tf=tf)


# ── Config hash guard ──────────────────────────────────────────────────────────

def _compute_config_hash(tf: str, broker: str, feature_cols: list) -> str:
    """MD5 hash of (OPTIMIZER_VERSION, tf, broker, feature_cols, n_states, IS ratio).

    Any change to these invalidates existing Optuna trials in the study DB.
    """
    tf_up = tf.upper()
    ss = SEARCH_SPACES.get(tf_up, SEARCH_SPACES["H1"])
    payload = json.dumps({
        "version":      OPTIMIZER_VERSION,
        "tf":           tf_up,
        "broker":       broker,
        "feature_cols": sorted(feature_cols),
        "n_states":     ss.get("n_states"),
        "is_ratio":     IS_OOS_SPLIT.get(tf_up, 0.8),
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _check_study_hash(study: optuna.Study, tf: str, broker: str, feature_cols: list) -> None:
    """Abort with exit(1) if study config hash does not match the current config.

    Prevents stale trials from biasing Optuna's TPE surrogate after search space
    or feature set changes. The hash is stored as a study user attribute on first run.
    """
    current = _compute_config_hash(tf, broker, feature_cols)
    stored  = study.user_attrs.get("config_hash")
    if stored is None:
        study.set_user_attr("config_hash", current)
        return
    if stored != current:
        logger.error(
            "Config hash mismatch for study '%s': stored=%s  current=%s\n"
            "Feature set or search space changed — delete models/study_%s.db "
            "and re-run --mode optimize to start fresh.",
            study.study_name, stored, current, broker,
        )
        sys.exit(1)


# ── Window-level fit + score ───────────────────────────────────────────────────

def _run_single_window(
    df_is:      pd.DataFrame,
    df_oos:     pd.DataFrame,
    n_states:   int,
    tf:         str,
    balance:    float,
    broker:     str,
    xgb_kwargs: dict,
    n_cv_folds: int = 2,
    oos_bars:   int = 2160,
) -> dict:
    """Fit HMM+XGBoost on IS, evaluate on OOS, return per-window metrics.

    Returns dict with keys:
        ok, error, oos_score, oos_sharpe, oos_n_trades, oos_fdd, is_cv_sharpe,
        hmm_zeroed
    """
    def _fail(msg):
        return {"ok": False, "error": msg, "oos_score": -50.0,
                "oos_sharpe": -10.0, "oos_n_trades": 0, "oos_fdd": 1.0,
                "is_cv_sharpe": 0.0, "hmm_zeroed": False}

    try:
        model_is, states_is, _ = fit_hmm(df_is, n_states=n_states, tf=tf)
    except Exception as exc:
        return _fail(f"HMM fit: {exc}")

    if min(model_is.transmat_[i, i] for i in range(n_states)) < 0.65:
        return _fail("degenerate HMM: min_persist < 0.65")

    try:
        states_oos = predict_states(model_is, df_oos)
    except Exception as exc:
        return _fail(f"OOS predict_states: {exc}")

    try:
        X_is, y_is, df_is_al, scaler_is = prepare_features(df_is, states_is, tf=tf)
        X_oos, _, df_oos_al, _          = prepare_features(
            df_oos, states_oos, feature_scaler=scaler_is, tf=tf
        )
    except Exception as exc:
        return _fail(f"prepare_features: {exc}")

    if len(X_is) < 500 or len(X_oos) < 50:
        return _fail(f"too few rows: IS={len(X_is)} OOS={len(X_oos)}")

    if list(X_oos.columns) != list(X_is.columns):
        return _fail(f"OOS/IS column mismatch: {list(X_oos.columns)} vs {list(X_is.columns)}")

    states_is_al  = states_is[df_is.index.isin(df_is_al.index)]
    states_oos_al = states_oos[df_oos.index.isin(df_oos_al.index)]

    # Inner IS CV (consistency check) — purged splitter to avoid leakage
    cv_sharpes: list = []
    tf_embargo = WFO_PARAMS.get(tf.upper(), WFO_PARAMS["H1"])["embargo_bars"]
    for _tr, _val in _make_purged_inner_cv_splits(X_is, n_cv_folds, embargo_bars=tf_embargo):
        X_tr, y_tr = X_is.iloc[_tr], y_is.iloc[_tr]
        X_val      = X_is.iloc[_val]
        if len(X_tr) < 200 or len(X_val) < 50:
            continue
        try:
            _m, _t, _ = train_xgb_ensemble(X_tr, y_tr, train_ratio=1.0, **xgb_kwargs)
            _, _p = get_predictions_ensemble(_m, _t, X_val)
            _dfv  = df_is_al[df_is_al.index.isin(X_val.index)]
            _stv  = states_is_al[df_is_al.index.isin(X_val.index)]
            if len(_dfv) < 20:
                continue
            _r = vectorized_backtest(
                _dfv, _p, _stv,
                split_idx=None, account_size=balance, broker=broker, tf=tf,
                hmm_transmat=model_is.transmat_,
            )
            cv_sharpes.append(_r.get("sharpe_ratio", 0.0))
        except Exception:
            pass

    mean_cv_sharpe = float(np.mean(cv_sharpes)) if cv_sharpes else 0.0

    try:
        models_w, thresh_w, ens_metrics = train_xgb_ensemble(X_is, y_is, train_ratio=1.0, **xgb_kwargs)
    except Exception as exc:
        return _fail(f"XGB train: {exc}")

    # Detect if XGBoost zeroed out hmm_state — a signal of over-regularisation
    # that breaks the HMM→XGB regime connection entirely.
    hmm_state_imp = ens_metrics.get("feature_importance", {}).get("hmm_state", 1.0)
    hmm_zeroed    = (float(hmm_state_imp) == 0.0)

    _, probs_oos = get_predictions_ensemble(models_w, thresh_w, X_oos)

    if len(df_oos_al) < 20 or len(probs_oos) != len(df_oos_al):
        return _fail(f"OOS length mismatch: df={len(df_oos_al)} probs={len(probs_oos)}")

    try:
        oos_result = vectorized_backtest(
            df_oos_al, probs_oos, states_oos_al,
            split_idx=None, account_size=balance, broker=broker, tf=tf,
            hmm_transmat=model_is.transmat_,
        )
    except Exception as exc:
        return _fail(f"OOS backtest: {exc}")

    n_trades   = oos_result.get("n_trades", 0)
    oos_fdd    = oos_result.get("floating_max_drawdown", oos_result.get("max_drawdown", 0.0))
    oos_sharpe = oos_result.get("sharpe_ratio", 0.0)

    if n_trades < HARD_TRADE_FLOORS.get(tf.upper(), 10) or oos_fdd > CPCV_MAX_FLOAT_DD:
        oos_score = -50.0
    else:
        oos_score = _score_from_backtest(
            oos_result, tf=tf, account_size=balance, n_bars=oos_bars
        )
        if hmm_zeroed:
            oos_score *= 0.5  # XGB ignoring HMM regime = signal quality failure
        if mean_cv_sharpe < -1.0:
            oos_score *= 0.5

    return {
        "ok":           True,
        "error":        None,
        "oos_score":    oos_score,
        "oos_sharpe":   oos_sharpe,
        "oos_n_trades": n_trades,
        "oos_fdd":      oos_fdd,
        "is_cv_sharpe": mean_cv_sharpe,
        "hmm_zeroed":   hmm_zeroed,
    }


def _run_wfo(
    df,
    n_states:     int,
    tf:           str,
    balance:      float,
    broker:       str,
    xgb_kwargs:   dict,
    is_bars:      int,
    oos_bars:     int,
    embargo_bars: int,
    step_bars:    int,
    wfo_mode:     str = "standard",
) -> dict:
    """Rolling Walk-Forward Optimization — slide IS/OOS windows across full df.

    df must already have kalman_return computed for this trial's obs_cov/trans_cov.
    HMM is fitted per window on IS data only (no future lookahead).

    Returns dict: wfo_score, n_windows, n_valid_windows, median_trades,
                  std_sharpe, window_scores, wfe_ratio
    """
    n               = len(df)
    window_scores:  list = []
    window_sharpes: list = []
    is_cv_sharpes:  list = []
    window_trades:  list = []
    n_valid_windows = 0
    n_cv_folds      = CV_FOLDS.get(tf.upper(), 2)

    # Pre-compute all valid window start positions, then skip the oldest ones
    # if the total exceeds MAX_WFO_WINDOWS.  Using the most-recent windows
    # avoids over-weighting old regimes and keeps trial cost bounded.
    all_starts: list[int] = []
    s = 0
    while s + is_bars + oos_bars + embargo_bars <= n:
        all_starts.append(s)
        s += step_bars
    max_wins = MAX_WFO_WINDOWS.get(tf.upper(), 12)
    if len(all_starts) > max_wins:
        all_starts = all_starts[-max_wins:]

    for start in all_starts:
        is_end    = start + is_bars
        oos_start = is_end + embargo_bars
        oos_end   = oos_start + oos_bars

        df_is  = df.iloc[start:is_end]
        df_oos = df.iloc[oos_start:min(oos_end, n)]

        if len(df_oos) < oos_bars // 2:
            continue

        win = _run_single_window(
            df_is=df_is, df_oos=df_oos,
            n_states=n_states, tf=tf,
            balance=balance, broker=broker,
            xgb_kwargs=xgb_kwargs,
            n_cv_folds=n_cv_folds,
            oos_bars=oos_bars,
        )

        # Skip windows that error out before producing any result
        if not win["ok"] and "degenerate" not in (win.get("error") or ""):
            logger.warning("WFO [%s] window start=%d skipped: %s", tf, start, win.get("error"))
            continue

        score = win["oos_score"]
        if score > -50.0:
            n_valid_windows += 1

        window_scores.append(score)
        window_sharpes.append(win["oos_sharpe"])
        window_trades.append(win["oos_n_trades"])
        is_cv_sharpes.append(win["is_cv_sharpe"])

        logger.info(
            "WFO [%s] window start=%d: trades=%d sharpe=%.3f cv_sharpe=%.3f score=%.3f",
            tf, start, win["oos_n_trades"], win["oos_sharpe"], win["is_cv_sharpe"], score,
        )

    if not window_scores:
        logger.warning(
            "WFO [%s]: ALL windows were skipped (no valid results). "
            "Check for exceptions in XGB training or backtester.", tf,
        )
        return {
            "wfo_score": -100.0, "n_windows": 0, "n_valid_windows": 0,
            "median_trades": 0, "std_sharpe": 0.0, "window_scores": [], "wfe_ratio": 0.0,
        }

    std_sharpe    = float(np.std(window_sharpes))
    median_trades = int(np.median(window_trades))
    mean_oos      = float(np.mean(window_sharpes))
    mean_is       = float(np.mean(is_cv_sharpes)) if is_cv_sharpes else 0.0
    # WFE (Walk-Forward Efficiency): OOS Sharpe / IS Sharpe — sign-preserving.
    if abs(mean_is) < 0.01:
        wfe_ratio = 0.0
    else:
        wfe_ratio = float(mean_oos / mean_is)

    valid_sc = [s for s in window_scores if s > -50.0]
    n_failed = len(window_scores) - len(valid_sc)

    if valid_sc:
        # Compute WFO score on VALID windows only — including -50.0 in std
        # inflates the penalty by 5-20× when even one window fails technically
        # (e.g. null XGB model → <20 trades → -50 floor).
        wfo_score = float(np.median(valid_sc)) - 0.20 * float(np.std(valid_sc))
        # Each hard-floor window still penalises the trial (0.5 per failure)
        if n_failed > 0:
            wfo_score -= n_failed * 0.5
    else:
        wfo_score = -100.0

    if valid_sc and all(s > 0.30 for s in valid_sc):
        wfo_score *= 1.05

    logger.info(
        "WFO [%s]: %d/%d valid | median_trades=%d | score=%.3f | "
        "std_sharpe=%.3f | WFE=%.2f | window_scores=%s",
        tf, n_valid_windows, len(window_scores), median_trades,
        wfo_score, std_sharpe, wfe_ratio,
        [round(s, 3) for s in window_scores],
    )
    return {
        "wfo_score":       wfo_score,
        "n_windows":       len(window_scores),
        "n_valid_windows": n_valid_windows,
        "median_trades":   median_trades,
        "std_sharpe":      std_sharpe,
        "window_scores":   window_scores,
        "wfe_ratio":       wfe_ratio,
    }


# ── Optuna objective ───────────────────────────────────────────────────────────

def make_objective(df: pd.DataFrame, tf: str, broker: str,
                   account_size: float = 15.0, wfo_mode: str = "standard"):
    """Return an Optuna objective that scores hyperparams via CPCV.

    Args:
        df:           Full processed parquet df (pre-loaded by caller).
                      Must contain 'log_return' for per-trial Kalman recomputation.
        tf:           Timeframe string.
        broker:       Broker name for cost model.
        account_size: Account balance in USD.
        wfo_mode:     Retained for backward-compat; unused with CPCV.

    Each trial only recomputes kalman_smooth (fast) instead of process_pipeline
    (slow), giving a 10-50× speedup on long M5/M15 studies.
    """
    tf_up = tf.upper()

    def objective(trial: optuna.Trial) -> float:
        # ── Kalman params ─────────────────────────────────────────────────────
        if tf_up == "M5":
            obs_cov   = trial.suggest_float("obs_cov",   0.05, 5.0,  log=True)
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.1, log=True)
        else:
            obs_cov   = trial.suggest_float("obs_cov",   0.5,  5.0,  log=True)
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.03, log=True)

        # ── n_states ──────────────────────────────────────────────────────────
        if tf_up in ("M5", "M15"):
            n_states = trial.suggest_int("n_states", 3, 4)
        else:  # H1
            n_states = trial.suggest_int("n_states", 3, 4)

        # ── XGBoost params ────────────────────────────────────────────────────
        if tf_up == "M5":
            max_depth        = trial.suggest_int("max_depth", 2, 4)
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-6, 0.1,  log=True)
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-6, 0.1,  log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 5, 25)
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
            n_estimators     = trial.suggest_int("n_estimators", 200, 600, step=50)
            subsample        = trial.suggest_float("subsample", 0.55, 0.85)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.8)
        elif tf_up == "H1":
            max_depth        = trial.suggest_int("max_depth", 4, 8)             # raised ceiling
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-6, 0.5,  log=True)
            reg_lambda       = trial.suggest_float("reg_lambda", 0.01, 2.0,  log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 50)     # lowered floor
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.15, log=True)
            n_estimators     = trial.suggest_int("n_estimators", 50, 400, step=50)
            subsample        = trial.suggest_float("subsample", 0.5, 0.9)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.9)
        else:  # M15
            max_depth        = trial.suggest_int("max_depth", 3, 7)
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-6, 0.1,  log=True)
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-6, 0.1,  log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 3, 30)
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
            n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=50)
            subsample        = trial.suggest_float("subsample", 0.5, 0.9)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        gamma            = trial.suggest_float("gamma", 1e-6, 0.3, log=True)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 2.0, log=True)

        try:
            # Recompute Kalman with trial params only — all other features stay from parquet
            kalman_col = kalman_smooth(df["log_return"].values, obs_cov, trans_cov)
            df_trial   = df.copy()
            df_trial["kalman_return"] = kalman_col

            xgb_kwargs = dict(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                scale_pos_weight=scale_pos_weight,
            )

            # Execute Combinatorial Purged CV
            tf_embargo = CPCV_PURGE_BARS.get(tf_up, 24)

            cpcv_result = execute_cpcv(
                df_full=df_trial,
                n_states=n_states,
                tf=tf_up,
                balance=account_size,
                broker=broker,
                xgb_kwargs=xgb_kwargs,
                n_splits=CPCV_N_BLOCKS,
                n_test_splits=CPCV_K_TEST,
                embargo_bars=tf_embargo,
            )

            score         = cpcv_result["cpcv_score"]
            n_valid_paths = cpcv_result["n_valid_paths"]
            std_sharpe    = cpcv_result["std_sharpe"]
            median_trades = cpcv_result["median_trades"]

            # Store metrics for Optuna history
            trial.set_user_attr("n_valid_paths", n_valid_paths)
            trial.set_user_attr("median_trades",   median_trades)
            trial.set_user_attr("std_sharpe",      std_sharpe)
            trial.set_user_attr("median_sharpe",   cpcv_result["median_sharpe"])

            logger.info(
                "Trial %d [%s/%s]: score=%.3f | valid_paths=%d/%d | "
                "std_sharpe=%.3f | median_trades=%d",
                trial.number, tf, broker, score, n_valid_paths, _N_PATHS,
                std_sharpe, median_trades,
            )
            append_trial_score(
                f"Trial {trial.number} [{tf}/{broker}]: score={score:.3f} | "
                f"valid_paths={n_valid_paths}/{_N_PATHS} | "
                f"std_sharpe={std_sharpe:.3f} | median_trades={median_trades}"
            )

            return score

        except Exception as exc:
            logger.warning("Trial %d failed: %s", trial.number, exc)
            return -100.0

        finally:
            gc.collect()

    return objective


# ── Progress callbacks ─────────────────────────────────────────────────────────

def _make_callbacks(total_target: int, study_name: str, already_done: int = 0) -> list:
    from src.notifier import send_telegram_msg

    start_time    = [time.time()]
    heartbeat_pct = set()

    def _callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        total_done   = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        session_done = max(0, total_done - already_done)
        if session_done <= 0:
            return

        if _PSUTIL_OK:
            mem = psutil.virtual_memory()
            if mem.percent >= RAM_HIGH_PCT:
                logger.warning("RAM at %.0f%% — pausing %ds.", mem.percent, RAM_PAUSE_SEC)
                time.sleep(RAM_PAUSE_SEC)

        if session_done % 5 == 0:
            elapsed   = time.time() - start_time[0]
            rate      = elapsed / session_done
            remaining = max(0, total_target - total_done)
            eta_sec   = remaining * rate
            if eta_sec >= 3600:
                eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            else:
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
            best    = study.best_value if study.best_trial else float("-inf")
            ram_str = (f"  RAM: {psutil.virtual_memory().percent:.0f}%" if _PSUTIL_OK else "")
            print(
                f"  [{total_done:>4}/{total_target}]  "
                f"Best Score: {best:+.3f}  |  ETA: {eta_str}{ram_str}"
            )

        if total_target > 0:
            milestone = (int(total_done / total_target * 100) // 10) * 10
            if milestone > 0 and milestone not in heartbeat_pct:
                heartbeat_pct.add(milestone)
                best = study.best_value if study.best_trial else float("-inf")
                send_telegram_msg(
                    f"Optimization <b>{milestone}%</b> complete\n"
                    f"Study: <code>{study_name}</code>\n"
                    f"Best Score: <b>{best:.3f}</b>  |  Trials: {total_done}/{total_target}"
                )

    return [_callback]


# ── Main optimization entry point ──────────────────────────────────────────────

def make_objective_stage1(
    df: pd.DataFrame,
    tf: str,
    broker: str,
    account_size: float = 15.0,
):
    """Single hold-out objective for Stage-1 (fast XGB exploration).

    Uses the last ``oos_bars`` of the dataset as a fixed hold-out instead of
    running all C(n_blocks, k_test) CPCV paths.  Approximately 5-10× faster
    per trial than the full CPCV objective — ideal for rapid search-space
    exploration before Stage-2 CPCV validation.
    """
    tf_up   = tf.upper()
    wfo_cfg = WFO_PARAMS.get(tf_up, WFO_PARAMS["H1"])
    oos_bars   = wfo_cfg["oos_bars"]
    n          = len(df)
    oos_start  = max(n - oos_bars, oos_bars)   # keep at least 1×oos for IS
    df_is_base = df.iloc[:oos_start]
    df_oos_base = df.iloc[oos_start:]
    n_cv_folds  = CV_FOLDS.get(tf_up, 2)

    def objective(trial: optuna.Trial) -> float:
        # ── Kalman params ─────────────────────────────────────────────────────
        if tf_up == "M5":
            obs_cov   = trial.suggest_float("obs_cov",   0.05, 5.0,  log=True)
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.1, log=True)
        else:
            obs_cov   = trial.suggest_float("obs_cov",   0.5,  5.0,  log=True)
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.03, log=True)

        n_states = trial.suggest_int("n_states", 3, 4)

        # ── XGBoost params (same ranges as full CPCV objective) ───────────────
        if tf_up == "M5":
            max_depth        = trial.suggest_int("max_depth", 2, 4)
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-6, 0.1,  log=True)
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-6, 0.1,  log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 5, 25)
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
            n_estimators     = trial.suggest_int("n_estimators", 200, 600, step=50)
            subsample        = trial.suggest_float("subsample", 0.55, 0.85)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.8)
        elif tf_up == "H1":
            max_depth        = trial.suggest_int("max_depth", 4, 8)
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-6, 0.5,  log=True)
            reg_lambda       = trial.suggest_float("reg_lambda", 0.01, 2.0,  log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 50)
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.15, log=True)
            n_estimators     = trial.suggest_int("n_estimators", 50, 400, step=50)
            subsample        = trial.suggest_float("subsample", 0.5, 0.9)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.9)
        else:  # M15
            max_depth        = trial.suggest_int("max_depth", 3, 7)
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-6, 0.1,  log=True)
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-6, 0.1,  log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 3, 30)
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
            n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=50)
            subsample        = trial.suggest_float("subsample", 0.5, 0.9)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        gamma            = trial.suggest_float("gamma", 1e-6, 0.3, log=True)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 2.0, log=True)

        # ── Recompute Kalman on each partition independently ──────────────────
        try:
            df_is  = df_is_base.copy()
            df_oos = df_oos_base.copy()
            df_is["kalman_return"]  = kalman_smooth(
                df_is["log_return"].values, obs_cov=obs_cov, trans_cov=trans_cov
            )
            df_oos["kalman_return"] = kalman_smooth(
                df_oos["log_return"].values, obs_cov=obs_cov, trans_cov=trans_cov
            )
        except Exception as exc:
            logger.warning("Stage1[%s] Kalman error trial %d: %s", tf, trial.number, exc)
            return -50.0

        xgb_kwargs = dict(
            max_depth=max_depth, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            min_child_weight=min_child_weight, learning_rate=learning_rate,
            n_estimators=n_estimators, subsample=subsample,
            colsample_bytree=colsample_bytree, gamma=gamma,
            scale_pos_weight=scale_pos_weight,
        )

        try:
            result = _run_single_window(
                df_is=df_is, df_oos=df_oos,
                n_states=int(n_states), tf=tf,
                balance=account_size, broker=broker,
                xgb_kwargs=xgb_kwargs,
                n_cv_folds=n_cv_folds,
                oos_bars=oos_bars,
            )
        except Exception as exc:
            logger.warning("Stage1[%s] trial %d error: %s", tf, trial.number, exc)
            return -50.0

        score = result.get("oos_score", -50.0)
        trial.set_user_attr("oos_sharpe", float(result.get("oos_sharpe", -10.0)))
        trial.set_user_attr("oos_trades", int(result.get("oos_n_trades", 0)))
        trial.set_user_attr("oos_fdd",    float(result.get("oos_fdd", 1.0)))
        return score

    return objective


def run_optimization_stage1(
    df:           pd.DataFrame,
    tf:           str,
    broker:       str,
    account_size: float = 15.0,
    n_trials:     int   = 60,
) -> optuna.Study:
    """Stage-1: fast single hold-out optimization (no CPCV).

    Run this before the full Stage-2 CPCV optimization to rapidly explore the
    hyperparameter landscape.  Each trial runs ~5-10× faster than CPCV
    because it evaluates a single IS/OOS split instead of C(4,2)=6 paths.

    On completion writes ``models/stage1_{tf}_{broker}.json`` with the best
    hyperparameters.  Stage-2 (``--stage trading``) automatically loads this
    file to warm-start Optuna's TPE sampler.

    Usage::

        python main.py --mode optimize --tf H1 --broker headway_cent \\
            --stage xgb --trials 60
        python main.py --mode optimize --tf H1 --broker headway_cent \\
            --stage trading --trials 130
    """
    from pathlib import Path

    tier    = _get_tier(account_size)
    name    = _study_name(broker=broker, tier=tier, tf=tf) + "_stage1"
    storage = _study_db(broker)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study  = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner,
    )

    already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining    = max(0, n_trials - already_done)

    if already_done > 0:
        pct = already_done / n_trials * 100
        print(
            f"\nResuming Stage-1: {already_done}/{n_trials} trials ({pct:.0f}%). "
            f"{remaining} remaining.\n"
        )
    else:
        print(
            f"\nStarting Stage-1 XGB study '{name}' — target {n_trials} trials.\n"
            f"Mode: single hold-out (no CPCV) — ~5x faster per trial than Stage-2.\n"
        )

    if remaining == 0:
        print("Stage-1 target already reached. Use higher --trials to continue.\n")
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            make_objective_stage1(df=df, tf=tf, broker=broker, account_size=account_size),
            n_trials=remaining,
            n_jobs=1,
            show_progress_bar=True,
            callbacks=_make_callbacks(n_trials, name, already_done=already_done),
        )

    # ── Persist best params for Stage-2 warm-start ───────────────────────────
    stage1_path = Path(f"models/stage1_{tf.lower()}_{broker}.json")
    stage1_path.parent.mkdir(parents=True, exist_ok=True)
    stage1_path.write_text(json.dumps({
        "params": study.best_params,
        "score":  study.best_value,
        "tf":     tf,
        "broker": broker,
    }, indent=2))

    logger.info(
        "Stage-1 complete: best=%.3f  saved to %s  params=%s",
        study.best_value, stage1_path, study.best_params,
    )
    print(
        f"\n{'*'*60}\n"
        f"  Stage-1 Complete\n"
        f"  Best score : {study.best_value:.3f}\n"
        f"  Saved to   : {stage1_path}\n"
        f"  Next step  : python main.py --mode optimize --tf {tf} "
        f"--broker {broker} --stage trading --trials 130\n"
        f"{'*'*60}\n"
    )
    return study


def run_optimization(
    df:           pd.DataFrame,
    tf:           str,
    broker:       str,
    account_size: float = 15.0,
    n_trials:     int   = 250,
    wfo_mode:     str   = "standard",
    n_jobs:       int   = 1,
    telegram_fn         = None,
    warm_start_stage1:  bool = False,   # enqueue Stage-1 best params as first trial
) -> optuna.Study:
    """Run (or resume) a rolling-WFO Optuna study and return the completed study.

    Args:
        df:           Pre-loaded processed parquet df (must have 'log_return').
        tf:           Timeframe.
        broker:       Broker name.
        account_size: Account balance in USD.
        n_trials:     Total trial target (studies resume — set higher to continue).
        wfo_mode:     "standard" or "fast".
        n_jobs:       Parallel workers (default 1; XGBoost already multi-threaded).
        telegram_fn:  Optional callable(message) — overrides src.notifier for tests.
    """
    tier    = _get_tier(account_size)
    name    = _study_name(broker=broker, tier=tier, tf=tf)
    storage = _study_db(broker)

    _wfo_map = WFO_PARAMS_FAST if wfo_mode == "fast" else WFO_PARAMS
    wfo_cfg  = _wfo_map.get(tf.upper(), WFO_PARAMS["H1"])

    if n_trials == 250:
        n_trials = WFO_TRIALS.get(tf.upper(), 60)

    n_jobs = 1  # always sequential within a trial; XGBoost is already multi-threaded

    os.makedirs("models", exist_ok=True)

    if tf.upper() == "M5":
        pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=9, reduction_factor=3)
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner,
    )

    feature_cols = list(get_feature_cols(df))
    _check_study_hash(study, tf=tf, broker=broker, feature_cols=feature_cols)

    # ── Stage-2 warm-start: seed Optuna TPE with Stage-1 best params ─────────
    # When --stage trading is passed, any previously saved Stage-1 JSON is
    # enqueued as the first trial so TPE explores the confirmed-good region
    # immediately rather than burning trials on cold-start random sampling.
    if warm_start_stage1:
        from pathlib import Path
        _s1_path = Path(f"models/stage1_{tf.lower()}_{broker}.json")
        if _s1_path.exists():
            try:
                _s1 = json.loads(_s1_path.read_text())
                study.enqueue_trial(_s1["params"])
                logger.info(
                    "Stage-2: warm-started from %s (stage1 score=%.3f)",
                    _s1_path, _s1.get("score", float("nan")),
                )
                print(f"  Warm-starting from Stage-1 params (score={_s1.get('score', '?'):.3f}).\n")
            except Exception as _e:
                logger.warning("Failed to load Stage-1 params for warm-start: %s", _e)
        else:
            logger.warning(
                "Stage-2 warm_start_stage1=True but no file found at %s. "
                "Run --stage xgb first.", _s1_path,
            )

    already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining    = max(0, n_trials - already_done)

    if already_done > 0:
        pct = already_done / n_trials * 100
        print(
            f"\nResuming: {already_done}/{n_trials} trials in study_{broker}.db "
            f"({pct:.0f}%). {remaining} remaining.\n"
        )
    else:
        print(
            f"\nStarting WFO study '{name}' — target {n_trials} trials.\n"
            f"WFO [{tf}]: IS={wfo_cfg['is_bars']} bars  OOS={wfo_cfg['oos_bars']} bars  "
            f"embargo={wfo_cfg['embargo_bars']} bars  step={wfo_cfg['step_bars']} bars\n"
            f"Recommended trial counts — H1:{WFO_TRIALS['H1']}  "
            f"M15:{WFO_TRIALS['M15']}  M5:{WFO_TRIALS['M5']}\n"
        )

    if remaining == 0:
        print("Target already reached. Use higher --trials to continue.\n")
        return study

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        make_objective(df=df, tf=tf, broker=broker,
                       account_size=account_size, wfo_mode=wfo_mode),
        n_trials=remaining,
        n_jobs=n_jobs,
        show_progress_bar=(n_jobs == 1),
        callbacks=_make_callbacks(n_trials, name, already_done=already_done),
    )

    best       = study.best_value
    total_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    logger.info(
        "Optimization done: best=%.3f  total=%d  params=%s",
        best, total_done, study.best_params,
    )

    pad = 46
    print(
        "\n************************************************************\n"
        "*                                                          *\n"
        "*              OPTIMIZATION COMPLETE                       *\n"
        f"*   Study : {name:<{pad}}*\n"
        f"*   Trials: {total_done:<{pad}}*\n"
        f"*   Best  : {best:<+{pad}.4f}*\n"
        "*                                                          *\n"
        "************************************************************\n"
    )

    from src.notifier import send_telegram_msg as _tg
    _notify = telegram_fn if telegram_fn is not None else _tg
    _notify(
        f"<b>Optimization 100% Complete!</b>\n"
        f"Study: <code>{name}</code>\n"
        f"Trials: <b>{total_done}</b> | Best: <b>{best:.3f}</b>\n"
        + "\n".join(f"  <code>{k}</code>: {v}" for k, v in study.best_params.items())
    )

    # Persist best-trial CPCV stats so --mode report can display them directly.
    # These are the ground-truth OOS metrics recorded during optimisation.
    try:
        import json as _json
        from pathlib import Path as _Path
        _tf_up  = tf.upper()
        _bt     = study.best_trial
        _attrs  = _bt.user_attrs  # {'median_sharpe', 'median_trades', 'n_valid_paths', 'std_sharpe'}
        _cpcv_data = {
            "cpcv_score":      float(best),
            "n_valid_paths":   int(_attrs.get("n_valid_paths", 0)),
            "median_sharpe":   float(_attrs.get("median_sharpe", 0.0)),
            "std_sharpe":      float(_attrs.get("std_sharpe", 0.0)),
            "median_trades":   int(_attrs.get("median_trades", 0)),
            # These aren't stored per-trial; leave as 0.0 to be filled by --mode wfa
            "median_win_rate": 0.0,
            "median_drawdown": 0.0,
            "median_return":   0.0,
            "path_scores":     [],
        }
        os.makedirs("reports", exist_ok=True)
        _cpcv_path = _Path(f"reports/cpcv_{_tf_up.lower()}_{broker}.json")
        _cpcv_path.write_text(_json.dumps(_cpcv_data, indent=2))
        logger.info(
            "Best-trial CPCV stats saved → %s  (median_sharpe=%.3f  median_trades=%d)",
            _cpcv_path, _cpcv_data["median_sharpe"], _cpcv_data["median_trades"],
        )
    except Exception as _save_err:
        logger.warning("Could not save CPCV stats after optimisation: %s", _save_err)

    return study


# ── Param extraction ───────────────────────────────────────────────────────────

def save_best_trial_cpcv(
    broker: str,
    tf: str,
    balance: float = 15.0,
) -> bool:
    """Extract and save the best trial's CPCV stats to reports/cpcv_{tf}_{broker}.json.

    Used to backfill the CPCV JSON from an existing completed study without
    re-running optimisation.  Returns True on success, False if study is empty.
    """
    import json as _json
    from pathlib import Path as _Path

    tier    = _get_tier(balance)
    name    = _study_name(broker=broker, tier=tier, tf=tf)
    storage = _study_db(broker)

    try:
        _study = optuna.load_study(study_name=name, storage=storage)
    except Exception as _e:
        logger.warning("Could not load study '%s': %s", name, _e)
        return False

    if not _study.best_trial:
        logger.warning("Study '%s' has no complete trials.", name)
        return False

    _bt    = _study.best_trial
    _attrs = _bt.user_attrs
    _data  = {
        "cpcv_score":      float(_study.best_value),
        "n_valid_paths":   int(_attrs.get("n_valid_paths", 0)),
        "median_sharpe":   float(_attrs.get("median_sharpe", 0.0)),
        "std_sharpe":      float(_attrs.get("std_sharpe", 0.0)),
        "median_trades":   int(_attrs.get("median_trades", 0)),
        "median_win_rate": 0.0,
        "median_drawdown": 0.0,
        "median_return":   0.0,
        "path_scores":     [],
    }
    os.makedirs("reports", exist_ok=True)
    _path = _Path(f"reports/cpcv_{tf.lower()}_{broker}.json")
    _path.write_text(_json.dumps(_data, indent=2))
    logger.info(
        "CPCV stats saved from best trial → %s  (score=%.3f  median_sharpe=%.3f  trades=%d)",
        _path, _data["cpcv_score"], _data["median_sharpe"], _data["median_trades"],
    )
    return True


def get_best_params(balance: float = 15.0, broker: str = "standard", tf: str = "H1") -> dict:
    """Load best hyperparameters from the persisted SQLite study.

    Fallback chain: tier-matched study → small-tier study → legacy study.db.
    """
    tier    = _get_tier(balance)
    name    = _study_name(broker=broker, tier=tier, tf=tf)
    storage = _study_db(broker)

    def _try_load(db: str) -> dict | None:
        try:
            return optuna.load_study(study_name=name, storage=db).best_params
        except Exception:
            return None

    params = _try_load(storage)
    if params is not None:
        return params

    _legacy = "sqlite:///models/study.db"
    if storage != _legacy and os.path.exists("models/study.db"):
        params = _try_load(_legacy)
        if params is not None:
            logger.info("Loaded params from legacy study.db for '%s'.", name)
            return params

    if tier == "small":
        raise KeyError(f"No Optuna study '{name}' found in {storage}")

    fallback = _study_name(broker=broker, tier="small", tf=tf)
    logger.info("Growth study not found — using small-tier '%s'.", fallback)
    return optuna.load_study(study_name=fallback, storage=storage).best_params


def extract_consensus_params(
    tf:      str,
    broker:  str,
    top_n:   int   = 10,
    min_wfe: float = 0.0,
) -> dict:
    """Extract consensus hyperparameters from top-N trials (median per param).

    More robust than the single best trial: median-of-top-N reflects what
    consistently worked, filtering out lucky outliers from a single run.

    Args:
        tf:      Timeframe.
        broker:  Broker name.
        top_n:   Number of top-scoring trials to aggregate.
        min_wfe: Filter trials below this walk-forward efficiency ratio.
                 Trials without wfe_ratio stored (older studies) are included.

    Returns:
        Dict of {param: median_value} plus a 'meta' key with diagnostics.
    """
    storage = _study_db(broker)
    study   = None
    for balance in (15.0, 100.0):
        tier = _get_tier(balance)
        name = _study_name(broker=broker, tier=tier, tf=tf)
        try:
            study = optuna.load_study(study_name=name, storage=storage)
            break
        except Exception:
            continue

    if study is None:
        raise KeyError(f"No study found for tf={tf} broker={broker} in {storage}")

    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise ValueError(f"No completed trials in study '{study.study_name}'")

    if min_wfe > 0:
        filtered = [t for t in complete if t.user_attrs.get("wfe_ratio", min_wfe) >= min_wfe]
        if filtered:
            complete = filtered

    ranked = sorted(complete, key=lambda t: t.value, reverse=True)[:top_n]

    consensus: dict = {}
    for p in ranked[0].params:
        vals = [t.params[p] for t in ranked if p in t.params]
        if not vals:
            continue
        if isinstance(vals[0], (int, float)):
            med = float(np.median(vals))
            consensus[p] = int(round(med)) if isinstance(vals[0], int) else med
        else:
            consensus[p] = Counter(vals).most_common(1)[0][0]

    wfes = [t.user_attrs.get("wfe_ratio", 0.0) for t in ranked]
    consensus["meta"] = {
        "top_n_actual": len(ranked),
        "min_score":    float(ranked[-1].value),
        "max_score":    float(ranked[0].value),
        "mean_wfe":     float(np.mean(wfes)),
        "study_name":   study.study_name,
    }

    logger.info(
        "Consensus [%s/%s]: top-%d | score_range=[%.3f, %.3f] | mean_WFE=%.2f",
        tf, broker, len(ranked), ranked[-1].value, ranked[0].value, float(np.mean(wfes)),
    )
    return consensus


# ── WFA post-hoc analysis ──────────────────────────────────────────────────────

def run_wfa(
    df:           pd.DataFrame,
    tf:           str,
    broker:       str,
    account_size: float = 15.0,
    wfo_mode:     str   = "standard",
    save_plots:   bool  = False,
) -> dict:
    """Run CPCV analysis using current best params — for post-hoc reporting.

    Uses the already-optimised params from the study DB to evaluate all
    C(6,2)=15 combinatorial train/test paths and return aggregate diagnostics.

    Returns the execute_cpcv result dict plus backward-compat WFO key aliases
    (wfo_score, n_windows, n_valid_windows, wfe_ratio, window_scores) so that
    the existing main.py WFA display code continues to work unchanged.
    """
    params    = get_best_params(balance=account_size, broker=broker, tf=tf)
    obs_cov   = params.get("obs_cov",   1.0)
    trans_cov = params.get("trans_cov", 0.01)
    n_states  = params.get("n_states",  3)
    if isinstance(n_states, float):
        n_states = int(round(n_states))

    kalman_col = kalman_smooth(df["log_return"].values, obs_cov, trans_cov)
    df_trial   = df.copy()
    df_trial["kalman_return"] = kalman_col

    xgb_kwargs = {k: v for k, v in params.items()
                  if k not in ("obs_cov", "trans_cov", "n_states")}

    tf_embargo = CPCV_PURGE_BARS.get(tf.upper(), 24)
    result = execute_cpcv(
        df_full=df_trial,
        n_states=n_states,
        tf=tf,
        balance=account_size,
        broker=broker,
        xgb_kwargs=xgb_kwargs,
        n_splits=CPCV_N_BLOCKS,
        n_test_splits=CPCV_K_TEST,
        embargo_bars=tf_embargo,
    )

    result["params"] = params
    # Backward-compat aliases so main.py --mode wfa display works unchanged
    result["wfo_score"]       = result["cpcv_score"]
    result["n_windows"]       = _N_PATHS
    result["n_valid_windows"] = result["n_valid_paths"]
    result["wfe_ratio"]       = 0.0           # not applicable in CPCV
    result["window_scores"]   = result.get("path_scores", [])

    # Persist CPCV summary so --mode report can load it as primary OOS metrics
    import json as _json
    from pathlib import Path as _Path
    _cpcv_path = _Path(f"reports/cpcv_{tf.lower()}_{broker}.json")
    _cpcv_path.parent.mkdir(parents=True, exist_ok=True)
    _saveable = {k: v for k, v in result.items() if k != "params"}
    # Convert any numpy types to native Python for JSON serialisation
    def _to_native(obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, list):            return [_to_native(x) for x in obj]
        return obj
    _cpcv_path.write_text(_json.dumps({k: _to_native(v) for k, v in _saveable.items()}, indent=2))
    logger.info("CPCV report saved → %s", _cpcv_path)

    return result


# ── CPCV (retained for legacy / notebook / sensitivity usage) ─────────────────

class CPCVSplitter:
    """Generate train/test mask pairs via Combinatorial Purged CV.

    Splits the timeline into n_blocks equal chronological blocks and yields all
    C(n_blocks, k_test) train/test mask pairs with a purge_bars gap at each boundary.
    """

    def __init__(self, n: int, n_blocks: int = CPCV_N_BLOCKS,
                 k_test: int = CPCV_K_TEST, purge_bars: int = 24):
        self.n          = n
        self.n_blocks   = n_blocks
        self.k_test     = k_test
        self.purge_bars = purge_bars
        self._blocks    = np.array_split(np.arange(n), n_blocks)

    def split(self):
        for test_combo in combinations(range(self.n_blocks), self.k_test):
            train_mask = np.ones(self.n, dtype=bool)
            test_mask  = np.zeros(self.n, dtype=bool)

            for blk_idx in test_combo:
                blk = self._blocks[blk_idx]
                test_mask[blk] = True
                lo = max(0, int(blk[0])  - self.purge_bars)
                hi = min(self.n, int(blk[-1]) + 1 + self.purge_bars)
                train_mask[lo:hi] = False

            train_mask[test_mask] = False
            yield train_mask, test_mask


def _n_paths_total() -> int:
    from math import comb
    return comb(CPCV_N_BLOCKS, CPCV_K_TEST)


_N_PATHS = _n_paths_total()


def execute_cpcv(
    df_full: pd.DataFrame,
    n_states: int,
    tf: str,
    balance: float,
    broker: str,
    xgb_kwargs: dict,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_bars: int = 24,
) -> dict:
    """Execute Combinatorial Purged CV with strict per-path HMM and Scaler fitting.

    For each of C(n_splits, n_test_splits) combinatorial train/test mask pairs:
      1. HMM is fitted *inside* the loop on purged training blocks only — no
         lookahead leakage from a globally pre-fitted model.
      2. StandardScaler is fitted on training rows only.
      3. XGBoost ensemble is trained on train_ratio=1.0 (CPCV owns the split).
      4. OOS backtest is run on the test blocks.

    Returns a dict with cpcv_score, n_valid_paths, median_sharpe, std_sharpe,
    median_trades, and path_scores.
    """
    splitter = CPCVSplitter(
        n=len(df_full), n_blocks=n_splits, k_test=n_test_splits,
        purge_bars=embargo_bars,
    )

    path_scores:    list = []
    path_sharpes:   list = []
    path_trades:    list = []
    path_winrates:  list = []
    path_drawdowns: list = []
    path_returns:   list = []
    n_valid_paths = 0
    min_trades = HARD_TRADE_FLOORS.get(tf.upper(), 10)

    for path_idx, (train_mask, test_mask) in enumerate(splitter.split()):
        try:
            df_train = df_full[train_mask]
            if len(df_train) < 500:
                continue

            # 1. Fit HMM STRICTLY on the purged training blocks (no lookahead)
            model_path, train_states, _ = fit_hmm(df_train, n_states=n_states, tf=tf)
            if min(model_path.transmat_[i, i] for i in range(n_states)) < 0.60:
                continue

            # 2. Predict states for the full dataset using the IS model
            all_states = predict_states(model_path, df_full)

            # 3. Fit scaler ONLY on the training data, then transform everything
            _, _, _, scaler = prepare_features(
                df_train, train_states, feature_scaler=None, tf=tf
            )
            X_all, y_all, df_all_aligned, _ = prepare_features(
                df_full, all_states, feature_scaler=scaler, tf=tf
            )

            # Align masks with the valid feature rows
            _kept = df_full.index.isin(df_all_aligned.index)
            train_mask_al = train_mask[_kept]
            test_mask_al  = test_mask[_kept]

            if train_mask_al.sum() < 100 or test_mask_al.sum() < 50:
                continue

            # 4. Train XGBoost on the training paths (train_ratio=1.0)
            X_tr, y_tr = X_all.iloc[train_mask_al], y_all.iloc[train_mask_al]
            models_ens, thresholds, ens_metrics = train_xgb_ensemble(
                X_tr, y_tr, train_ratio=1.0, **xgb_kwargs
            )

            # Penalise immediately if the model zeroed out hmm_state
            if float(ens_metrics.get("feature_importance", {}).get("hmm_state", 1.0)) == 0.0:
                return {
                    "cpcv_score": -50.0, "n_valid_paths": 0,
                    "median_sharpe": -10.0, "std_sharpe": 0.0,
                    "median_trades": 0, "path_scores": [],
                }

            # 5. Get predictions and backtest on the OOS test paths
            _, probs_all = get_predictions_ensemble(models_ens, thresholds, X_all)
            states_all_al = df_all_aligned["hmm_state"].values

            test_df     = df_all_aligned.iloc[test_mask_al]
            test_probs  = probs_all[test_mask_al]
            test_states = states_all_al[test_mask_al]

            # ── Regime oscillation guard ──────────────────────────────────
            # If the HMM state sequence has more than 20% transition bars
            # the path is degenerate — too-rapid regime switching causes
            # trade churn even with MIN_CONFIRMATION_BARS=5.
            # 20% = regime changes on 1 in 5 bars; 5% is typical for H1.
            # Raised from 12%: the tighter threshold was discarding too many
            # valid paths from legitimate volatile periods.
            if len(test_states) > 1:
                _n_trans = int(np.sum(np.diff(test_states) != 0))
                if _n_trans / len(test_states) > 0.20:
                    logger.debug(
                        "CPCV path %d: high oscillation rate %.1f%% — skipped",
                        path_idx + 1, _n_trans / len(test_states) * 100,
                    )
                    continue

            path_result = vectorized_backtest(
                test_df, test_probs, test_states,
                split_idx=None, account_size=balance, broker=broker, tf=tf,
                hmm_transmat=model_path.transmat_,
            )

            n_trades = path_result.get("n_trades", 0)
            fdd = path_result.get("floating_max_drawdown", path_result.get("max_drawdown", 0.0))

            if n_trades < min_trades or fdd > CPCV_MAX_FLOAT_DD:
                path_scores.append(-50.0)
                path_sharpes.append(-10.0)
                path_trades.append(n_trades)
                path_winrates.append(0.0)
                path_drawdowns.append(float(fdd))
                path_returns.append(0.0)
                continue

            n_valid_paths += 1
            path_scores.append(_score_result(path_result, broker=broker, tf=tf))
            path_sharpes.append(path_result.get("sharpe_ratio", 0.0))
            path_trades.append(n_trades)
            path_winrates.append(float(path_result.get("win_rate", 0.0)))
            path_drawdowns.append(float(fdd))
            path_returns.append(float(path_result.get("total_return", 0.0)))

        except Exception as exc:
            logger.debug("CPCV path %d failed: %s", path_idx + 1, exc)
            continue

    # ── Progressive CPCV Scoring ─────────────────────────────────────────────
    # 1. Hard Floor: reject outright if fewer than 4/6 paths survived.
    #    Below this threshold the gradient is meaningless for Optuna.
    if n_valid_paths < 4:
        logger.warning(
            "CPCV [%s]: only %d/%d valid paths — hard floor, rejecting trial.",
            tf, n_valid_paths, int(comb(CPCV_N_BLOCKS, CPCV_K_TEST)),
        )
        return {
            "cpcv_score": -50.0, "n_valid_paths": n_valid_paths,
            "median_sharpe": -10.0, "std_sharpe": 0.0,
            "median_trades": int(np.median(path_trades) if path_trades else 0),
            "path_scores": path_scores,
        }

    # 2. Base Score: median across ALL path scores (including -50 sentinels for
    #    invalid paths) minus a variance penalty.  Sentinel values naturally
    #    drag the median downward when paths fail, complementing the explicit
    #    path penalty below.
    _med_score = float(np.median(path_scores))
    _std_score = float(np.std(path_scores))
    base_score = _med_score - (0.25 * _std_score)

    # 3. Path Penalty: subtract 2.0 per missing path to force Optuna to seek
    #    6/6.  A 4/6 trial incurs −4.0; a 5/6 trial incurs −2.0; 6/6 is 0.0.
    path_penalty = (6 - n_valid_paths) * 2.0
    final_score  = base_score - path_penalty

    valid_sharpes = [s for s in path_sharpes if s > -9.0]
    logger.info(
        "CPCV [%s]: %d/%d valid paths | median_score=%.3f std=%.3f | "
        "path_penalty=%.1f | final_score=%.3f",
        tf, n_valid_paths, int(comb(CPCV_N_BLOCKS, CPCV_K_TEST)),
        _med_score, _std_score, path_penalty, final_score,
    )
    return {
        "cpcv_score":      final_score,
        "n_valid_paths":   n_valid_paths,
        "median_sharpe":   float(np.median(valid_sharpes)) if valid_sharpes else -10.0,
        "std_sharpe":      float(np.std(valid_sharpes))    if len(valid_sharpes) > 1 else 0.0,
        "median_trades":   int(np.median(path_trades) if path_trades else 0),
        "median_win_rate": float(np.median(path_winrates) if path_winrates else 0.0),
        "median_drawdown": float(np.median(path_drawdowns) if path_drawdowns else 0.0),
        "median_return":   float(np.median(path_returns) if path_returns else 0.0),
        "path_scores":     path_scores,
        "path_sharpes":    path_sharpes,
        "path_trades":     path_trades,
        "path_winrates":   path_winrates,
        "path_drawdowns":  path_drawdowns,
        "path_returns":    path_returns,
    }


def compute_cpcv_score(
    df_full,
    balance: float,
    broker:  str,
    tf:      str,
    n_states: int,
    xgb_kwargs: dict,
) -> float:
    """Backward-compat shim — delegates to execute_cpcv.

    Retained so that existing notebook calls remain valid.
    """
    result = execute_cpcv(
        df_full=df_full, n_states=n_states, tf=tf,
        balance=balance, broker=broker, xgb_kwargs=xgb_kwargs,
        n_splits=CPCV_N_BLOCKS, n_test_splits=CPCV_K_TEST,
        embargo_bars=CPCV_PURGE_BARS.get(tf.upper(), 24),
    )
    return result["cpcv_score"]



