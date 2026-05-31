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

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from src.processor import kalman_smooth
from src.engine_hmm import fit_hmm, predict_states, STATE_NAMES
from src.engine_xgb import (
    prepare_features, train_xgb_ensemble, get_predictions_ensemble,
    compute_regime_stats, get_feature_cols,
    train_regime_models, get_regime_predictions,
)
from src.backtester import vectorized_backtest
from src.cpcv_reporting import build_cpcv_report, write_cpcv_report, write_legacy_cpcv_report
from src.optuna_param_guard import audit_search_space, search_space_as_text
from src.regime_diagnostics import (
    summarize_regimes,
    get_occupancy_percentage,
    write_regime_diagnostics,
)
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
        "n_states":          (3,     3,    "int"),  # canonical: always 3
        "max_depth":         (3,     6,    "int"),   # spec: 3..6
        "reg_alpha":         (1e-6,  0.5,  "log"),
        "reg_lambda":        (0.01,  2.0,  "log"),
        "min_child_weight":  (1,     50,   "int"),
        "learning_rate":     (0.01,  0.08, "log"),   # spec: 0.01..0.08
        "n_estimators":      (200,   1200, "int"),   # spec: 200..1200
        "subsample":         (0.5,   0.9,  "float"),
        "colsample_bytree":  (0.4,   0.9,  "float"),
        "gamma":             (1e-6,  0.3,  "log"),
        "scale_pos_weight":  (0.5,   2.0,  "log"),
    },
    "M15": {
        "obs_cov":           (0.5,   5.0,  "log"),
        "trans_cov":         (0.001, 0.03, "log"),
        "n_states":          (3,     3,    "int"),  # canonical: always 3
        "max_depth":         (3,     8,    "int"),   # spec: 3..8
        "reg_alpha":         (1e-6,  0.1,  "log"),
        "reg_lambda":        (1e-6,  0.1,  "log"),
        "min_child_weight":  (3,     30,   "int"),
        "learning_rate":     (0.01,  0.15, "log"),   # spec: 0.01..0.15
        "n_estimators":      (300,   1500, "int"),   # spec: 300..1500
        "subsample":         (0.5,   0.9,  "float"),
        "colsample_bytree":  (0.5,   1.0,  "float"),
        "gamma":             (1e-6,  0.3,  "log"),
        "scale_pos_weight":  (0.5,   2.0,  "log"),
    },
    "M5": {
        "obs_cov":           (0.05,  5.0,  "log"),
        "trans_cov":         (0.001, 0.1,  "log"),
        "n_states":          (3,     3,    "int"),  # canonical: always 3
        "max_depth":         (2,     5,    "int"),   # spec: 2..5
        "reg_alpha":         (1e-6,  0.1,  "log"),
        "reg_lambda":        (1e-6,  0.1,  "log"),
        "min_child_weight":  (5,     25,   "int"),
        "learning_rate":     (0.005, 0.05, "log"),   # spec: 0.005..0.05
        "n_estimators":      (500,   2500, "int"),   # spec: 500..2500
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

# Validation and deployment thresholds are fixed governance constants.
VALIDATION_CONFIG = {
    "H1": {"min_median_sharpe": 0.25, "min_median_pf": 1.10, "max_trades_per_100": 8},
    "M15": {"min_median_sharpe": 0.20, "min_median_pf": 1.05, "max_trades_per_100": 20},
    "M5": {"min_median_sharpe": 0.10, "min_median_pf": 1.00, "max_trades_per_100": 35},
}


# ── Utility helpers ────────────────────────────────────────────────────────────

def _study_db(broker: str) -> str:
    return f"sqlite:///models/study_{broker}.db"


def _study_db_tf(broker: str, tf: str) -> str:
    """Per-TF SQLite DB for the unified pipeline optimizer."""
    return f"sqlite:///models/study_{broker}_{tf.lower()}.db"


def get_optuna_storage_url(broker: str) -> str:
    """Canonical per-broker Optuna storage URL (all TF studies in one DB).

    All studies for a broker are colocated so optuna-dashboard can display
    every active TF study from a single --storage argument.
    """
    return f"sqlite:///models/study_{broker}.db"


def get_optuna_study_name(tf: str, broker: str) -> str:
    """Canonical unified-pipeline study name for a given TF + broker pair."""
    return f"{tf.lower()}_{broker}_single_pipeline"


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


def _aligned_states_from_prepare(
    df_full: pd.DataFrame,
    all_states: np.ndarray,
    df_all_aligned: pd.DataFrame,
) -> np.ndarray:
    """Return states aligned 1:1 with df_all_aligned index.

    Works correctly for H1 (integer hmm_state), M15, and M5 (OHE state_*)
    because it filters directly on the original states array by index position,
    independent of whether hmm_state is present or OHE-encoded in df_all_aligned.

    Raises ValueError when alignment length does not match, preventing silent
    row-count mismatches in CPCV path scoring.
    """
    aligned_mask = df_full.index.isin(df_all_aligned.index)
    aligned_states = np.asarray(all_states)[aligned_mask]
    if len(aligned_states) != len(df_all_aligned):
        raise ValueError(
            f"State alignment mismatch: states={len(aligned_states)} "
            f"rows={len(df_all_aligned)}"
        )
    return aligned_states


def resolve_n_states(tf: str, params: dict) -> int:
    """Return the canonical n_states for *tf*.

    The canonical regime contract mandates exactly 3 states for all TFs:
    TREND (0), MEAN_REVERSION (1), VOLATILITY_SHOCK (2).
    Any other value is rejected with a ValueError.
    """
    requested = int(params.get("n_states", 3))
    if requested != 3:
        raise ValueError(
            f"n_states must be 3 for all TFs (canonical regime contract). "
            f"Received {requested} for {tf}. "
            f"Supported states: TREND(0), MEAN_REVERSION(1), VOLATILITY_SHOCK(2)."
        )
    return 3


def _enforce_three_states(params: dict, context: str) -> int:
    """Validate and return n_states=3, raising immediately on violation.

    Use this at every fit_hmm call site so 4-state drift cannot slip
    through without an explicit error and traceback.
    """
    n = int(params.get("n_states", 3))
    if n != 3:
        raise ValueError(
            f"{context}: n_states must be 3 (canonical contract), got {n}. "
            f"Delete or reset the Optuna study if old 4/5-state trials are stored."
        )
    return 3


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


def composite_score(metrics: dict) -> float:
    """Weighted composite CPCV objective score (Phase E rebuild).

    Formula:
        0.35 * deflated_sharpe
      + 0.25 * calmar
      + 0.20 * profit_factor    (raw, not -1)
      + 0.10 * expectancy
      + 0.10 * stability_score
      - penalties

    Penalties:
        trade_count_variance  — variance in per-fold trade counts (normalised)
        fold_instability      — std-dev of per-fold scores
        regime_instability    — penalise if MR leakage > 0
    """
    deflated_sharpe = float(metrics.get("deflated_sharpe", metrics.get("sharpe", 0.0)))
    calmar          = float(metrics.get("calmar", 0.0))
    pf              = float(metrics.get("profit_factor", 1.0))
    expectancy      = float(metrics.get("expectancy", 0.0))
    stability       = float(metrics.get("stability_score",
                             metrics.get("return_consistency", 0.0)))

    base = (
        0.35 * np.clip(deflated_sharpe, -5.0, 5.0)
      + 0.25 * np.clip(calmar,          -5.0, 5.0)
      + 0.20 * np.clip(pf,               0.0, 5.0)
      + 0.10 * np.clip(expectancy,       -2.0, 2.0)
      + 0.10 * np.clip(stability,         0.0, 1.0)
    )
    penalties = 0.0
    penalties += float(metrics.get("trade_count_variance",  0.0))
    penalties += float(metrics.get("fold_instability",      0.0))
    penalties += float(metrics.get("regime_instability",    0.0))
    # Explicit MR leakage penalty: any MR trade leaks are a hard quality failure
    mr_leak = int(metrics.get("mr_leak_count", 0))
    if mr_leak > 0:
        penalties += float(mr_leak) * 5.0   # 5 pts per leaked MR trade
    return float(base - penalties)


def compute_regime_duration(states: np.ndarray) -> np.ndarray:
    """Compute consecutive bars spent in the current regime at each bar.

    Returns an int32 array of the same length as *states* where value[i]
    is the run-length of the current regime up to and including bar i.
    Used as a causal feature for M15/M5 continuation entry gates.
    """
    dur = np.zeros(len(states), dtype=np.int32)
    run = 0
    prev = None
    for i, s in enumerate(states):
        if prev is None or s != prev:
            run = 1
        else:
            run += 1
        dur[i] = run
        prev = s
    return dur


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
        model_is, states_is, _ = fit_hmm(
            df_is, n_states=_enforce_three_states({"n_states": n_states}, "_run_single_window"), tf=tf
        )
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
        st_tr  = states_is_al[np.isin(np.arange(len(states_is_al)), _tr)]
        st_val = states_is_al[np.isin(np.arange(len(states_is_al)), _val)]
        if len(X_tr) < 200 or len(X_val) < 50:
            continue
        try:
            _rm = train_regime_models(X_tr, y_tr, st_tr, tf=tf, **xgb_kwargs)
            _p  = get_regime_predictions(X_val, st_val,
                                         _rm["trend_model"], _rm["shock_model"])
            _dfv  = df_is_al[df_is_al.index.isin(X_val.index)]
            _stv  = st_val
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

    # Train regime-specific models on the full IS window
    try:
        regime_result = train_regime_models(X_is, y_is, states_is_al, tf=tf, **xgb_kwargs)
    except Exception as exc:
        return _fail(f"XGB regime train: {exc}")

    hmm_zeroed = (regime_result["trend_model"] is None and
                  regime_result["shock_model"] is None)

    probs_oos = get_regime_predictions(
        X_oos, states_oos_al,
        regime_result["trend_model"], regime_result["shock_model"],
    )

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
            # Both regime models absent = complete signal quality failure
            oos_score *= 0.5
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
        # Pinned to 3 for ALL TFs — canonical regime contract.
        # TREND(0), MEAN_REVERSION(1), VOLATILITY_SHOCK(2).
        # Any 4/5-state trials in a resumed study are legacy and will score
        # against floors; new trials will always use n_states=3.
        n_states = 3

        # ── XGBoost params ────────────────────────────────────────────────────
        if tf_up == "M5":
            max_depth        = trial.suggest_int("max_depth", 2, 4)
            reg_alpha        = trial.suggest_float("reg_alpha", 0.001, 0.1,  log=True)
            reg_lambda       = trial.suggest_float("reg_lambda", 0.001, 0.1,  log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 5, 25)
            learning_rate    = trial.suggest_float("learning_rate", 0.02, 0.10, log=True)
            n_estimators     = trial.suggest_int("n_estimators", 50, 400, step=50)
            subsample        = trial.suggest_float("subsample", 0.5, 0.85)
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
                trial_number=trial.number,
            )

            score         = cpcv_result["cpcv_score"]
            n_valid_paths = cpcv_result["n_valid_paths"]
            std_sharpe    = cpcv_result["std_sharpe"]
            median_trades = cpcv_result["median_trades"]

            med_dd = float(np.median(cpcv_result.get("path_drawdowns", [0.0])))
            p75_dd = float(np.percentile(cpcv_result.get("path_drawdowns", [0.0]), 75))

            # Store metrics for Optuna history
            trial.set_user_attr("n_valid_paths",    n_valid_paths)
            trial.set_user_attr("median_trades",    median_trades)
            trial.set_user_attr("std_sharpe",       std_sharpe)
            trial.set_user_attr("median_sharpe",    cpcv_result["median_sharpe"])
            trial.set_user_attr("median_drawdown",  med_dd)
            trial.set_user_attr("p75_drawdown",     p75_dd)

            logger.info(
                "Trial %d [%s/%s]: score=%.3f | valid_paths=%d/%d | std_sharpe=%.3f | "
                "median_trades=%d | med_dd=%.3f | p75_dd=%.3f",
                trial.number, tf, broker, score, n_valid_paths, _N_PATHS,
                std_sharpe, median_trades, med_dd, p75_dd,
            )
            append_trial_score(
                f"Trial {trial.number} [{tf}/{broker}]: score={score:.3f} | "
                f"valid_paths={n_valid_paths}/{_N_PATHS} | "
                f"std_sharpe={std_sharpe:.3f} | median_trades={median_trades} | "
                f"med_dd={med_dd:.3f} | p75_dd={p75_dd:.3f}"
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

        # Canonical contract: all TFs use exactly 3 states.
        n_states = 3

        # ── XGBoost params (same ranges as full CPCV objective) ───────────────
        if tf_up == "M5":
            max_depth        = trial.suggest_int("max_depth", 2, 4)
            reg_alpha        = trial.suggest_float("reg_alpha", 0.001, 0.1,  log=True)
            reg_lambda       = trial.suggest_float("reg_lambda", 0.001, 0.1,  log=True)
            min_child_weight = trial.suggest_int("min_child_weight", 5, 25)
            learning_rate    = trial.suggest_float("learning_rate", 0.02, 0.10, log=True)
            n_estimators     = trial.suggest_int("n_estimators", 50, 400, step=50)
            subsample        = trial.suggest_float("subsample", 0.5, 0.85)
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

        # ── Stage-1 scoring: 5-fold TS log-loss for M5/M15, backtest for H1 ───
        # M5/M15: optimise pure predictive accuracy via averaged XGB log-loss
        # across 5 TimeSeriesSplit folds on the IS data.
        #
        # Why folds instead of a single hold-out:
        #   A single 90-day OOS window can fall entirely in a ranging period
        #   where gold direction is near-50/50, making every trial return
        #   log_loss ≈ ln(2) = 0.693 regardless of hyperparameters.  Five
        #   folds spanning the full 10-year IS window cover both trending and
        #   ranging regimes — the mean log-loss becomes informative.
        #
        # HMM is fitted once on IS (unchanged — keeps Stage-1 fast).
        # A 6-bar embargo is dropped from each val fold start to prevent the
        # forward-looking binary target from leaking across the fold boundary.
        _STAGE1_N_FOLDS  = 5
        # Embargo must match the forward-return horizon used to create labels in
        # prepare_features._TF_HORIZON — see engine_xgb.py.
        _STAGE1_EMBARGO  = {"H1": 6, "M15": 12, "M5": 18}.get(tf_up, 6)
        _STAGE1_MIN_BARS = 500  # minimum training bars per fold
        if tf_up in ("M5", "M15"):
            try:
                _hmm, _states_is, _ = fit_hmm(
                    df_is,
                    n_states=_enforce_three_states({"n_states": int(n_states)}, "stage1_cpcv"),
                    tf=tf,
                )
                X_is_s, y_is_s, _, _ = prepare_features(df_is, _states_is, tf=tf)
                if len(X_is_s) < _STAGE1_MIN_BARS * 2:
                    return -50.0

                tscv       = TimeSeriesSplit(n_splits=_STAGE1_N_FOLDS)
                fold_aucs: list = []
                for _tr_idx, _val_idx in tscv.split(X_is_s):
                    # Drop first `embargo` bars of val to prevent forward-return leakage
                    _val_idx = _val_idx[_STAGE1_EMBARGO:]
                    if len(_tr_idx) < _STAGE1_MIN_BARS or len(_val_idx) < 100:
                        continue
                    X_tr   = X_is_s.iloc[_tr_idx]
                    y_tr   = y_is_s.iloc[_tr_idx]
                    X_val  = X_is_s.iloc[_val_idx]
                    y_val  = y_is_s.iloc[_val_idx]
                    # Skip folds with only one class (TBM can exclude entire
                    # stretches of one direction in low-volatility regimes)
                    if y_val.nunique() < 2 or y_tr.nunique() < 2:
                        continue
                    try:
                        _fm, _ft, _ = train_xgb_ensemble(
                            X_tr, y_tr, train_ratio=1.0, **xgb_kwargs
                        )
                        _, _probs = get_predictions_ensemble(_fm, _ft, X_val)
                        fold_aucs.append(roc_auc_score(y_val, _probs))
                    except Exception:
                        pass

                if not fold_aucs:
                    return -50.0
                auc = float(np.mean(fold_aucs))
                trial.set_user_attr("mean_auc",   round(auc, 5))
                trial.set_user_attr("n_cv_folds",  len(fold_aucs))
                trial.set_user_attr("is_bars",     int(len(X_is_s)))
                logger.info(
                    "Stage1[%s] trial %d: mean_auc=%.5f over %d folds",
                    tf, trial.number, auc, len(fold_aucs),
                )
                # Return AUC - 0.5 so score is ~0 for random, positive for learning.
                # Optuna maximises → higher AUC wins.
                return auc - 0.5
            except Exception as exc:
                logger.warning("Stage1[%s] trial %d error: %s", tf, trial.number, exc)
                return -50.0
        else:
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



# ── Unified trial summary format ──────────────────────────────────────────────

def format_trial_summary(row: dict) -> str:
    """Produce a uniform single-line trial log entry for all TFs."""
    keys = [
        "tf", "trial", "objective_score", "cpcv_mean_score", "cpcv_std_score",
        "valid_paths", "total_paths", "sharpe_median", "pf_median", "calmar_median",
        "expectancy_median", "floating_dd_median", "trades_median", "mr_leak_count",
        "regime_occupancy", "trade_distribution",
        "activation_events", "partial_close_events", "trail_updates",
        "regime_shift_forced_closes", "avg_activation_pnl_usd",
        "pruned", "elapsed_sec",
    ]
    return " | ".join(f"{k}={row.get(k)}" for k in keys)


# ── H1 profitability safeguards ─────────────────────────────────────────────

def _apply_validation_floors(metrics: dict, tf: str) -> "tuple[float, list[str]]":
    """Apply fixed validation quality floors by timeframe."""
    tf_up = tf.upper()
    cfg = VALIDATION_CONFIG.get(tf_up, VALIDATION_CONFIG["H1"])

    penalty: float = 0.0
    reasons: list = []

    min_sharpe = float(cfg.get("min_median_sharpe", 0.0))
    min_pf = float(cfg.get("min_median_pf", 1.0))

    med_sharpe  = float(metrics.get("median_sharpe",   0.0))
    med_pf      = float(metrics.get("median_pf",        1.0))

    if med_sharpe < min_sharpe:
        penalty += 10.0 + (min_sharpe - med_sharpe) * 10.0
        reasons.append(f"floor_sharpe:{med_sharpe:.3f}<{min_sharpe:.3f}")

    if med_pf < min_pf:
        penalty += 6.0 + (min_pf - med_pf) * 12.0
        reasons.append(f"floor_pf:{med_pf:.3f}<{min_pf:.3f}")

    return penalty, reasons


def _turnover_penalty(metrics: dict, tf: str) -> "tuple[float, list[str]]":
    """Penalise excessive churn using fixed max-trade governance by timeframe."""
    tf_up = tf.upper()
    cfg = VALIDATION_CONFIG.get(tf_up, VALIDATION_CONFIG["H1"])

    trades_per_100 = float(metrics.get("trades_per_100_bars", 0.0))
    max_t100 = float(cfg.get("max_trades_per_100", 20.0))

    penalty: float = 0.0
    reasons: list = []

    if trades_per_100 > max_t100:
        penalty += (trades_per_100 - max_t100) * 1.5
        reasons.append(f"turnover:{trades_per_100:.2f}>{max_t100:.2f}")

    return penalty, reasons


# ── TF-specific objective weights ────────────────────────────────────────────

def _tf_objective_breakdown(cpcv_result: dict, tf: str) -> dict:
    """Return weighted objective component breakdown for timeframe-specific scoring."""
    tf_up             = tf.upper()
    median_sharpe     = float(cpcv_result.get("median_sharpe",     0.0))
    median_pf         = float(cpcv_result.get("median_pf",         1.0))
    median_calmar     = float(cpcv_result.get("median_calmar",     0.0))
    median_expectancy = float(cpcv_result.get("median_expectancy", 0.0))
    stability         = float(cpcv_result.get("median_win_rate",   0.0))
    median_dd         = float(cpcv_result.get("median_drawdown",   0.0))

    drawdown_control = max(0.0, 1.0 - median_dd / max(CPCV_MAX_FLOAT_DD, 1e-6))

    if tf_up == "H1":
        weights = {"expectancy": 0.35, "calmar": 0.35, "profit_factor": 0.30}
        components = {
            "expectancy": median_expectancy,
            "calmar": median_calmar,
            "profit_factor": median_pf,
        }
    elif tf_up == "M15":
        weights = {"profit_factor": 0.40, "sharpe": 0.30, "calmar": 0.30}
        components = {
            "profit_factor": median_pf,
            "sharpe": median_sharpe,
            "calmar": median_calmar,
        }
    else:  # M5
        weights = {"stability": 0.40, "profit_factor": 0.30, "drawdown_control": 0.30}
        components = {
            "stability": stability,
            "profit_factor": median_pf,
            "drawdown_control": drawdown_control,
        }

    base_score = float(sum(weights[k] * components[k] for k in weights))
    return {
        "weights": weights,
        "components": components,
        "base_score": base_score,
    }


def _tf_pipeline_score(cpcv_result: dict, tf: str) -> float:
    return float(_tf_objective_breakdown(cpcv_result, tf).get("base_score", 0.0))


# ── Unified single-stage pipeline optimizer ───────────────────────────────────

class PipelineOptimizer:
    """Unified end-to-end CPCV optimizer for a single timeframe.

    Replaces the Stage-1 (AUC) / Stage-2 (CPCV) two-stage flow with one
    objective per trial: HMM -> regimes -> XGB regime models -> signal policy
    -> CPCV backtest.  The objective score is TF-specific (see _tf_pipeline_score).

    Usage::

        opt = PipelineOptimizer(tf="H1", broker="headway_cent",
                                account_size=15.0, df=df)
        study.optimize(opt.objective, n_trials=80)
    """

    def __init__(
        self,
        tf:           str,
        broker:       str,
        account_size: float,
        df:           "pd.DataFrame",
        wfo_mode:     str = "standard",
    ):
        self.tf           = tf.upper()
        self.broker       = broker
        self.account_size = account_size
        self.df           = df
        self.wfo_mode     = wfo_mode

        from pathlib import Path as _Path
        self._jsonl_path = _Path(
            f"reports/optimization_trials_{tf.lower()}_{broker}.jsonl"
        )
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def active_param_names(self) -> list[str]:
        """Return the exact Optuna-tuned parameter names for this timeframe."""
        base = [
            "obs_cov",
            "trans_cov",
            "persistence_threshold",
            "max_depth",
            "learning_rate",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
        ]
        return sorted(base)

    # ── Parameter space ───────────────────────────────────────────────────────

    def suggest_params(self, trial: "optuna.Trial") -> dict:
        """Suggest hyperparameters for one trial.

        Unified parameter space:
        - HMM: n_states, obs_cov, trans_cov, persistence_threshold
        - XGB: per-TF ranges
        Validation floors, CPCV gates, and reporting thresholds are fixed.
        """
        tf_up = self.tf

        # ── Kalman smoothing ──────────────────────────────────────────────
        if tf_up == "M5":
            obs_cov   = trial.suggest_float("obs_cov",   0.05, 5.0,  log=True)
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.1, log=True)
        else:
            obs_cov   = trial.suggest_float("obs_cov",   0.5,  5.0,  log=True)
            trans_cov = trial.suggest_float("trans_cov", 0.001, 0.03, log=True)

        # Pinned to 3 for ALL TFs — canonical regime contract.
        # TREND(0), MEAN_REVERSION(1), VOLATILITY_SHOCK(2).
        n_states = 3

        # HMM persistence gate: int [2,6] -> min-diagonal threshold [0.40,0.60].
        # pt=2 -> 0.40 (permissive), pt=6 -> 0.60 (current default / strict).
        persistence_threshold = trial.suggest_int("persistence_threshold", 2, 6)
        # TODO: wire transition_penalty when engine_hmm exposes transmat smoothing API.
        # TODO: wire median_filter_window when engine_hmm exposes state post-filter API.
        # TODO: wire gmm_n_components / gmm_covariance_type when processor
        #       supports per-trial GMM fitting (currently fitted once in process_pipeline).

        # ── XGBoost (TF-specific ranges per spec) ─────────────────────────
        if tf_up == "H1":
            max_depth        = trial.suggest_int(  "max_depth",        3,  6)
            learning_rate    = trial.suggest_float("learning_rate",    0.01, 0.08,  log=True)
            subsample        = trial.suggest_float("subsample",        0.6,  0.95)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6,  0.95)
            min_child_weight = trial.suggest_int(  "min_child_weight", 5,  30)
            gamma            = trial.suggest_float("gamma",            0.0,  5.0)
            n_estimators     = trial.suggest_int(  "n_estimators",     200, 1200)
        elif tf_up == "M15":
            max_depth        = trial.suggest_int(  "max_depth",        3,  8)
            learning_rate    = trial.suggest_float("learning_rate",    0.01, 0.15,  log=True)
            subsample        = trial.suggest_float("subsample",        0.5,  1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5,  1.0)
            min_child_weight = trial.suggest_int(  "min_child_weight", 3,  25)
            gamma            = trial.suggest_float("gamma",            0.0, 10.0)
            n_estimators     = trial.suggest_int(  "n_estimators",     300, 1500)
        else:  # M5
            max_depth        = trial.suggest_int(  "max_depth",        2,  5)
            learning_rate    = trial.suggest_float("learning_rate",    0.005, 0.05, log=True)
            subsample        = trial.suggest_float("subsample",        0.7,  1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.7,  1.0)
            min_child_weight = trial.suggest_int(  "min_child_weight", 10, 50)
            gamma            = trial.suggest_float("gamma",            1.0, 15.0)
            n_estimators     = trial.suggest_int(  "n_estimators",     500, 2500)

        reg_alpha        = trial.suggest_float("reg_alpha",        1e-6, 0.5,  log=True)
        reg_lambda       = trial.suggest_float("reg_lambda",       0.01, 2.0,  log=True)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5,  2.0,  log=True)

        _signal_cfg = {
            "entry_probability_threshold": {"H1": 0.60, "M15": 0.55}.get(tf_up, None),
            "confirmation_bars": {"H1": 4, "M15": 3, "M5": 2}.get(tf_up, 2),
        }

        return {
            "kalman": {
                "obs_cov":   obs_cov,
                "trans_cov": trans_cov,
            },
            "hmm": {
                "n_states":              n_states,
                "persistence_threshold": persistence_threshold,
            },
            "xgb": {
                "max_depth":        max_depth,
                "learning_rate":    learning_rate,
                "n_estimators":     n_estimators,
                "subsample":        subsample,
                "colsample_bytree": colsample_bytree,
                "min_child_weight": min_child_weight,
                "gamma":            gamma,
                "reg_alpha":        reg_alpha,
                "reg_lambda":       reg_lambda,
                "scale_pos_weight": scale_pos_weight,
            },
            "signal": _signal_cfg,
        }

    # ── Single trial execution ────────────────────────────────────────────────

    def run_single_trial(self, trial: "optuna.Trial") -> dict:
        """Run full CPCV pipeline for one trial; return result dict."""
        t_start = time.time()
        params  = self.suggest_params(trial)

        kalman_col = kalman_smooth(
            self.df["log_return"].values,
            params["kalman"]["obs_cov"],
            params["kalman"]["trans_cov"],
        )
        df_trial = self.df.copy()
        df_trial["kalman_return"] = kalman_col

        xgb_kwargs = params["xgb"]
        hmm_params = params["hmm"]
        signal_cfg = params.get("signal", {})
        n_states   = int(hmm_params["n_states"])
        tf_embargo = CPCV_PURGE_BARS.get(self.tf, 24)

        # Phase 1: lightweight occupancy diagnostics + early gate before CPCV loops.
        _gate_model, gate_states, _ = fit_hmm(
            df_trial,
            n_states=_enforce_three_states({"n_states": n_states}, "run_single_trial_gate"),
            tf=self.tf,
        )
        _gate_returns = df_trial["log_return"].values[: len(gate_states)]
        _diag = summarize_regimes(gate_states, _gate_returns, STATE_NAMES)
        write_regime_diagnostics(
            tf=self.tf,
            states=gate_states,
            returns=_gate_returns,
            state_labels=STATE_NAMES,
            output_dir="reports",
        )
        trend_pct = get_occupancy_percentage(_diag, "TREND")
        shock_pct = get_occupancy_percentage(_diag, "VOLATILITY_SHOCK")
        if trend_pct < 0.15 or shock_pct < 0.05:
            fail_reason = []
            if trend_pct < 0.15:
                fail_reason.append(f"trend_pct<{0.15:.2f}")
            if shock_pct < 0.05:
                fail_reason.append(f"shock_pct<{0.05:.2f}")
            fail_tag = ",".join(fail_reason)
            summary = {
                "tf": self.tf,
                "trial": trial.number,
                "objective_score": -90.0,
                "cpcv_mean_score": -90.0,
                "cpcv_std_score": 0.0,
                "valid_paths": 0,
                "total_paths": _N_PATHS,
                "sharpe_median": 0.0,
                "pf_median": 1.0,
                "calmar_median": 0.0,
                "expectancy_median": 0.0,
                "floating_dd_median": 0.0,
                "trades_median": 0,
                "mr_leak_count": 0,
                "activation_events": 0,
                "partial_close_events": 0,
                "trail_updates": 0,
                "regime_shift_forced_closes": 0,
                "avg_activation_pnl_usd": 0.0,
                "p_floor": 0.0,
                "p_turn": 0.0,
                "floor_reasons": f"occupancy_gate:{fail_tag}",
                "pruned": True,
                "elapsed_sec": round(time.time() - t_start, 1),
            }
            return {
                "objective_score": -90.0,
                "cpcv": {
                    "cpcv_score": -90.0,
                    "n_valid_paths": 0,
                    "std_sharpe": 0.0,
                    "median_sharpe": 0.0,
                    "median_trades": 0,
                    "median_drawdown": 0.0,
                    "median_pf": 1.0,
                    "median_expectancy": 0.0,
                    "median_calmar": 0.0,
                    "total_mr_leaks": 0,
                    "regime_occupancy": {
                        "TREND": float(_diag.get("TREND", {}).get("percentage", 0.0)),
                        "MEAN_REVERSION": float(_diag.get("MEAN_REVERSION", {}).get("percentage", 0.0)),
                        "VOLATILITY_SHOCK": float(_diag.get("VOLATILITY_SHOCK", {}).get("percentage", 0.0)),
                    },
                    "trade_distribution": {
                        "TREND": 0,
                        "MEAN_REVERSION": 0,
                        "VOLATILITY_SHOCK": 0,
                    },
                    "path_scores": [],
                    "path_sharpes": [],
                    "path_trades": [],
                    "path_winrates": [],
                    "path_drawdowns": [],
                    "path_returns": [],
                    "path_pfs": [],
                    "path_expectancies": [],
                },
                "summary": summary,
                "objective_breakdown": {"weights": {}, "components": {}, "base_score": 0.0},
                "penalties": {"occupancy_gate": 90.0},
            }

        cpcv = execute_cpcv(
            df_full      = df_trial,
            n_states     = n_states,
            tf           = self.tf,
            balance      = self.account_size,
            broker       = self.broker,
            xgb_kwargs   = xgb_kwargs,
            hmm_params   = hmm_params,
            signal_cfg   = signal_cfg,
            n_splits     = CPCV_N_BLOCKS,
            n_test_splits= CPCV_K_TEST,
            embargo_bars = tf_embargo,
            trial_number = trial.number,
        )

        # TF-specific objective base score
        objective_breakdown = _tf_objective_breakdown(cpcv, self.tf)
        base_score      = float(objective_breakdown.get("base_score", 0.0))
        objective_score = base_score

        # Penalties
        std_sharpe        = float(cpcv.get("std_sharpe",    0.0))
        n_valid           = int(  cpcv.get("n_valid_paths", 0))
        median_dd         = float(cpcv.get("median_drawdown", 0.0))
        mr_leaks          = int(  cpcv.get("total_mr_leaks",  0))
        median_trades     = int(  cpcv.get("median_trades",   0))

        variance_penalty = 0.10 * std_sharpe
        instability_penalty = 0.50 * max(0, _N_PATHS - n_valid)
        dd_penalty = 20.0 if median_dd > CPCV_MAX_FLOAT_DD else 0.0
        mr_penalty = float(mr_leaks) * 5.0

        _test_bars = max(len(self.df) * CPCV_K_TEST // CPCV_N_BLOCKS, 1)
        _cpcv_ext = dict(cpcv)
        _cpcv_ext["trades_per_100_bars"] = (median_trades / _test_bars) * 100.0

        p_floor, floor_reasons = _apply_validation_floors(cpcv, self.tf)
        p_turn, turn_reasons = _turnover_penalty(_cpcv_ext, self.tf)

        occ = cpcv.get("regime_occupancy", {})
        occ_trend = float(occ.get("TREND", 0.0)) / 100.0
        occ_shock = float(occ.get("VOLATILITY_SHOCK", 0.0)) / 100.0
        occ_penalty = 0.0
        occ_reasons: list[str] = []
        if occ_trend < 0.15:
            occ_penalty += (0.15 - occ_trend) * 20.0
            occ_reasons.append("regime_occ_trend_low")
        if occ_shock < 0.05:
            occ_penalty += (0.05 - occ_shock) * 25.0
            occ_reasons.append("regime_occ_shock_low")

        m15_trade_penalty = 0.0
        if self.tf == "M15" and median_trades > 400:
            m15_trade_penalty = float(median_trades - 400) * 0.03

        penalties = {
            "variance": variance_penalty,
            "instability": instability_penalty,
            "drawdown": dd_penalty,
            "mr_leak": mr_penalty,
            "validation_floor": p_floor,
            "overtrading": p_turn,
            "regime_occupancy_instability": occ_penalty,
            "m15_trade_excess": m15_trade_penalty,
        }
        objective_score -= float(sum(penalties.values()))

        all_reasons = floor_reasons + turn_reasons + occ_reasons
        logger.info(
            "trial=%d tf=%s base=%.4f p_var=%.4f p_inst=%.4f p_dd=%.4f "
            "p_mr=%.4f p_floor=%.4f p_turn=%.4f p_occ=%.4f p_m15=%.4f final=%.4f reasons=%s",
            trial.number, self.tf,
            base_score,
            variance_penalty, instability_penalty, dd_penalty, mr_penalty,
            p_floor, p_turn, occ_penalty, m15_trade_penalty,
            objective_score,
            "|".join(all_reasons) or "none",
        )

        elapsed = round(time.time() - t_start, 1)

        summary = {
            "tf":                 self.tf,
            "trial":              trial.number,
            "objective_score":    round(float(objective_score),               4),
            "cpcv_mean_score":    round(float(cpcv.get("cpcv_score", 0.0)),   4),
            "cpcv_std_score":     round(float(std_sharpe),                    4),
            "valid_paths":        n_valid,
            "total_paths":        _N_PATHS,
            "sharpe_median":      round(float(cpcv.get("median_sharpe",    0.0)), 4),
            "pf_median":          round(float(cpcv.get("median_pf",        1.0)), 4),
            "calmar_median":      round(float(cpcv.get("median_calmar",    0.0)), 4),
            "expectancy_median":  round(float(cpcv.get("median_expectancy",0.0)), 6),
            "floating_dd_median": round(float(median_dd),                     4),
            "trades_median":      median_trades,
            "mr_leak_count":      mr_leaks,
            "activation_events":         int(cpcv.get("total_activation_events",    0)),
            "partial_close_events":       int(cpcv.get("total_partial_close_events", 0)),
            "trail_updates":              int(cpcv.get("total_trail_updates",         0)),
            "regime_shift_forced_closes": int(cpcv.get("total_regime_forced_closes", 0)),
            "avg_activation_pnl_usd":     round(float(cpcv.get("avg_activation_pnl_usd", 0.0)), 4),
            "regime_occupancy":    dict(cpcv.get("regime_occupancy", {})),
            "trade_distribution":  dict(cpcv.get("trade_distribution", {})),
            "p_floor":            round(p_floor, 4),
            "p_turn":             round(p_turn,  4),
            "floor_reasons":      "|".join(floor_reasons) or "none",
            "pruned":             False,
            "elapsed_sec":        elapsed,
        }

        return {
            "objective_score": objective_score,
            "cpcv": cpcv,
            "summary": summary,
            "objective_breakdown": objective_breakdown,
            "penalties": penalties,
        }

    # ── Optuna objective ──────────────────────────────────────────────────────

    def objective(self, trial: "optuna.Trial") -> float:
        """Optuna-compatible objective function."""
        t_start = time.time()
        try:
            result  = self.run_single_trial(trial)
            score   = result["objective_score"]
            summary = result["summary"]
            cpcv    = result["cpcv"]
            objective_breakdown = result.get("objective_breakdown", {})
            penalties = result.get("penalties", {})

            # Store attrs for Optuna history / callbacks
            trial.set_user_attr("n_valid_paths",   summary["valid_paths"])
            trial.set_user_attr("median_trades",   summary["trades_median"])
            trial.set_user_attr("std_sharpe",      summary["cpcv_std_score"])
            trial.set_user_attr("median_sharpe",   summary["sharpe_median"])
            trial.set_user_attr("median_drawdown", summary["floating_dd_median"])
            trial.set_user_attr("mr_leak_count",   summary["mr_leak_count"])
            trial.set_user_attr("median_win_rate",   float(cpcv.get("median_win_rate", 0.0)))
            trial.set_user_attr("median_return",     float(cpcv.get("median_return", 0.0)))
            trial.set_user_attr("median_pf",         float(cpcv.get("median_pf", 1.0)))
            trial.set_user_attr("median_expectancy", float(cpcv.get("median_expectancy", 0.0)))
            trial.set_user_attr("median_calmar",     float(cpcv.get("median_calmar", 0.0)))

            trial.set_user_attr("path_scores",       list(cpcv.get("path_scores", [])))
            trial.set_user_attr("path_sharpes",      list(cpcv.get("path_sharpes", [])))
            trial.set_user_attr("path_trades",       list(cpcv.get("path_trades", [])))
            trial.set_user_attr("path_winrates",     list(cpcv.get("path_winrates", [])))
            trial.set_user_attr("path_drawdowns",    list(cpcv.get("path_drawdowns", [])))
            trial.set_user_attr("path_returns",      list(cpcv.get("path_returns", [])))
            trial.set_user_attr("path_pfs",          list(cpcv.get("path_pfs", [])))
            trial.set_user_attr("path_expectancies", list(cpcv.get("path_expectancies", [])))

            trial.set_user_attr("total_activation_events",    int(cpcv.get("total_activation_events", 0)))
            trial.set_user_attr("total_partial_close_events", int(cpcv.get("total_partial_close_events", 0)))
            trial.set_user_attr("total_regime_forced_closes", int(cpcv.get("total_regime_forced_closes", 0)))
            trial.set_user_attr("total_trail_updates",        int(cpcv.get("total_trail_updates", 0)))
            trial.set_user_attr("avg_activation_pnl_usd",     float(cpcv.get("avg_activation_pnl_usd", 0.0)))
            trial.set_user_attr("regime_occupancy", cpcv.get("regime_occupancy", {}))
            trial.set_user_attr("trade_distribution", cpcv.get("trade_distribution", {}))
            trial.set_user_attr("objective_breakdown", objective_breakdown)
            trial.set_user_attr("penalties", penalties)

            log_line = format_trial_summary(summary)
            logger.info(log_line)
            append_trial_score(log_line)

            try:
                with open(self._jsonl_path, "a", encoding="utf-8") as _fh:
                    _fh.write(json.dumps(summary) + "\n")
            except Exception:
                pass

            return float(score)

        except Exception as exc:
            elapsed = round(time.time() - t_start, 1)
            logger.warning("Trial %d [%s] failed: %s", trial.number, self.tf, exc)
            _err_row = {
                "tf": self.tf, "trial": trial.number, "objective_score": -100.0,
                "cpcv_mean_score": None, "cpcv_std_score": None,
                "valid_paths": 0, "total_paths": _N_PATHS,
                "sharpe_median": None, "pf_median": None, "calmar_median": None,
                "expectancy_median": None, "floating_dd_median": None,
                "trades_median": 0, "mr_leak_count": 0,
                "regime_occupancy": {}, "trade_distribution": {},
                "activation_events": 0, "partial_close_events": 0,
                "trail_updates": 0, "regime_shift_forced_closes": 0,
                "avg_activation_pnl_usd": 0.0,
                "pruned": False, "elapsed_sec": elapsed,
            }
            try:
                with open(self._jsonl_path, "a", encoding="utf-8") as _fh:
                    _fh.write(json.dumps(_err_row) + "\n")
            except Exception:
                pass
            return -100.0

        finally:
            gc.collect()


def _cpcv_from_attrs(attrs: dict, final_score: float) -> dict:
    """Rebuild a CPCV payload from persisted Optuna trial attrs."""
    return {
        "cpcv_score": float(final_score),
        "n_valid_paths": int(attrs.get("n_valid_paths", 0)),
        "median_sharpe": float(attrs.get("median_sharpe", 0.0)),
        "std_sharpe": float(attrs.get("std_sharpe", 0.0)),
        "median_trades": int(attrs.get("median_trades", 0)),
        "median_win_rate": float(attrs.get("median_win_rate", 0.0)),
        "median_drawdown": float(attrs.get("median_drawdown", 0.0)),
        "median_return": float(attrs.get("median_return", 0.0)),
        "median_pf": float(attrs.get("median_pf", 1.0)),
        "median_expectancy": float(attrs.get("median_expectancy", 0.0)),
        "median_calmar": float(attrs.get("median_calmar", 0.0)),
        "path_scores": list(attrs.get("path_scores", [])),
        "path_sharpes": list(attrs.get("path_sharpes", [])),
        "path_trades": list(attrs.get("path_trades", [])),
        "path_winrates": list(attrs.get("path_winrates", [])),
        "path_drawdowns": list(attrs.get("path_drawdowns", [])),
        "path_returns": list(attrs.get("path_returns", [])),
        "path_pfs": list(attrs.get("path_pfs", [])),
        "path_expectancies": list(attrs.get("path_expectancies", [])),
        "regime_occupancy": dict(attrs.get("regime_occupancy", {})),
        "trade_distribution": dict(attrs.get("trade_distribution", {})),
        "total_mr_leaks": int(attrs.get("mr_leak_count", 0)),
        "total_activation_events": int(attrs.get("total_activation_events", 0)),
        "total_partial_close_events": int(attrs.get("total_partial_close_events", 0)),
        "total_regime_forced_closes": int(attrs.get("total_regime_forced_closes", 0)),
        "total_trail_updates": int(attrs.get("total_trail_updates", 0)),
        "avg_activation_pnl_usd": float(attrs.get("avg_activation_pnl_usd", 0.0)),
    }

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
    logger.warning(
        "[DEPRECATED] run_optimization_stage1 is obsolete. "
        "Two-stage optimization has been removed. "
        "Forwarding to unified single-stage CPCV pipeline."
    )
    print(
        "\n[DEPRECATED] Stage-1 optimization has been removed.\n"
        "The unified pipeline optimizes end-to-end trading outcomes via CPCV.\n"
        "Running unified pipeline instead...\n"
    )
    return run_optimization(
        df=df, tf=tf, broker=broker, account_size=account_size, n_trials=n_trials,
    )


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

    # Unified pruner + sampler for all TFs (spec: TPE multivariate + Hyperband)
    pruner = optuna.pruners.HyperbandPruner(min_resource=10, reduction_factor=3)
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        seed=42,
        n_startup_trials=20,
    )

    # Per-broker storage: all TF studies colocated so the Optuna dashboard
    # can display every active study from a single --storage argument.
    name    = get_optuna_study_name(tf, broker)
    storage = get_optuna_storage_url(broker)
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )

    feature_cols = list(get_feature_cols(df))
    _check_study_hash(study, tf=tf, broker=broker, feature_cols=feature_cols)

    _existing_params: set[str] = set()
    for _t in study.trials:
        if _t.params:
            _existing_params.update(str(k) for k in _t.params.keys())
    if _existing_params:
        try:
            audit_search_space(_existing_params)
        except RuntimeError as _audit_err:
            raise RuntimeError(
                f"Existing study contains forbidden/unknown params: {_audit_err}. "
                "Reset the study (--reset_study) before continuing."
            )

    if warm_start_stage1:
        logger.warning(
            "warm_start_stage1 is deprecated and ignored in unified mode. "
            "The two-stage flow has been removed."
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

    _pipeline_opt = PipelineOptimizer(
        tf=tf, broker=broker, account_size=account_size,
        df=df, wfo_mode=wfo_mode,
    )

    _active_params = _pipeline_opt.active_param_names()
    print("Active Optuna search space:")
    print(search_space_as_text(_active_params))
    audit_search_space(_active_params)
    logger.info("Optuna search space audited: %s", _active_params)

    study.optimize(
        _pipeline_opt.objective,
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

    try:
        _bt = study.best_trial
        _attrs = _bt.user_attrs
        _cpcv_data = _cpcv_from_attrs(_attrs, final_score=float(best))
        _objective_breakdown = dict(_attrs.get("objective_breakdown", {}))
        _penalties = dict(_attrs.get("penalties", {}))

        _report = build_cpcv_report(
            tf=tf,
            broker=broker,
            optimization_summary={
                "study_name": name,
                "best_trial": int(_bt.number),
                "completed_trials": int(total_done),
                "search_space": list(_active_params),
            },
            cpcv_result=_cpcv_data,
            objective_breakdown=_objective_breakdown,
            final_score=float(best),
            penalties=_penalties,
        )
        _path_main = write_cpcv_report(_report, tf=tf, output_dir="reports")
        _path_legacy = write_legacy_cpcv_report(_report, tf=tf, broker=broker, output_dir="reports")
        logger.info("Unified CPCV report saved → %s", _path_main)
        logger.info("Legacy CPCV report mirror saved → %s", _path_legacy)

        _occ = _cpcv_data.get("regime_occupancy", {})
        _trd = _cpcv_data.get("trade_distribution", {})
        print(f"\n{tf.upper()}")
        print(f"TREND: {float(_occ.get('TREND', 0.0)):.1f}%")
        print(f"MEAN_REVERSION: {float(_occ.get('MEAN_REVERSION', 0.0)):.1f}%")
        print(f"VOLATILITY_SHOCK: {float(_occ.get('VOLATILITY_SHOCK', 0.0)):.1f}%")
        print("\nTrades:")
        print(f"Trend = {int(_trd.get('TREND', 0))}")
        print(f"Shock = {int(_trd.get('VOLATILITY_SHOCK', 0))}")
        _mr_tr = int(_trd.get("MEAN_REVERSION", 0))
        print(f"MR = {_mr_tr}")
        if _mr_tr > 0:
            print("DEFECT: MR trades are non-zero.")
    except Exception as _save_err:
        logger.warning("Could not save unified CPCV report after optimisation: %s", _save_err)

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
    _data = _cpcv_from_attrs(_attrs, final_score=float(_study.best_value))
    _report = build_cpcv_report(
        tf=tf,
        broker=broker,
        optimization_summary={
            "study_name": _study.study_name,
            "best_trial": int(_bt.number),
            "completed_trials": len([t for t in _study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "search_space": sorted(list(_bt.params.keys())),
        },
        cpcv_result=_data,
        objective_breakdown=dict(_attrs.get("objective_breakdown", {})),
        final_score=float(_study.best_value),
        penalties=dict(_attrs.get("penalties", {})),
    )
    _path = write_cpcv_report(_report, tf=tf, output_dir="reports")
    write_legacy_cpcv_report(_report, tf=tf, broker=broker, output_dir="reports")
    logger.info(
        "CPCV stats saved from best trial → %s  (score=%.3f  median_sharpe=%.3f  trades=%d)",
        _path, _data["cpcv_score"], _data["median_sharpe"], _data["median_trades"],
    )
    return True


def get_best_params(balance: float = 15.0, broker: str = "standard", tf: str = "H1") -> dict:
    """Load best hyperparameters from the persisted SQLite study.

    Fallback chain:
        1. Unified per-TF study: study_{broker}_{tf}.db / {tf}_{broker}_single_pipeline
        2. Tier-matched legacy study: study_{broker}.db
        3. Small-tier legacy study
        4. Legacy study.db
    """
    # 1a. Canonical broker-level DB (current unified pipeline)
    _new_name    = get_optuna_study_name(tf, broker)
    _new_storage = get_optuna_storage_url(broker)
    try:
        _p = optuna.load_study(study_name=_new_name, storage=_new_storage).best_params
        if _p:
            return _p
    except Exception:
        pass

    # 1b. Legacy per-TF DB (backward compat for studies created before patch 3)
    _tf_db = _study_db_tf(broker, tf)
    try:
        _p = optuna.load_study(study_name=_new_name, storage=_tf_db).best_params
        if _p:
            logger.info("Loaded params from legacy per-TF DB %s", _tf_db)
            return _p
    except Exception:
        pass

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

    # Exclude all non-XGB params (includes new unified pipeline params)
    _NON_XGB = frozenset(("obs_cov", "trans_cov", "n_states", "persistence_threshold", "meta"))
    xgb_kwargs = {k: v for k, v in params.items() if k not in _NON_XGB}
    hmm_params_wfa = {
        "n_states":              n_states,
        "persistence_threshold": int(params.get("persistence_threshold", 6)),
    }

    tf_embargo = CPCV_PURGE_BARS.get(tf.upper(), 24)
    result = execute_cpcv(
        df_full=df_trial,
        n_states=n_states,
        tf=tf,
        balance=account_size,
        broker=broker,
        xgb_kwargs=xgb_kwargs,
        hmm_params=hmm_params_wfa,
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

    _report = build_cpcv_report(
        tf=tf,
        broker=broker,
        optimization_summary={
            "study_name": f"wfa_{tf.lower()}_{broker}",
            "best_trial": None,
            "completed_trials": None,
            "search_space": [],
        },
        cpcv_result=result,
        objective_breakdown={"weights": {}, "components": {}, "base_score": float(result.get("cpcv_score", 0.0))},
        final_score=float(result.get("cpcv_score", 0.0)),
        penalties={},
    )
    _path_main = write_cpcv_report(_report, tf=tf, output_dir="reports")
    _path_legacy = write_legacy_cpcv_report(_report, tf=tf, broker=broker, output_dir="reports")
    logger.info("CPCV report saved → %s", _path_main)
    logger.info("CPCV legacy mirror saved → %s", _path_legacy)

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
    trial_number: int | None = None,
    hmm_params: dict | None = None,
    signal_cfg: dict | None = None,
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

    path_scores:       list = []
    path_sharpes:      list = []
    path_trades:       list = []
    path_winrates:     list = []
    path_drawdowns:    list = []
    path_returns:      list = []
    path_pfs:          list = []
    path_expectancies: list = []
    path_mr_leaks:     list = []
    path_occ_trend:    list = []
    path_occ_mr:       list = []
    path_occ_shock:    list = []
    path_trd_trend:    list = []
    path_trd_mr:       list = []
    path_trd_shock:    list = []
    # lifecycle parity telemetry accumulators
    path_activation_events:        list = []
    path_partial_close_events:     list = []
    path_regime_forced_closes:     list = []
    path_trail_updates:            list = []
    path_avg_activation_pnl:       list = []
    n_valid_paths = 0
    min_trades = HARD_TRADE_FLOORS.get(tf.upper(), 10)

    # Entry probability threshold override for policy path.
    _entry_prob = (signal_cfg or {}).get("entry_probability_threshold")
    _eval_cfg = None
    if _entry_prob is not None:
        _eval_cfg = {
            "entry_probability_threshold": float(_entry_prob),
            # Backward-compat alias consumed by existing backtester wiring.
            "h1_entry_prob": float(_entry_prob),
        }

    for path_idx, (train_mask, test_mask) in enumerate(splitter.split()):
        try:
            df_train = df_full[train_mask]
            if len(df_train) < 500:
                continue

            # 1. Fit HMM STRICTLY on the purged training blocks (no lookahead)
            model_path, train_states, _ = fit_hmm(
                df_train,
                n_states=_enforce_three_states({"n_states": n_states}, "execute_cpcv"),
                tf=tf,
            )
            # persistence_threshold [2,6] -> min-diagonal gate [0.40,0.60]
            _pt = int((hmm_params or {}).get("persistence_threshold", 6))
            _min_diag = 0.30 + _pt * 0.05
            if min(model_path.transmat_[i, i] for i in range(n_states)) < _min_diag:
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

            # 4. Train regime-specific models on the training paths
            X_tr, y_tr = X_all.iloc[train_mask_al], y_all.iloc[train_mask_al]
            states_tr_al = _aligned_states_from_prepare(df_full, all_states, df_all_aligned)
            states_tr_aligned = states_tr_al[train_mask_al]

            regime_result_cpcv = train_regime_models(
                X_tr, y_tr, states_tr_aligned, tf=tf, **xgb_kwargs
            )
            trend_model_c = regime_result_cpcv["trend_model"]
            shock_model_c = regime_result_cpcv["shock_model"]

            # hmm_zeroed if both models absent (penalised per-path below)
            _regime_imp = 0.0 if (trend_model_c is None and shock_model_c is None) else 1.0

            # 5. Merge probabilities then backtest on the OOS test paths
            states_all_al = _aligned_states_from_prepare(df_full, all_states, df_all_aligned)
            probs_all = get_regime_predictions(X_all, states_all_al, trend_model_c, shock_model_c)

            test_df     = df_all_aligned.iloc[test_mask_al]
            test_probs  = probs_all[test_mask_al]
            test_states = states_all_al[test_mask_al]

            # Phase 3: add regime_duration feature for M15/M5 so _m15_entry_ok
            # receives the accurate run-length for each bar in the test window.
            if tf.upper() in ("M15", "M5"):
                _reg_dur = compute_regime_duration(test_states)
                test_df = test_df.copy()
                test_df["regime_duration"] = _reg_dur

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
                evaluator_config=_eval_cfg,
            )

            n_trades = int(path_result.get("n_trades", 0))
            fdd = float(path_result.get("floating_max_drawdown", path_result.get("max_drawdown", 0.0)))

            # Compute base score using the same composite logic as before.
            if tf.upper() in ("M15", "M5"):
                _calmar = path_result.get("total_return", 0.0) / max(path_result.get("max_drawdown", 1e-6), 1e-6)
                _cs_metrics = {
                    "sharpe":        float(path_result.get("sharpe_ratio", 0.0)),
                    "calmar":        float(np.clip(_calmar, -5.0, 5.0)),
                    "profit_factor": float(path_result.get("profit_factor", 1.0)),
                    "expectancy":    float(path_result.get("expected_payoff", 0.0)),
                }
                base_score = float(composite_score(_cs_metrics))
            else:
                base_score = float(_score_result(path_result, broker=broker, tf=tf))

            # Continuous penalties — graded so Optuna gets a gradient even for
            # sub-threshold paths.  Hard validity flag is kept strict (20% DD cap).
            trade_shortfall = max(0.0, float(min_trades - n_trades)) / max(float(min_trades), 1.0)
            dd_excess = max(0.0, fdd - CPCV_MAX_FLOAT_DD)

            regime_penalty = 0.0
            if _regime_imp == 0.0:
                regime_penalty = 2.0

            path_score = base_score - (2.0 * trade_shortfall) - (8.0 * dd_excess) - regime_penalty

            is_valid = (n_trades >= min_trades) and (fdd <= CPCV_MAX_FLOAT_DD)
            if is_valid:
                n_valid_paths += 1

            path_scores.append(path_score)
            path_sharpes.append(float(path_result.get("sharpe_ratio", 0.0)))
            path_trades.append(n_trades)
            path_winrates.append(float(path_result.get("win_rate", 0.0)))
            path_drawdowns.append(fdd)
            path_returns.append(float(path_result.get("total_return", 0.0)))
            path_pfs.append(float(path_result.get("profit_factor", 1.0)))
            path_expectancies.append(float(path_result.get("expected_payoff", 0.0)))
            path_mr_leaks.append(int(path_result.get("mr_leak_count",
                                                       path_result.get("mr_trades", 0))))
            _occ = path_result.get("regime_occupancy", {})
            path_occ_trend.append(float(_occ.get("TREND", 0.0)))
            path_occ_mr.append(float(_occ.get("MEAN_REVERSION", 0.0)))
            path_occ_shock.append(float(_occ.get("VOLATILITY_SHOCK", 0.0)))

            _dist = path_result.get("trade_distribution", {})
            path_trd_trend.append(int(_dist.get("TREND", 0)))
            path_trd_mr.append(int(_dist.get("MEAN_REVERSION", 0)))
            path_trd_shock.append(int(_dist.get("VOLATILITY_SHOCK", 0)))
            # lifecycle counters
            path_activation_events.append(int(path_result.get("activation_events", 0)))
            path_partial_close_events.append(int(path_result.get("partial_close_events", 0)))
            path_regime_forced_closes.append(int(path_result.get("regime_shift_forced_closes", 0)))
            path_trail_updates.append(int(path_result.get("trail_updates", 0)))
            path_avg_activation_pnl.append(float(path_result.get("avg_activation_pnl_usd", 0.0)))

        except Exception as exc:
            logger.debug("CPCV path %d failed: %s", path_idx + 1, exc)
            continue

    # ── Progressive CPCV Scoring ─────────────────────────────────────────────
    # Adaptive minimum valid paths: strict only after Optuna has enough data
    # to build a reliable surrogate (trial >= 80); during warm-up accept fewer
    # valid paths so the study still gets informative gradient.
    total_paths = int(comb(CPCV_N_BLOCKS, CPCV_K_TEST))

    if trial_number is None:
        min_valid_required = 3
    elif trial_number < 30:
        min_valid_required = 1
    elif trial_number < 80:
        min_valid_required = 2
    else:
        min_valid_required = 3

    med_score = float(np.median(path_scores)) if path_scores else -50.0
    std_score = float(np.std(path_scores))    if path_scores else 0.0
    base_score = med_score - (0.25 * std_score)

    # 1.5 per invalid path — graded so partial recovery is rewarded.
    path_penalty = float(total_paths - n_valid_paths) * 1.5
    final_score  = base_score - path_penalty

    if n_valid_paths < min_valid_required:
        final_score -= 5.0
        logger.warning(
            "CPCV [%s]: only %d/%d valid paths (required=%d at trial=%s) - soft fail.",
            tf, n_valid_paths, total_paths, min_valid_required, str(trial_number),
        )

    valid_sharpes = [s for s in path_sharpes if s > -9.0]
    logger.info(
        "CPCV [%s]: %d/%d valid paths | median_score=%.3f std=%.3f | "
        "path_penalty=%.1f | final_score=%.3f",
        tf, n_valid_paths, total_paths,
        med_score, std_score, path_penalty, final_score,
    )
    # Per-path calmar for TF-specific objective
    _path_calmars = [
        (r / d if d > 1e-6 else (5.0 if r > 0 else 0.0))
        for r, d in zip(path_returns, path_drawdowns)
    ]

    regime_occupancy = {
        "TREND": float(np.median(path_occ_trend)) if path_occ_trend else 0.0,
        "MEAN_REVERSION": float(np.median(path_occ_mr)) if path_occ_mr else 0.0,
        "VOLATILITY_SHOCK": float(np.median(path_occ_shock)) if path_occ_shock else 0.0,
    }
    trade_distribution = {
        "TREND": int(np.median(path_trd_trend)) if path_trd_trend else 0,
        "MEAN_REVERSION": int(np.median(path_trd_mr)) if path_trd_mr else 0,
        "VOLATILITY_SHOCK": int(np.median(path_trd_shock)) if path_trd_shock else 0,
    }

    return {
        "cpcv_score":        final_score,
        "n_valid_paths":     n_valid_paths,
        "median_sharpe":     float(np.median(valid_sharpes)) if valid_sharpes else -10.0,
        "std_sharpe":        float(np.std(valid_sharpes))    if len(valid_sharpes) > 1 else 0.0,
        "median_trades":     int(np.median(path_trades) if path_trades else 0),
        "median_win_rate":   float(np.median(path_winrates) if path_winrates else 0.0),
        "median_drawdown":   float(np.median(path_drawdowns) if path_drawdowns else 0.0),
        "median_return":     float(np.median(path_returns) if path_returns else 0.0),
        "median_pf":         float(np.median(path_pfs) if path_pfs else 1.0),
        "median_expectancy": float(np.median(path_expectancies) if path_expectancies else 0.0),
        "median_calmar":     float(np.clip(
            np.median(_path_calmars) if _path_calmars else 0.0, -5.0, 5.0
        )),
        "total_mr_leaks":    int(sum(path_mr_leaks)),
        "regime_occupancy":  regime_occupancy,
        "trade_distribution": trade_distribution,
        # lifecycle parity telemetry
        "total_activation_events":       int(sum(path_activation_events)),
        "total_partial_close_events":    int(sum(path_partial_close_events)),
        "total_regime_forced_closes":    int(sum(path_regime_forced_closes)),
        "total_trail_updates":           int(sum(path_trail_updates)),
        "avg_activation_pnl_usd":        round(
            float(sum(path_avg_activation_pnl)) / max(len(path_avg_activation_pnl), 1), 4
        ),
        "path_scores":       path_scores,
        "path_sharpes":      path_sharpes,
        "path_trades":       path_trades,
        "path_winrates":     path_winrates,
        "path_drawdowns":    path_drawdowns,
        "path_returns":      path_returns,
        "path_pfs":          path_pfs,
        "path_expectancies": path_expectancies,
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



