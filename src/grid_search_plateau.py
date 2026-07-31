"""src/grid_search_plateau.py -- Grid-Sensitivity Plateau optimizer (M5/M15).

Ported verbatim from pipeline_verification_bundle/GoldRegimeX_Explorer.ipynb.
Sweeps ONLY xgb_threshold via CPCV (purge+embargo), selects a plateau center,
trains one frozen model per timeframe, validates IS/OOS, exports the live pkl."""
from __future__ import annotations
import os, sys, json, math, time, itertools, warnings, pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = _REPO_ROOT / "pipeline_verification_bundle"
for _p in (_REPO_ROOT, _BUNDLE, _BUNDLE / "shared"):
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    class BaseEstimator: pass
    class ClassifierMixin: pass
    class StandardScaler:
        def fit_transform(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)
            self.std_[self.std_ == 0] = 1.0
            return (X - self.mean_) / self.std_
        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.mean_) / self.std_

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except Exception:
    HMM_OK = False
    class GaussianHMM:
        def __init__(self, n_components=3, covariance_type="full", n_iter=120, random_state=42, **kwargs):
            # kwargs (e.g. min_covar) ignored by the lightweight fallback.
            self.n_components = int(n_components)
            self.random_state = int(random_state)
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.n_features_ = X.shape[1] if X.ndim == 2 else 1
            return self
        def predict(self, X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            score = np.nan_to_num(X[:, 0], nan=0.0)
            if len(score) >= 3:
                q1, q2 = np.quantile(score, [1/3, 2/3])
            else:
                q1, q2 = 0.0, 0.0
            return np.where(score <= q1, 0, np.where(score <= q2, 1, 2)).astype(int)

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False
    class _FallbackXGBClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def fit(self, X, y):
            y = np.asarray(y, dtype=int)
            counts = np.bincount(y, minlength=3).astype(float)
            if counts.sum() == 0:
                counts[:] = 1.0
            self.prior_ = counts / counts.sum()
            return self
        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            return np.tile(self.prior_, (len(X), 1))
        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1).astype(int)
    class _XGBModule:
        XGBClassifier = _FallbackXGBClassifier
    xgb = _XGBModule()

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

try:
    from joblib import Parallel, delayed
    JOBLIB_OK = True
except Exception:
    JOBLIB_OK = False
    class Parallel:
        def __init__(self, n_jobs=1, backend=None, verbose=0):
            self.n_jobs = n_jobs
        def __call__(self, jobs):
            return [job() if callable(job) else job for job in jobs]
    def delayed(fn):
        def wrapper(*args, **kwargs):
            return lambda: fn(*args, **kwargs)
        return wrapper

try:
    from src.trade_lifecycle import config_for_tf
    from src.risk_manager import BROKER_CONFIGS
    SRC_OK = True
except Exception:
    SRC_OK = False
    def config_for_tf(*args, **kwargs):
        return {}
    BROKER_CONFIGS = {}

# vectorized_backtest belonged to the retired legacy backtester and is unused
# by the grid-search-plateau engine (which uses run_ml_filtered_backtest).
vectorized_backtest = None

from pipeline_verification_bundle.shared.session_filter import SessionFilter, ENABLE_SESSION_FILTER, session_col_from_value
from pipeline_verification_bundle.shared.features import build_features, build_ml_features, feature_hash, ema, true_range, atr, rsi, adx

warnings.filterwarnings("ignore")

ML_TARGET_PARAMS = None
pipeline = {}

def _show(x, *a, **k):
    try:
        if isinstance(x, pd.DataFrame):
            print(x.to_string()); return
    except Exception:
        pass
    print(x)


# == CONFIG (cell 3) ==
# -----------------------------
# Global config
# -----------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Strategy TF contract
EXEC_TF = "M5"
TREND_TF = "M15"

# Account and execution assumptions
INITIAL_BALANCE_CENTS = 1500.0
SPREAD_CAP_POINTS = 40.0          # 40 points max = 4.0 pips
STOP_LOSS_PIPS = 15.0
RR_MULT = 2.0
MAX_POSITIONS_PER_CYCLE = 2
LOT_CYCLE_SMALL = [0.02, 0.02]    # base per-leg lots for the ~15 USD account
# --- Balance-tiered position sizing (evaluated in the live-simulation what-if) ---
# Step each leg's lot UP once REALIZED profit from the 15 USD start reaches this many
# cents. 5000 cents = 50 USD profit => account balance ~6500 cents (65 USD).
PROFIT_SCALE_THRESHOLD_CENTS = 5000.0
LOT_CYCLE_SCALED_OPTIONS = [0.03, 0.04]  # larger per-leg lots to evaluate above the threshold
BALANCE_SCALE_THRESHOLD_CENTS = 5000.0  # legacy alias (profit threshold, cents)
# --- DEPLOYED live-simulation sizing policy (applied ONLY in the Live Trading
#     Simulation cell, NOT in the optimization/sensitivity grid above) ---
LIVE_ENABLE_LOT_SCALING = True   # live replay escalates lots once profit clears the threshold
LIVE_SCALED_LOT = 0.03           # per-leg lot used above the threshold in the live replay

# Runtime controls
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

# CPCV-like controls for strategy parameter search
COARSE_CPCV_N_BLOCKS = 4
COARSE_CPCV_K_VAL = 2
FINE_CPCV_N_BLOCKS = 4
FINE_CPCV_K_VAL = 2
EMBARGO_HOURS = 24

# --- EXPERIMENT: time-of-day session filter -------------------------------
# True  => neutralize the NY/London/Asian time-of-day gate. ALL other filters
#          stay ON (trend/ADX, RSI pullback, HMM regime gate, spread cap,
#          ML threshold). Does NOT touch the model, HMM, or features.
# False => restore normal session gating.
DISABLE_SESSION_FILTER = False

# Holdout split
HOLDOUT_FRAC = 0.20
MAX_DATA_YEARS = 5   # Set to int (e.g. 5 or 10) to use only the last N years of data; None = use all available

BARS_PER_DAY = {"M15": 96, "M5": 288}
BARS_PER_YEAR = {"M15": 252 * 96, "M5": 252 * 288}

# Source paths
TF_TO_XAU_RAW = {
    "M15": Path("data/raw/XAU_15m_data.csv"),
    "M5": Path("data/raw/XAU_5m_data.csv"),
}

# Cross-commodity master close files (tab-separated MetaTrader format)
TF_TO_XAG_MASTER = {
    "M15": Path("data/raw/XAGUSD_M15_201601040100_202605072245.csv"),
    "M5":  Path("data/raw/XAGUSD_M5_201601040105_202605072255.csv"),
}
TF_TO_XTI_MASTER = {
    "M15": Path("data/raw/XTIUSD_M15_201702102000_202605072345.csv"),
    "M5":  Path("data/raw/XTIUSD_M5_201702102000_202605072355.csv"),
}

# USDCHF processed master (intraday DXY proxy, ~0.85 DXY correlation).
# Comma-separated, ISO dates - produced by src/data_consolidator.py.
TF_TO_USDCHF_MASTER = {
    "M15": Path("data/processed/USDCHF_master_M15.csv"),
    "M5":  Path("data/processed/USDCHF_master_M5.csv"),
}

# -----------------------------
# Timeframes (Phase 4 / Phase 6)
# -----------------------------
TIMEFRAMES = ["M15", "M5"]

# -----------------------------
# HMM Feature Registry (Phase 3)
# Explicit, version-controlled feature list for the stationary HMM.
# Avoids silently consuming all numeric columns.
# -----------------------------
HMM_FEATURES = {
    "volatility": [
        "atr_20", "atr_normalized", "volatility_20",
        "synth_vix_zscore", "atr14", "atr100", "atr_expansion",
    ],
    "cross_commodity": [
        "log_return", "xag_log_return", "xti_log_return", "usdchf_log_return",
        "gold_silver_ratio_z", "gold_oil_ratio_z", "gold_chf_ratio_z",
    ],
    "oscillators": [
        "rsi5",
    ],
    "temporal": [
        "hour_sin", "hour_cos",
    ],
    "regime": [
        "regime_code",
    ],
}


def get_hmm_feature_list():
    "Flatten the HMM feature registry into a single list."
    features = []
    for group in HMM_FEATURES.values():
        features.extend(group)
    return features


# -----------------------------
# Centralized Train/OOS Split (Phase 5)
# Single consistent split mechanism used by all stages.
# -----------------------------
def split_dataset(df, holdout_frac):
    "Chronological train/OOS split. Returns (train_df, oos_df, split_time)."
    df = df.sort_index().copy()
    if len(df) < 1000:
        raise RuntimeError("Not enough rows for split. Got %d rows." % len(df))
    split_idx = int(len(df) * (1.0 - holdout_frac))
    split_idx = max(2000, min(split_idx, len(df) - 1))
    train = df.iloc[:split_idx].copy()
    oos = df.iloc[split_idx:].copy()
    split_time = train.index[-1]
    return train, oos, split_time


# -----------------------------
# Pipeline Container (Phase 6-7 / Phase 11)
# Each timeframe owns its own state. No shared mutable state.
# -----------------------------
class TimeframePipeline:
    "Owns all state for a single timeframe ML experiment."
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.raw_all = None           # Raw panel data
        self.train_df = None          # IS training data
        self.oos_df = None            # OOS holdout data
        self.split_time = None        # Split timestamp
        self.train_feat = None        # Engineered features (train)
        self.model = None             # Trained HMMXGBComposite
        self.threshold = None         # Optimised xgb_threshold
        self.plateau = None           # Plateau centre dict
        self.metrics_is = None        # IS validation metrics
        self.metrics_oos = None       # OOS validation metrics
        self.trades_is = None         # IS trades dataframe
        self.trades_oos = None        # OOS trades dataframe


# Initialise one pipeline per timeframe


# == bridge loader (cell 5) ==
# ---------------------------------------------------------
# INGESTION LAYER: Import validated parameters from Strategy Tester
# FIX (2026-07-07): if no rows survive the strict DD cap for a timeframe,
# fall back to the row with the lowest max_drawdown available in the CSV
# (with a clear WARNING) instead of raising, so Explorer can proceed.
# The strict-fail path is preserved but only fires when BOTH the strict
# filter AND the lowest-DD fallback yield nothing, which only happens if
# the handoff CSV is truly empty for the timeframe.
# ---------------------------------------------------------
import json

def load_optimized_strategies(filepath="reports/strategy_winners_for_explorer.csv", max_dd=25.0):
    try:
        winners_df = pd.read_csv(filepath)
        print(f"Loaded {len(winners_df)} candidate strategies from bridge.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Bridge file {filepath} not found. Run Strategy_Tester.ipynb first.")

    best_params_by_tf = {}

    for tf in ["M15", "M5"]:
        tf_df = winners_df[winners_df["timeframe"] == tf].copy()
        if tf_df.empty:
            print(f"WARNING: {tf}: bridge CSV has zero rows for this timeframe.")
            best_params_by_tf[tf] = None
            continue

        strict = tf_df[tf_df["max_drawdown"] <= max_dd].copy()
        if not strict.empty:
            top_row = strict.sort_values("robust_score", ascending=False).iloc[0]
            source = "strict"
        else:
            # Fallback: lowest-DD row available for this timeframe.
            top_row = tf_df.sort_values("max_drawdown", ascending=True).iloc[0]
            source = "fallback_lowest_dd"
            print(
                f"WARNING: {tf}: no strategy in bridge CSV survives the {max_dd}% Max DD cap. "
                f"Falling back to lowest-DD available -> DD={top_row['max_drawdown']:.2f}%. "
                f"Re-run Strategy Tester so the handoff includes DD-survivors, "
                f"or lower the DD cap."
            )

        params_dict = json.loads(top_row["parameter_set"])
        best_params_by_tf[tf] = {
            "strategy_name": top_row["strategy_name"],
            "exit_model": top_row["exit_model"],
            "parameters": params_dict,
            "expected_pf": top_row["profit_factor"],
            "expected_dd": top_row["max_drawdown"],
            "selection_source": source,
        }
        tag = " (fallback)" if source != "strict" else ""
        print(
            f"Acquired {tf} Baseline{tag} -> "
            f"PF: {top_row['profit_factor']:.2f} | DD: {top_row['max_drawdown']:.2f}%"
        )

    return best_params_by_tf

# == cell 6 ==
# -----------------------------
# Data loading helpers
# -----------------------------

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    if "open" in cols:
        rename_map[cols["open"]] = "Open"
    if "high" in cols:
        rename_map[cols["high"]] = "High"
    if "low" in cols:
        rename_map[cols["low"]] = "Low"
    if "close" in cols:
        rename_map[cols["close"]] = "Close"
    if "volume" in cols:
        rename_map[cols["volume"]] = "Volume"

    out = df.rename(columns=rename_map)
    need = ["Open", "High", "Low", "Close"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    return out

def read_xau_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=";")
    if "Date" not in df.columns:
        raise ValueError(f"Date column missing in {path}")

    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = _normalize_ohlcv(df)
    return df

def read_mt4_csv(path: Path) -> pd.DataFrame:
    """Read MetaTrader-exported CSV (tab-separated, <UPPERCASE> headers, date+time cols)."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="	")
    # Strip angle brackets from column names: <DATE> → DATE, <CLOSE> → CLOSE etc.
    df.columns = [c.strip("<>") for c in df.columns]
    # Combine DATE + TIME into a datetime index
    df["DateTime"] = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["TIME"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()
    # Normalise to title-case for _normalize_ohlcv compatibility
    rename_map = {}
    for tag in ["OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "VOL", "SPREAD"]:
        if tag in df.columns:
            rename_map[tag] = tag.capitalize()
    # Special: TICKVOL → Volume (since MT4 TICKVOL is the useful volume field)
    if "Tickvol" in df.columns:
        rename_map["Tickvol"] = "Volume"
    df = df.rename(columns=rename_map)
    return df

def read_master_close(path: Path) -> pd.Series:
    """Read commodity series — auto-detects XAU semicolon format vs MT4 tab format."""
    if not path.exists():
        raise FileNotFoundError(path)

    # Peek at first line to detect format
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()

    if "	" in header or header.strip().startswith("<"):
        # MT4 tab-separated format (XAG/XTI)
        df = read_mt4_csv(path)
        col_name = "Close" if "Close" in df.columns else "close"
        return df[col_name].astype(float).sort_index()
    else:
        # Delimited-with-header: XAU semicolon, or processed master (comma, ISO dates).
        sep = ";" if ";" in header else ","
        df = pd.read_csv(path, sep=sep, index_col=0, parse_dates=True).sort_index()
        if "Close" in df.columns:
            s = df["Close"].astype(float)
        elif "close" in df.columns:
            s = df["close"].astype(float)
        else:
            raise ValueError(f"Close column missing in {path}")
        return s

def load_panel(tf: str) -> pd.DataFrame:
    tf = tf.upper()
    xau = read_xau_raw(TF_TO_XAU_RAW[tf])

    out = xau.copy()

    # Cross-commodity series (uses TF_TO_XAG_MASTER / TF_TO_XTI_MASTER from config).
    xag_map = TF_TO_XAG_MASTER
    xti_map = TF_TO_XTI_MASTER

    if isinstance(xag_map, dict) and tf in xag_map and Path(xag_map[tf]).exists():
        xag = read_master_close(Path(xag_map[tf])).rename("XAG_Close")
        out["XAG_Close"] = xag.reindex(out.index).ffill()
    else:
        # Fallback keeps old feature functions from breaking.
        out["XAG_Close"] = out["Close"].astype(float)

    if isinstance(xti_map, dict) and tf in xti_map and Path(xti_map[tf]).exists():
        xti = read_master_close(Path(xti_map[tf])).rename("XTI_Close")
        out["XTI_Close"] = xti.reindex(out.index).ffill()
    else:
        out["XTI_Close"] = out["Close"].astype(float)

    # USDCHF (intraday DXY proxy, ~0.85 DXY correlation) - processed master.
    usdchf_map = globals().get("TF_TO_USDCHF_MASTER")
    if isinstance(usdchf_map, dict) and tf in usdchf_map and Path(usdchf_map[tf]).exists():
        usdchf = read_master_close(Path(usdchf_map[tf])).rename("USDCHF_Close")
        out["USDCHF_Close"] = usdchf.reindex(out.index).ffill()
    else:
        out["USDCHF_Close"] = out["Close"].astype(float)

    # Drop rows where cross-commodity data is NaN
    # (but only if we actually loaded real external data — skip if fallback was used
    #  to avoid dropping the entire early history where XAG/XTI don't exist yet)
    has_real_xag = isinstance(xag_map, dict) and tf in xag_map and Path(xag_map[tf]).exists()
    has_real_xti = isinstance(xti_map, dict) and tf in xti_map and Path(xti_map[tf]).exists()
    if has_real_xag or has_real_xti:
        # Only drop if BOTH external series are NaN — keep rows where at least one exists
        # Forward-fill already applied above; remaining NaN means no data at all for that period
        drop_mask = out["XAG_Close"].isna() & out["XTI_Close"].isna()
        if drop_mask.any():
            print(f"  [{tf}] Dropping {drop_mask.sum()} rows with no XAG or XTI data (before their start dates)")
        out = out[~drop_mask].copy()
        # Fill any remaining single-commodity NaNs with XAU Close as fallback
        out["XAG_Close"] = out["XAG_Close"].fillna(out["Close"].astype(float))
        out["XTI_Close"] = out["XTI_Close"].fillna(out["Close"].astype(float))

    # Keep USDCHF_Close gap-free (leading NaN before its first bar -> XAU Close).
    out["USDCHF_Close"] = out["USDCHF_Close"].fillna(out["Close"].astype(float))

    # --- MAX_DATA_YEARS filter: truncate to last N years ---
    max_years = MAX_DATA_YEARS
    if max_years is not None and isinstance(max_years, (int, float)) and max_years > 0:
        cutoff = out.index.max() - pd.DateOffset(years=int(max_years))
        before_count = len(out)
        out = out.loc[out.index >= cutoff].copy()
        print(f"  [{tf}] MAX_DATA_YEARS={max_years}: {before_count} → {len(out)} rows (cutoff {cutoff:%Y-%m-%d})")

    return out

# == cell 7 ==
# ---------------------------------------------------------
# Feature engineering + labels
# Phase 8/9: indicators, session features and feature engineering now come from
# the SINGLE shared implementation (shared.features / shared.session_filter).
# Only the TBM labelling (triple_barrier) stays local and UNCHANGED.
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from numba import njit

from pipeline_verification_bundle.shared.features import (
    ema, true_range, atr, rsi, adx, synth_vix_zscore, build_ml_features, feature_hash,
)
from pipeline_verification_bundle.shared.session_filter import SessionFilter

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    # Single shared session implementation (Phase 9).
    return SessionFilter().add_session_features(df)

def _add_usdchf_features(exec_panel: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    # USDCHF (intraday DXY proxy) cross-asset features, mirroring the XAG/XTI
    # methodology in shared.features.build_ml_features: log return + rolling
    # z-score (window rw=64) of the gold/USDCHF ratio. Computed on the full
    # panel then aligned to the engineered feature index.
    if "USDCHF_Close" not in exec_panel.columns:
        return feats
    p = exec_panel.copy()
    usdchf_log_return = np.log(p["USDCHF_Close"] / p["USDCHF_Close"].shift(1))
    gold_chf_ratio = p["Close"] / p["USDCHF_Close"]
    rw = 64
    m = gold_chf_ratio.rolling(rw, min_periods=rw).mean()
    s = gold_chf_ratio.rolling(rw, min_periods=rw).std().replace(0, np.nan)
    gold_chf_ratio_z = ((gold_chf_ratio - m) / s).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feats = feats.copy()
    feats["usdchf_log_return"] = usdchf_log_return.reindex(feats.index).fillna(0.0)
    feats["gold_chf_ratio_z"] = gold_chf_ratio_z.reindex(feats.index).fillna(0.0)
    return feats

def build_features(exec_panel: pd.DataFrame, exec_tf: str) -> pd.DataFrame:
    # Single shared feature implementation (Phase 8); TBM applied locally after.
    feats = build_ml_features(exec_panel, exec_tf, spread_cap_points=SPREAD_CAP_POINTS)
    feats = _add_usdchf_features(exec_panel, feats)
    return triple_barrier(feats, exec_tf)

@njit(cache=True)
def _triple_barrier_numba(close, high, low, atr_v, horizon, atr_mult):
    n = len(close)
    label = np.zeros(n, dtype=np.int8)
    event_end_pos = np.arange(n, dtype=np.int32)

    for i in range(n):
        if i + 1 >= n or np.isnan(atr_v[i]):
            label[i] = 0
            event_end_pos[i] = i
            continue

        up = close[i] + atr_mult * atr_v[i]
        dn = close[i] - atr_mult * atr_v[i]
        end_i = min(n - 1, i + horizon)

        hit = 0
        hit_pos = end_i
        for j in range(i + 1, end_i + 1):
            up_hit = high[j] >= up
            dn_hit = low[j] <= dn

            if up_hit and dn_hit:
                hit = 0
                hit_pos = j
                break
            if up_hit:
                hit = 1
                hit_pos = j
                break
            if dn_hit:
                hit = -1
                hit_pos = j
                break

        label[i] = hit
        event_end_pos[i] = hit_pos

    return label, event_end_pos

def triple_barrier(df: pd.DataFrame, tf: str, atr_mult: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    horizon = 12 if tf.upper() == "M5" else 4

    close = out["Close"].to_numpy()
    high = out["High"].to_numpy()
    low = out["Low"].to_numpy()
    atr_v = out["atr_20"].to_numpy()

    label, event_end_pos = _triple_barrier_numba(close, high, low, atr_v, horizon, atr_mult)

    out["tb_label"] = label
    out["event_end_pos"] = event_end_pos
    out["event_end_time"] = out.index[event_end_pos]
    return out

LABEL_COLS = {"tb_label", "event_end_pos", "event_end_time"}

def hmm_feature_columns(df: pd.DataFrame) -> list[str]:
    registry_features = get_hmm_feature_list()
    available = [c for c in registry_features if c in df.columns]
    if not available:
        non_stationary = {"Open", "High", "Low", "Close", "macro_ema50", "macro_ema200"}
        numeric_df = df.select_dtypes(include=['number', 'bool'])
        available = [c for c in numeric_df.columns if c not in LABEL_COLS and c not in non_stationary]
        print("WARNING: HMM feature registry empty, fell back to %d dynamic columns" % len(available))
    return available


# == cell 8 ==
# -----------------------------
# CPCV splitter with Purge + Embargo
# -----------------------------

@dataclass
class CPCVPurgedEmbargo:
    n_blocks: int = 6
    k_val_blocks: int = 2
    embargo_bars: int = 96

    def split(self, n: int, event_end_pos: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        all_idx = np.arange(n, dtype=np.int32)
        blocks = [np.array(x, dtype=np.int32) for x in np.array_split(all_idx, self.n_blocks)]
        block_ranges = [(b[0], b[-1]) for b in blocks if len(b) > 0]

        block_ids = list(range(len(blocks)))
        for combo in itertools.combinations(block_ids, self.k_val_blocks):
            val_blocks = [blocks[c] for c in combo]
            val_idx = np.sort(np.concatenate(val_blocks))

            val_ranges = [block_ranges[c] for c in combo]
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]

            # Purge overlap with label windows
            keep = np.ones(len(train_idx), dtype=bool)
            for k, i in enumerate(train_idx):
                e_i = int(event_end_pos[i])
                for v_start, v_end in val_ranges:
                    if (i <= v_end) and (e_i >= v_start):
                        keep[k] = False
                        break
            train_idx = train_idx[keep]

            # Embargo after each validation range
            embargo_mask = np.zeros(n, dtype=bool)
            for _, v_end in val_ranges:
                e_s = v_end + 1
                e_e = min(n - 1, v_end + self.embargo_bars)
                if e_s <= e_e:
                    embargo_mask[e_s:e_e + 1] = True
            train_idx = train_idx[~embargo_mask[train_idx]]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield np.sort(train_idx), val_idx

# == cell 9 ==
# -----------------------------
# HMM + XGBoost composite
# -----------------------------

class HMMXGBComposite(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_components=3,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
    ):
        self.n_components = int(n_components)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

        self.hmm = None
        self.hmm_scaler = None
        self.xgb = None
        self.feature_names_ = None
        self.hmm_features_ = None

        self.y_to_cls_ = {-1: 0, 0: 1, 1: 2}
        self.cls_to_y_ = {0: -1, 1: 0, 2: 1}

    def fit(self, X: pd.DataFrame, y: pd.Series, hmm_features: list[str]):
        X = X.copy()
        y = pd.Series(y).astype(int)

        # FIX: Strictly isolate numeric columns to prevent strings like 'TREND' from crashing XGBoost
        X_numeric = X.select_dtypes(include=['number', 'bool'])
        self.feature_names_ = list(X_numeric.columns)
        self.hmm_features_ = list(hmm_features)

        X_hmm = X[self.hmm_features_].to_numpy(dtype=float)
        self.hmm_scaler = StandardScaler()
        X_hmm = self.hmm_scaler.fit_transform(X_hmm)

        # Robust HMM fit with convergence-aware fallback
        # Strategy: try multiple covariance configs; check monitor_.converged
        # after fit.  If a config does not converge, skip to the next one.
        # If none converge, use the one with the highest final log-likelihood.
        import logging

        # Temporarily suppress hmmlearn's "Model is not converging" logger noise
        _hmm_logger = logging.getLogger("hmmlearn")
        _prev_level = _hmm_logger.level
        _hmm_logger.setLevel(logging.CRITICAL + 1)

        covar_configs = [
            {"covariance_type": "diag", "min_covar": 0.01},
            {"covariance_type": "diag", "min_covar": 0.1},
            {"covariance_type": "full", "min_covar": 0.01},
            {"covariance_type": "full", "min_covar": 0.1},
            {"covariance_type": "spherical", "min_covar": 0.01},
        ]
        self.hmm = None
        last_err = None
        best_hmm = None
        best_ll = -np.inf

        for _cfg in covar_configs:
            try:
                _hmm = GaussianHMM(
                    n_components=self.n_components,
                    n_iter=300,
                    tol=1e-3,
                    random_state=self.random_state,
                    **_cfg,
                )
                _hmm.fit(X_hmm)
                _did_converge = bool(_hmm.monitor_.converged)
                _regimes = _hmm.predict(X_hmm)
                _ll = _hmm.score(X_hmm)

                if _did_converge:
                    self.hmm = _hmm
                    regimes = _regimes
                    last_err = None
                    _ct = _cfg["covariance_type"]
                    _mc = _cfg["min_covar"]
                    print(f"  HMM converged ({_ct}/{_mc}) LL={_ll:.1f}")
                    break
                else:
                    # Non-converged but usable - keep as fallback if LL is best
                    if _ll > best_ll:
                        best_ll = _ll
                        best_hmm = (_hmm, _regimes, _cfg)
                    continue

            except Exception as _e:
                last_err = _e
                continue

        # Restore hmmlearn logger level
        _hmm_logger.setLevel(_prev_level)

        # If no config fully converged, use the best non-converged one
        if self.hmm is None and best_hmm is not None:
            self.hmm = best_hmm[0]
            regimes = best_hmm[1]
            _cfg = best_hmm[2]
            _ct = _cfg["covariance_type"]
            _mc = _cfg["min_covar"]
            print(f"  HMM: no config fully converged; best LL ({_ct}/{_mc}) LL={best_ll:.1f}")

        if self.hmm is None:
            raise RuntimeError(
                f"HMM fit failed for all covariance configs: {last_err}"
            ) from last_err
        reg_oh = np.eye(self.n_components, dtype=float)[regimes]

        # FIX: Use the safely isolated X_numeric array instead of the raw X dataframe
        X_xgb = np.hstack([X_numeric.to_numpy(dtype=float), reg_oh])
        y_cls = y.map(self.y_to_cls_).to_numpy(dtype=int)

        self.xgb = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=self.random_state,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,
        )
        self.xgb.fit(X_xgb, y_cls)
        return self

    def _augment(self, X: pd.DataFrame) -> np.ndarray:
        # Use self.feature_names_ which now safely contains ONLY numeric features
        X_num = X[self.feature_names_].to_numpy(dtype=float)
        X_hmm = X[self.hmm_features_].to_numpy(dtype=float)
        X_hmm = self.hmm_scaler.transform(X_hmm)
        regimes = self.hmm.predict(X_hmm)
        reg_oh = np.eye(self.n_components, dtype=float)[regimes]
        return np.hstack([X_num, reg_oh])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aug = self._augment(X)
        cls = self.xgb.predict(X_aug).astype(int)
        y = np.array([self.cls_to_y_[int(c)] for c in cls], dtype=np.int8)
        return y

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        X_aug = self._augment(X)
        return self.xgb.predict_proba(X_aug)


# == cell 10 ==
# -----------------------------
# Numba Engine + ML Trend Pullback Wrapper
# -----------------------------
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

# Position split constants (required by numba engine body)
POSITION_A = 0.02
POSITION_B = 0.02
# --- Balance-tiered position sizing globals (baked into numba at compile; call
#     _run_backtest_numba.recompile() after changing them). OFF by default so every
#     optimization/backtest stays fixed-lot and byte-identical; only the live-simulation
#     what-if turns this on. ---
ENABLE_LOT_SCALING = False
PROFIT_SCALE_THRESHOLD_CENTS = 5000.0   # step up once realized profit >= 5000 cents ($50)
SCALED_POSITION_A = 0.03
SCALED_POSITION_B = 0.03
LEG_C_ATR_STOP = 0.5
LEG_C_ATR_TARGET = 0.5

PIP_SIZE_PRICE = 0.10
PIP_VALUE_CENTS_PER_1LOT = 100.0
SLIPPAGE_PIPS = 0.30
COMMISSION_CENTS_PER_TRADE = 0.0
# Intrabar policy for candles where BOTH stop-loss and take-profit are touched.
#   0 = STOP_FIRST (conservative, engine default)   1 = TARGET_FIRST
INTRABAR_POLICY_CODE = 0

@njit(cache=True)
def _run_backtest_numba(
    sig, high, low, close, spread, atr14, regime_code, time_minutes, 
    entry_atr_stop, entry_atr_target, leg_a_atr_target, exit_mode_code, 
    time_stop_minutes, trail_mult, initial_balance, pip_size_price, 
    pip_value_per_1lot, slippage_pips, spread_cap_points, commission_cents, is_m5, open_
):
    n = sig.shape[0]
    max_trades = n * 3

    trade_entry_idx = np.empty(max_trades, dtype=np.int64)
    trade_exit_idx = np.empty(max_trades, dtype=np.int64)
    trade_leg = np.empty(max_trades, dtype=np.int8)
    trade_side = np.empty(max_trades, dtype=np.int8)
    trade_entry_px = np.empty(max_trades, dtype=np.float64)
    trade_exit_px = np.empty(max_trades, dtype=np.float64)
    trade_move_pips = np.empty(max_trades, dtype=np.float64)
    trade_pnl_cents = np.empty(max_trades, dtype=np.float64)
    trade_reason = np.empty(max_trades, dtype=np.int8)
    trade_entry_regime = np.empty(max_trades, dtype=np.int32)
    trade_lot = np.empty(max_trades, dtype=np.float64)  # exact lot per closed trade

    eq = np.empty(max_trades + 1, dtype=np.float64)
    eq[0] = initial_balance
    eq_count = 1

    # MT5-style bar-by-bar mark-to-market series (one value per candle)
    bar_equity = np.empty(n, dtype=np.float64)
    bar_balance = np.empty(n, dtype=np.float64)
    bar_count = 0

    legs = np.zeros((3, 7), dtype=np.float64)
    trade_count = 0
    balance = initial_balance
    active_trade_regime = 0
    leg_a_profit_hit = 0
    leg_b_profit_hit = 0

    enable_fixed_tp = exit_mode_code in (0, 2)
    enable_mr = exit_mode_code in (1, 2, 3, 4)
    enable_time_stop = exit_mode_code == 4
    enable_trail = exit_mode_code == 4

    for i in range(n):
        # 1. STRICT INSOLVENCY CIRCUIT BREAKER
        # If account drops below 1000 cents ($10.00), the broker issues a margin call. Dead.
        if balance <= 1000.0:
            break

        s = int(sig[i])
        current_regime = int(regime_code[i])

        is_flat = (legs[0, 0] == 0.0 and legs[1, 0] == 0.0 and legs[2, 0] == 0.0)
        leg_a_closed_b_open = (legs[0, 0] == 0.0 and legs[1, 0] == 1.0 and leg_a_profit_hit == 1)
        leg_b_closed_a_open = (legs[1, 0] == 0.0 and legs[0, 0] == 1.0 and leg_b_profit_hit == 1)
        can_scale_in = (legs[2, 0] == 0.0 and (leg_a_closed_b_open or leg_b_closed_a_open))

        # ENTRY & SCALE-IN LOGIC
        if (is_flat or can_scale_in) and s != 0:
            sp_points = spread[i]
            atr_now = atr14[i]
            
            effective_spread = (sp_points / 10.0) * pip_size_price
            if np.isfinite(atr_now) and atr_now > 0.0 and effective_spread <= (0.8 * atr_now):
                side = 1 if s > 0 else -1

                # Balance-tiered lots: escalate each leg once realized profit clears the threshold.
                if ENABLE_LOT_SCALING and (balance - initial_balance) >= PROFIT_SCALE_THRESHOLD_CENTS:
                    cur_lot_a = SCALED_POSITION_A
                    cur_lot_b = SCALED_POSITION_B
                else:
                    cur_lot_a = POSITION_A
                    cur_lot_b = POSITION_B

                if can_scale_in:
                    runner_idx = 1 if leg_a_closed_b_open else 0
                    if side != int(legs[runner_idx, 2]):
                        continue 

                spread_price = effective_spread
                slippage_price = slippage_pips * pip_size_price
                close_px = close[i]

                # RISK MANAGEMENT: Dynamic Asymmetric Guard Factor
                if is_m5 == 1:
                    guard_factor = 0.85 if side == 1 else 0.75
                else:
                    guard_factor = 0.65

                if is_flat:
                    leg_a_profit_hit = 0
                    leg_b_profit_hit = 0
                    
                    intended_stop_dist = entry_atr_stop * atr_now
                    actual_stop_dist = intended_stop_dist * guard_factor
                    tp_dist = entry_atr_target * atr_now * guard_factor

                    if side == 1:
                        entry_px = close_px + spread_price + slippage_price
                        stop_px = entry_px - actual_stop_dist
                        fixed_tp_px = entry_px + tp_dist
                        leg_a_tp_px = entry_px + (leg_a_atr_target * atr_now) if leg_a_atr_target > 0.0 else np.nan
                    else:
                        entry_px = close_px - slippage_price
                        stop_px = entry_px + actual_stop_dist + spread_price
                        fixed_tp_px = entry_px - tp_dist
                        leg_a_tp_px = entry_px - (leg_a_atr_target * atr_now) if leg_a_atr_target > 0.0 else np.nan

                    active_trade_regime = current_regime
                    legs[0, 0], legs[0, 1], legs[0, 2], legs[0, 3], legs[0, 4], legs[0, 5], legs[0, 6] = 1.0, cur_lot_a, side, entry_px, stop_px, leg_a_tp_px, i
                    legs[1, 0], legs[1, 1], legs[1, 2], legs[1, 3], legs[1, 4], legs[1, 5], legs[1, 6] = 1.0, cur_lot_b, side, entry_px, stop_px, fixed_tp_px if enable_fixed_tp else np.nan, i

                elif can_scale_in:
                    leg_c_lot = cur_lot_a if leg_a_closed_b_open else cur_lot_b
                    
                    tighter_stop = 0.5 * atr_now
                    actual_stop_dist = tighter_stop * guard_factor
                    tighter_tp = 0.5 * atr_now * guard_factor

                    if side == 1:
                        entry_px = close_px + spread_price + slippage_price
                        stop_px = entry_px - actual_stop_dist
                        fixed_tp_px = entry_px + tighter_tp
                    else:
                        entry_px = close_px - slippage_price
                        stop_px = entry_px + actual_stop_dist + spread_price
                        fixed_tp_px = entry_px - tighter_tp

                    legs[2, 0], legs[2, 1], legs[2, 2], legs[2, 3], legs[2, 4], legs[2, 5], legs[2, 6] = 1.0, leg_c_lot, side, entry_px, stop_px, fixed_tp_px, i

        bar_high, bar_low, bar_close, atr_now = high[i], low[i], close[i], atr14[i]
        bar_open = open_[i]

        # EXIT PROCESSING
        for leg_idx in range(3):
            if legs[leg_idx, 0] == 0.0:
                continue

            leg_side = int(legs[leg_idx, 2])
            leg_entry = legs[leg_idx, 3]
            leg_stop = legs[leg_idx, 4]
            leg_tp = legs[leg_idx, 5]
            leg_lot = legs[leg_idx, 1]
            leg_entry_i = int(legs[leg_idx, 6])

            if enable_trail and leg_idx == 1 and np.isfinite(atr_now) and atr_now > 0.0:
                dist = trail_mult * atr_now
                if leg_side == 1:
                    if bar_close - dist > leg_stop:
                        leg_stop = bar_close - dist
                else:
                    if bar_close + dist < leg_stop:
                        leg_stop = bar_close + dist
                legs[leg_idx, 4] = leg_stop

            # 1. Stop Loss Hit  (intrabar policy resolves both-touched candles)
            stop_hit = (bar_low <= leg_stop) if leg_side == 1 else (bar_high >= leg_stop)
            tp_also_hit = False
            if np.isfinite(leg_tp):
                tp_also_hit = (bar_high >= leg_tp) if leg_side == 1 else (bar_low <= leg_tp)
            take_stop = stop_hit
            if stop_hit and tp_also_hit:
                if INTRABAR_POLICY_CODE == 1:
                    take_stop = False  # TARGET_FIRST: let the target fill first
                elif INTRABAR_POLICY_CODE == 2:
                    # OPEN_TO_CLOSE_SEQUENCE: reconstruct the intrabar path from candle shape.
                    # Bull candle (close>=open): O->L->H->C ; bear candle: O->H->L->C.
                    bullish = bar_close >= bar_open
                    take_stop = bullish if leg_side == 1 else (not bullish)
            if take_stop:
                exit_px = leg_stop
                move_pips = ((exit_px - leg_entry) / pip_size_price) * leg_side
                pnl_cents = move_pips * (leg_lot * pip_value_per_1lot) - commission_cents
                trade_entry_idx[trade_count] = leg_entry_i
                trade_exit_idx[trade_count] = i
                trade_leg[trade_count] = leg_idx
                trade_side[trade_count] = leg_side
                trade_entry_px[trade_count] = leg_entry
                trade_exit_px[trade_count] = exit_px
                trade_move_pips[trade_count] = move_pips
                trade_pnl_cents[trade_count] = pnl_cents
                trade_reason[trade_count] = 1
                trade_entry_regime[trade_count] = active_trade_regime
                trade_lot[trade_count] = leg_lot
                trade_count += 1
                balance += pnl_cents
                eq[eq_count] = balance
                eq_count += 1
                legs[leg_idx, 0] = 0.0
                if leg_idx == 1: leg_a_profit_hit = 0
                continue

            # 2. Leg 0 (A) Profit Target
            if leg_idx == 0 and np.isfinite(leg_tp):
                tp_hit = (bar_high >= leg_tp) if leg_side == 1 else (bar_low <= leg_tp)
                if tp_hit:
                    exit_px = leg_tp
                    move_pips = ((exit_px - leg_entry) / pip_size_price) * leg_side
                    pnl_cents = move_pips * (leg_lot * pip_value_per_1lot) - commission_cents
                    trade_entry_idx[trade_count] = leg_entry_i
                    trade_exit_idx[trade_count] = i
                    trade_leg[trade_count] = leg_idx
                    trade_side[trade_count] = leg_side
                    trade_entry_px[trade_count] = leg_entry
                    trade_exit_px[trade_count] = exit_px
                    trade_move_pips[trade_count] = move_pips
                    trade_pnl_cents[trade_count] = pnl_cents
                    trade_reason[trade_count] = 2
                    trade_entry_regime[trade_count] = active_trade_regime
                    trade_lot[trade_count] = leg_lot
                    trade_count += 1
                    balance += pnl_cents
                    eq[eq_count] = balance
                    eq_count += 1
                    legs[leg_idx, 0] = 0.0
                    leg_a_profit_hit = 1
                    continue

            # 3. Leg B or Leg C Fixed TP
            if leg_idx in (1, 2) and enable_fixed_tp and np.isfinite(leg_tp):
                tp_hit = (bar_high >= leg_tp) if leg_side == 1 else (bar_low <= leg_tp)
                if tp_hit:
                    exit_px = leg_tp
                    move_pips = ((exit_px - leg_entry) / pip_size_price) * leg_side
                    pnl_cents = move_pips * (leg_lot * pip_value_per_1lot) - commission_cents
                    trade_entry_idx[trade_count] = leg_entry_i
                    trade_exit_idx[trade_count] = i
                    trade_leg[trade_count] = leg_idx
                    trade_side[trade_count] = leg_side
                    trade_entry_px[trade_count] = leg_entry
                    trade_exit_px[trade_count] = exit_px
                    trade_move_pips[trade_count] = move_pips
                    trade_pnl_cents[trade_count] = pnl_cents
                    trade_reason[trade_count] = 3
                    trade_entry_regime[trade_count] = active_trade_regime
                    trade_lot[trade_count] = leg_lot
                    trade_count += 1
                    balance += pnl_cents
                    eq[eq_count] = balance
                    eq_count += 1
                    legs[leg_idx, 0] = 0.0
                    if leg_idx == 1:
                        leg_b_profit_hit = 1
                    continue

            # 4. Structural MR Exit (Clears all active legs)
            if leg_idx == 1 and enable_mr and current_regime == 3:
                for l_id in range(3):
                    if legs[l_id, 0] == 1.0:
                        m_pips = ((bar_close - legs[l_id, 3]) / pip_size_price) * int(legs[l_id, 2])
                        p_cents = m_pips * (legs[l_id, 1] * pip_value_per_1lot) - commission_cents
                        trade_entry_idx[trade_count] = int(legs[l_id, 6])
                        trade_exit_idx[trade_count] = i
                        trade_leg[trade_count] = l_id
                        trade_side[trade_count] = int(legs[l_id, 2])
                        trade_entry_px[trade_count] = legs[l_id, 3]
                        trade_exit_px[trade_count] = bar_close
                        trade_move_pips[trade_count] = m_pips
                        trade_pnl_cents[trade_count] = p_cents
                        trade_reason[trade_count] = 4
                        trade_entry_regime[trade_count] = active_trade_regime
                        trade_lot[trade_count] = legs[l_id, 1]
                        trade_count += 1
                        balance += p_cents
                        legs[l_id, 0] = 0.0
                eq[eq_count] = balance
                eq_count += 1
                leg_a_profit_hit = 0
                leg_b_profit_hit = 0
                break

            # 5. Dynamic Time Stop (Clears all active legs)
            if leg_idx == 1 and enable_time_stop and time_stop_minutes > 0.0:
                if (time_minutes[i] - time_minutes[leg_entry_i]) >= time_stop_minutes:
                    for l_id in range(3):
                        if legs[l_id, 0] == 1.0:
                            m_pips = ((bar_close - legs[l_id, 3]) / pip_size_price) * int(legs[l_id, 2])
                            p_cents = m_pips * (legs[l_id, 1] * pip_value_per_1lot) - commission_cents
                            trade_entry_idx[trade_count] = int(legs[l_id, 6])
                            trade_exit_idx[trade_count] = i
                            trade_leg[trade_count] = l_id
                            trade_side[trade_count] = int(legs[l_id, 2])
                            trade_entry_px[trade_count] = legs[l_id, 3]
                            trade_exit_px[trade_count] = bar_close
                            trade_move_pips[trade_count] = m_pips
                            trade_pnl_cents[trade_count] = p_cents
                            trade_reason[trade_count] = 5
                            trade_entry_regime[trade_count] = active_trade_regime
                            trade_lot[trade_count] = legs[l_id, 1]
                            trade_count += 1
                            balance += p_cents
                            legs[l_id, 0] = 0.0
                    eq[eq_count] = balance
                    eq_count += 1
                    leg_a_profit_hit = 0
                    leg_b_profit_hit = 0
                    break

        # --- MT5 mark-to-market: record equity & balance at candle close ---
        floating = 0.0
        for lz in range(3):
            if legs[lz, 0] == 1.0:
                mp = ((bar_close - legs[lz, 3]) / pip_size_price) * int(legs[lz, 2])
                floating += mp * (legs[lz, 1] * pip_value_per_1lot)
        bar_balance[bar_count] = balance
        bar_equity[bar_count] = balance + floating
        bar_count += 1

    # EOD Cleanup
    if legs[0, 0] == 1.0 or legs[1, 0] == 1.0 or legs[2, 0] == 1.0:
        last_i = n - 1
        for leg_idx in range(3):
            if legs[leg_idx, 0] == 1.0:
                m_pips = ((close[last_i] - legs[leg_idx, 3]) / pip_size_price) * int(legs[leg_idx, 2])
                p_cents = m_pips * (legs[leg_idx, 1] * pip_value_per_1lot) - commission_cents
                trade_entry_idx[trade_count] = int(legs[leg_idx, 6])
                trade_exit_idx[trade_count] = last_i
                trade_leg[trade_count] = leg_idx
                trade_side[trade_count] = int(legs[leg_idx, 2])
                trade_entry_px[trade_count] = legs[leg_idx, 3]
                trade_exit_px[trade_count] = close[last_i]
                trade_move_pips[trade_count] = m_pips
                trade_pnl_cents[trade_count] = p_cents
                trade_reason[trade_count] = 6
                trade_entry_regime[trade_count] = active_trade_regime
                trade_lot[trade_count] = legs[leg_idx, 1]
                trade_count += 1
                balance += p_cents
        eq[eq_count] = balance
        eq_count += 1

    return (
        trade_entry_idx, trade_exit_idx, trade_leg, trade_lot, trade_side, trade_entry_px,
        trade_exit_px, trade_move_pips, trade_pnl_cents, trade_reason,
        trade_entry_regime, trade_count, eq, eq_count, balance,
        bar_equity, bar_balance, bar_count
    )

def compute_metrics(trades_df: pd.DataFrame, equity_curve: list[float], initial_balance: float, ending_balance: float) -> dict:
    if trades_df.empty:
        return {
            "profit_factor": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, 
            "max_drawdown": 0.0, "expectancy": 0.0, "win_rate": 0.0, "trade_count": 0, 
            "net_profit": float(ending_balance - initial_balance), "profit_per_trade": 0.0, "net_return_pct": 0.0
        }

    pnl = trades_df["pnl_cents"].to_numpy(dtype=float)
    trade_count = int(len(pnl))
    net_profit = float(np.sum(pnl))
    gross_profit = float(np.sum(pnl[pnl > 0]))
    gross_loss = float(-np.sum(pnl[pnl < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)
    win_rate = float(np.mean(pnl > 0))
    expectancy = float(np.mean(pnl))
    profit_per_trade = float(net_profit / max(trade_count, 1))

    r = pnl / max(initial_balance, 1e-9)
    r_std = float(np.std(r, ddof=0))
    sharpe = float((np.mean(r) / r_std) * math.sqrt(len(r))) if r_std > 0 else 0.0

    downside = r[r < 0]
    d_std = float(np.std(downside, ddof=0)) if len(downside) > 0 else 0.0
    sortino = float((np.mean(r) / d_std) * math.sqrt(len(r))) if d_std > 0 else 0.0

    eq = np.array(equity_curve, dtype=float)
    
    # RISK MANAGEMENT: Drawdown calculations based on an average of multiple positions
    window = min(3, len(eq))
    avg_eq = eq  # use raw equity for true peak-to-trough drawdown
    peaks = np.maximum.accumulate(avg_eq)
    dd = peaks - avg_eq
    max_dd_pct = float(np.max(dd / np.maximum(peaks, 1e-9)) * 100.0) if len(dd) > 0 else 0.0
    
    net_return_pct = float((ending_balance / initial_balance - 1.0) * 100.0)
    calmar = float(net_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

    return {
        "profit_factor": profit_factor, "sharpe": sharpe, "sortino": sortino, "calmar": calmar, 
        "max_drawdown": max_dd_pct, "expectancy": expectancy, "win_rate": win_rate, "trade_count": trade_count, 
        "net_profit": net_profit, "profit_per_trade": profit_per_trade, "net_return_pct": net_return_pct
    }


def score_metrics(met: dict) -> float:
    return float(met.get("sharpe", 0.0))

def _legacy_session_col_from_value(session_filter):
    if session_filter is None:
        return "session_mask_none"
    s = str(session_filter).lower()
    if s == "london":
        return "session_mask_london"
    if s == "ny":
        return "session_mask_ny"
    if s == "london_ny":
        return "session_mask_london_ny"
    raise ValueError(f"Unsupported session_filter: {session_filter}")

class TrendPullbackStrategy:
    name = "trend_pullback"

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        adx_threshold = float(params.get("adx_threshold", 15.0))
        pullback_rsi = float(params.get("pullback_rsi", 40.0))
        confirmation_bars = int(params.get("confirmation_bars", 1))
        session_filter = params.get("session_filter", None)
        session_col = session_col_from_value(session_filter)
        # EXPERIMENT: neutralize the time-of-day gate when disabled (keeps other filters)
        if globals().get("DISABLE_SESSION_FILTER", False):
            session_col = "session_mask_none"

        trend_up_raw = (df["macro_ema50"] > df["macro_ema200"]) & (df["macro_adx14"] > adx_threshold)
        trend_dn_raw = (df["macro_ema50"] < df["macro_ema200"]) & (df["macro_adx14"] > adx_threshold)

        if confirmation_bars > 1:
            trend_up = trend_up_raw.rolling(confirmation_bars, min_periods=confirmation_bars).sum().eq(confirmation_bars)
            trend_dn = trend_dn_raw.rolling(confirmation_bars, min_periods=confirmation_bars).sum().eq(confirmation_bars)
        else:
            trend_up, trend_dn = trend_up_raw, trend_dn_raw

        long_cond = trend_up & (df["rsi5"] < pullback_rsi) & df[session_col].astype(bool)
        short_cond = trend_dn & (df["rsi5"] > (100.0 - pullback_rsi)) & df[session_col].astype(bool)

        sig = pd.Series(0, index=df.index, dtype=np.int8)
        sig.loc[long_cond.fillna(False)] = 1
        sig.loc[short_cond.fillna(False)] = -1
        return sig

def run_ml_filtered_backtest(timeframe: str, df: pd.DataFrame, ml_probs: np.ndarray, base_params: dict, xgb_threshold: float):
    strategy = TrendPullbackStrategy()
    raw_signals = strategy.generate_signals(df, base_params)
    # Route signals: only trade TREND regime (regime_code == 1) on lower timeframes
    trend_mask = df["regime_code"].to_numpy() == 1
    raw_signals = raw_signals.where(trend_mask, 0)
    
    
    raw_arr = raw_signals.to_numpy(dtype=np.int8)
    prob_down = ml_probs[:, 0]
    prob_up = ml_probs[:, 2]
    
    filtered_arr = raw_arr.copy()
    filtered_arr[(raw_arr == 1) & (prob_up < xgb_threshold)] = 0
    filtered_arr[(raw_arr == -1) & (prob_down < xgb_threshold)] = 0

    idx = df.index
    time_minutes = (idx.view("int64") // 60000000000).astype(np.int64)
    sig = filtered_arr
    
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)
    open_r = df["Open"].to_numpy(dtype=np.float64) if "Open" in df.columns else close
    spread = df["spread"].fillna(40.0).to_numpy(dtype=np.float64)
    atr14 = df["atr14"].to_numpy(dtype=np.float64)
    regime_code = df["regime_code"].to_numpy(dtype=np.int32)
    
    is_m5 = 1 if timeframe == "M5" else 0
    exit_model_map = {"fixed_tp": 0, "mr_exit": 1, "fixed_tp_plus_mr": 2, "partial_tp_plus_mr": 3, "partial_tp_mr_time_stop": 4}
    exit_mode_code = int(exit_model_map.get(base_params.get("exit_model", "fixed_tp"), 0))

    initial_bal = INITIAL_BALANCE_CENTS

    # BRIDGE REPAIR: Smart-check multiple common dictionary key variants exported by Strategy Tester
    sl_dist = float(base_params.get("atr_stop", base_params.get("sl_atr", base_params.get("atr_mult_sl", base_params.get("stop_loss_multiplier", 2.0)))))
    tp_dist = float(base_params.get("atr_target", base_params.get("tp_atr", base_params.get("atr_mult_tp", base_params.get("take_profit_multiplier", 2.0)))))
    leg_a_tp = float(base_params.get("leg_a_atr_target", base_params.get("leg_a_tp", 1.0)))

    (trade_entry_idx, trade_exit_idx, trade_leg, trade_lot, trade_side, trade_entry_px,
     trade_exit_px, trade_move_pips, trade_pnl_cents, trade_reason,
     trade_entry_regime, trade_count, eq, eq_count, ending_balance,
     _bar_equity, _bar_balance, _bar_count) = _run_backtest_numba(
        sig, high, low, close, spread, atr14, regime_code, time_minutes,
        sl_dist, tp_dist, leg_a_tp, exit_mode_code, 
        float(base_params.get("time_stop_minutes", -1.0)), float(base_params.get("trail_mult", 0.0)),
        float(initial_bal), 0.10, 100.0, 0.30, 40.0, 0.0, int(is_m5), open_r
    )

    if trade_count == 0:
        return pd.DataFrame(), compute_metrics(pd.DataFrame(), [], initial_bal, float(ending_balance))
    
    trades_df = pd.DataFrame({"pnl_cents": trade_pnl_cents[:trade_count]})
    return trades_df, compute_metrics(trades_df, eq[:eq_count].tolist(), initial_bal, float(ending_balance))

# == cell 11 ==
# -----------------------------
# CPCV combo evaluator for M15 Trend + M5 Execution strategy
# -----------------------------
import math
import time
import warnings

def _empty_combo_result(xgb_threshold, exec_tf):
    return {
        "timeframe": exec_tf,
        "xgb_threshold": float(xgb_threshold),
        "mean_sharpe": 0.0,
        "mean_sharpe_raw": 0.0,
        "variance_sharpe": 0.0,
        "stability_adjusted_sharpe": 0.0,
        "turnover_penalty": 0.0,
        "mean_trades_per_100": 0.0,
        "n_paths": 0,
        "median_trades": 0,
    }

# Timeframe-scoped cache (Phase 7: no global mutable state)
_ML_FOLD_CACHE = {tf: {} for tf in TIMEFRAMES}

def _get_or_compute_ml_folds(exec_df, n_blocks, k_val_blocks, embargo_bars, exec_tf):
    tf_cache = _ML_FOLD_CACHE.get(exec_tf, {})

    cache_key = f"{exec_tf}_{id(exec_df)}_{n_blocks}_{k_val_blocks}"
    if cache_key in tf_cache:
        return tf_cache[cache_key]

    print(f"\n[!] Pre-computing Features and ML Models for {exec_tf}...")

    t_start = time.time()
    d_exec = exec_df.sort_index().copy()

    base_params = dict(ML_TARGET_PARAMS[exec_tf]["parameters"])
    base_params["exit_model"] = ML_TARGET_PARAMS[exec_tf]["exit_model"]

    feat = build_features(d_exec, exec_tf)
    event_end_pos = feat["event_end_pos"].to_numpy(dtype=np.int32)
    hmm_cols = hmm_feature_columns(feat)

    splitter = CPCVPurgedEmbargo(
        n_blocks=int(n_blocks),
        k_val_blocks=int(k_val_blocks),
        embargo_bars=int(embargo_bars),
    )

    folds_data = []
    splits = list(splitter.split(len(feat), event_end_pos))

    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        if len(train_idx) < 200 or len(val_idx) < 250:
            continue

        print(f" -> Training ML Fold {fold_idx}/{len(splits)}...")
        train_features = feat.iloc[train_idx]
        val_features = feat.iloc[val_idx]
        train_labels = train_features["tb_label"]

        model = HMMXGBComposite(random_state=RANDOM_STATE)
        model.fit(train_features, train_labels, hmm_features=hmm_cols)
        ml_probs = model.predict_proba_raw(val_features)

        folds_data.append((val_features, ml_probs))

    print(f"Pre-computation for {exec_tf} complete in {(time.time()-t_start)/60:.1f} minutes.\n")

    cache_data = {
        "folds_data": folds_data,
        "base_params": base_params
    }
    _ML_FOLD_CACHE[exec_tf][cache_key] = cache_data
    return cache_data

def evaluate_combo_cpcv(
    exec_df: pd.DataFrame,
    xgb_threshold: float,
    n_blocks: int,
    k_val_blocks: int,
    embargo_bars: int,
    exec_tf: str
) -> dict:
    warnings.filterwarnings("ignore")

    if not isinstance(ML_TARGET_PARAMS, dict) or ML_TARGET_PARAMS.get(exec_tf) is None:
        return _empty_combo_result(xgb_threshold, exec_tf)

    cache_data = _get_or_compute_ml_folds(exec_df, n_blocks, k_val_blocks, embargo_bars, exec_tf)
    folds_data = cache_data["folds_data"]
    base_params = cache_data["base_params"]

    path_scores = []
    path_trades = []
    path_t100 = []

    for val_features, ml_probs in folds_data:
        _, met = run_ml_filtered_backtest(
            timeframe=exec_tf,
            df=val_features,
            ml_probs=ml_probs,
            base_params=base_params,
            xgb_threshold=float(xgb_threshold),
        )

        path_scores.append(float(met.get("sharpe", 0.0)))
        n_trades = int(met.get("trade_count", 0))
        path_trades.append(n_trades)
        path_t100.append(float((n_trades / max(len(val_features), 1)) * 100.0))

    if len(path_scores) == 0:
        return _empty_combo_result(xgb_threshold, exec_tf)

    arr = np.array(path_scores, dtype=float)
    mean_raw = float(np.mean(arr))
    var_score = float(np.var(arr))
    mean_t100 = float(np.mean(np.array(path_t100, dtype=float)))

    limit = float(4.0)
    lam = float(3.0)
    excess = max(0.0, mean_t100 - limit)
    turnover_penalty = float(lam * excess)

    mean_score = float(mean_raw - turnover_penalty)
    stab = float(mean_score / (math.sqrt(max(var_score, 1e-12)) + 1e-6))

    return {
        "timeframe": exec_tf,
        "xgb_threshold": float(xgb_threshold),
        "mean_sharpe": float(mean_score),
        "mean_sharpe_raw": float(mean_raw),
        "variance_sharpe": float(var_score),
        "stability_adjusted_sharpe": float(stab),
        "turnover_penalty": float(turnover_penalty),
        "mean_trades_per_100": float(mean_t100),
        "n_paths": int(len(path_scores)),
        "median_trades": int(np.median(np.array(path_trades, dtype=int))),
    }

def run_grid_parallel(
    exec_df: pd.DataFrame,
    grid: list[tuple[float]],
    n_blocks: int,
    k_val_blocks: int,
    embargo_bars: int,
    exec_tf: str,
    n_jobs: int = 1,
) -> pd.DataFrame:
    t0 = time.time()
    _get_or_compute_ml_folds(exec_df, n_blocks, k_val_blocks, embargo_bars, exec_tf)

    out_local = []
    for i, g in enumerate(grid, 1):
        xgb_threshold = float(g[0])
        res = evaluate_combo_cpcv(exec_df, xgb_threshold, n_blocks, k_val_blocks, embargo_bars, exec_tf)
        out_local.append(res)

    df_out = pd.DataFrame(out_local)
    elapsed = time.time() - t0
    print(f"Grid sweep for {exec_tf} complete in {elapsed:.2f} seconds")
    return df_out

# == cell 12 ==
# -----------------------------
# Strategy parameter plateau tools
# -----------------------------

def build_coarse_grid():
    # Grid searches ML confidence threshold
    # Lowered to realistic financial ML probability ranges for minority classes
    xgb_threshold_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    return [(float(x),) for x in xgb_threshold_grid]

def build_refined_grid_from_top(
    coarse_df: pd.DataFrame,
    top_k: int = 3,
    step: float = 0.02,
) -> list[tuple[float]]:
    cd = coarse_df.copy()
    cd = cd[np.isfinite(cd["stability_adjusted_sharpe"])].sort_values(
        ["stability_adjusted_sharpe", "mean_sharpe"], ascending=False
    )
    top = cd.head(top_k)

    refined = set()
    for _, r in top.iterrows():
        center = float(r["xgb_threshold"])
        for delta in (-2 * step, -step, 0.0, step, 2 * step):
            cand = round(center + delta, 4)
            # Bounded to match the lowered threshold reality
            if 0.05 <= cand <= 0.45:
                refined.add((float(cand),))

    return sorted(list(refined))

def plot_plateau_heatmaps(results_df: pd.DataFrame, title_prefix: str = "Strategy Surface"):
    if results_df.empty:
        print("No results to plot")
        return

    d = results_df.copy()
    d = d[np.isfinite(d["mean_sharpe"]) & np.isfinite(d["xgb_threshold"])]
    if d.empty:
        print("No finite results to plot")
        return

    d = d.sort_values("xgb_threshold")
    plt.figure(figsize=(8, 4))
    plt.plot(d["xgb_threshold"], d["mean_sharpe"], marker="o", label="mean_sharpe")
    if "stability_adjusted_sharpe" in d.columns:
        plt.plot(d["xgb_threshold"], d["stability_adjusted_sharpe"], marker="s", label="stability_adjusted_sharpe")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(f"{title_prefix} | xgb_threshold vs score")
    plt.xlabel("xgb_threshold")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def select_plateau_center(fine_df: pd.DataFrame, min_mean_sharpe: float = -1e9, neighbor_width: float = 0.05) -> dict | None:
    d = fine_df.copy()
    d = d[np.isfinite(d["mean_sharpe"]) & np.isfinite(d["stability_adjusted_sharpe"])]
    d = d[d["mean_sharpe"] >= float(min_mean_sharpe)]
    if d.empty:
        return None

    best = None
    best_score = -1e9

    for _, row in d.iterrows():
        thr = float(row["xgb_threshold"])
        neigh = d[d["xgb_threshold"].between(thr - neighbor_width, thr + neighbor_width)]

        plateau_width = int((neigh["mean_sharpe"] > 0).sum())
        local_stab = float(neigh["stability_adjusted_sharpe"].mean())
        local_mean = float(neigh["mean_sharpe"].mean())
        score = plateau_width * 1.5 + local_stab + 0.15 * local_mean

        if score > best_score:
            best_score = score
            best = {
                "xgb_threshold": thr,
                "mean_sharpe": float(row["mean_sharpe"]),
                "mean_sharpe_raw": float(row.get("mean_sharpe_raw", np.nan)),
                "variance_sharpe": float(row["variance_sharpe"]),
                "stability_adjusted_sharpe": float(row["stability_adjusted_sharpe"]),
                "turnover_penalty": float(row.get("turnover_penalty", 0.0)),
                "mean_trades_per_100": float(row.get("mean_trades_per_100", np.nan)),
                "plateau_width_local": plateau_width,
                "selection_score": float(score),
            }

    return best


# == Orchestrator ==
def run_optimization(n_jobs=None):
    global ML_TARGET_PARAMS, N_JOBS, pipeline
    os.chdir(_REPO_ROOT)
    if n_jobs is not None:
        N_JOBS = int(n_jobs)
    _repo_root = _REPO_ROOT
    ML_TARGET_PARAMS = load_optimized_strategies(max_dd=30.0)
    pipeline = {tf: TimeframePipeline(tf) for tf in TIMEFRAMES}
    # -- cell 13 --
    # -----------------------------
    # Main run: load M5 + M15 data and split into pipeline containers
    # Phase 5:  Uses split_dataset for consistent splits
    # Phase 11: Data stored in pipeline container, not loose globals
    # -----------------------------

    for tf in TIMEFRAMES:
        pipeline[tf].raw_all = load_panel(tf).sort_index().copy()

    # --- IS / OOS chronological split ---
    # M5 is the reference: its split determines split_time.
    # M15 aligns to the same split_time boundary so both timeframes
    # share a consistent cut-off.

    m5_is, m5_oos, split_time = split_dataset(pipeline["M5"].raw_all, HOLDOUT_FRAC)
    pipeline["M5"].train_df = m5_is
    pipeline["M5"].oos_df = m5_oos
    pipeline["M5"].split_time = split_time

    # Align M15 to the same split_time boundary
    m15_all_sorted = pipeline["M15"].raw_all.sort_index()
    pipeline["M15"].train_df = m15_all_sorted.loc[m15_all_sorted.index <= split_time].copy()
    pipeline["M15"].oos_df = m15_all_sorted.loc[m15_all_sorted.index > split_time].copy()
    pipeline["M15"].split_time = split_time

    print("=" * 60)
    print("IS / OOS DATA SPLIT SUMMARY")
    print("=" * 60)
    print("Holdout fraction:  %.0f%%  (OOS)" % (HOLDOUT_FRAC * 100))
    print("Split timestamp:  ", split_time)
    for tf in TIMEFRAMES:
        p = pipeline[tf]
        total = len(p.raw_all)
        print("\n  %s total:  %d" % (tf, total))
        print("  %s IS   :  %d   (%.1f%%)   %s -> %s" % (tf, len(p.train_df), len(p.train_df)/total*100, p.train_df.index[0].strftime('%Y-%m-%d'), p.train_df.index[-1].strftime('%Y-%m-%d')))
        print("  %s OOS  :  %d   (%.1f%%)   %s -> %s" % (tf, len(p.oos_df), len(p.oos_df)/total*100, p.oos_df.index[0].strftime('%Y-%m-%d'), p.oos_df.index[-1].strftime('%Y-%m-%d')))
    print()
    print("Note: OOS typically has fewer trades because (1) it covers")
    print("      a shorter calendar span (%.0f%% of data) and (2) regime" % (HOLDOUT_FRAC * 100))
    print("      mix may differ, affecting signal generation.")
    print("=" * 60)

    # Backward-compatible references for cells not yet fully refactored
    m5_train = pipeline["M5"].train_df
    m5_oos = pipeline["M5"].oos_df
    m15_train = pipeline["M15"].train_df
    m15_oos = pipeline["M15"].oos_df
    m5_all = pipeline["M5"].raw_all
    m15_all = pipeline["M15"].raw_all

    # -- cell 14 --
    # -----------------------------
    # Coarse pass
    # -----------------------------
    coarse_grid = build_coarse_grid()
    coarse_results_dict = {}

    for tf in ["M15", "M5"]:
        print(f"\n--- COARSE PASS ({tf}) ---")
        active_train_df = m5_train if tf == "M5" else m15_train

        res = run_grid_parallel(
            exec_df=active_train_df,
            grid=coarse_grid,
            n_blocks=4,
            k_val_blocks=1,
            embargo_bars=12,
            exec_tf=tf,
            n_jobs=N_JOBS
        )

        coarse_results_dict[tf] = res
        print(f"\n{tf} Coarse Results Top 3:")
        _show(res.sort_values(["stability_adjusted_sharpe", "mean_sharpe"], ascending=False).head(3))

    # Combine all results
    coarse_results = pd.concat(coarse_results_dict.values(), ignore_index=True)


    # -- cell 15 --
    # -----------------------------
    # Fine pass
    # -----------------------------
    fine_results_dict = {}

    for tf in ["M15", "M5"]:
        print(f"\n--- FINE PASS ({tf}) ---")
        active_train_df = m5_train if tf == "M5" else m15_train
        tf_coarse = coarse_results[coarse_results["timeframe"] == tf]

        refined_grid = build_refined_grid_from_top(tf_coarse, top_k=3, step=0.02)

        res = run_grid_parallel(
            exec_df=active_train_df,
            grid=refined_grid,
            n_blocks=4,
            k_val_blocks=1,
            embargo_bars=12,
            exec_tf=tf,
            n_jobs=N_JOBS
        )

        fine_results_dict[tf] = res
        print(f"\n{tf} Fine Results Top 3:")
        _show(res.sort_values(["stability_adjusted_sharpe", "mean_sharpe"], ascending=False).head(3))

    # Combine all results
    fine_results = pd.concat(fine_results_dict.values(), ignore_index=True)


    # -- cell 16 --
    # -----------------------------
    # Sanity checks
    # -----------------------------

    print("Fine unique n_paths:", sorted(fine_results["n_paths"].dropna().unique().tolist()))
    print("Fine mean trades/100 (top 10):", fine_results["mean_trades_per_100"].head(10).round(3).tolist())
    print("Fine turnover penalty (top 10):", fine_results["turnover_penalty"].head(10).round(3).tolist())

    print("\nTop 10 with raw vs penalized:")
    _show(
        fine_results[
            [
                "xgb_threshold",
                "mean_sharpe_raw",
                "turnover_penalty",
                "mean_sharpe",
                "stability_adjusted_sharpe",
                "mean_trades_per_100",
            ]
        ].head(10)
    )

    # -- cell 17 --
    # -----------------------------
    # Model lifecycle + Final model selection + IS/OOS validation
    # Phase 2-3: Separate train_ml_model from evaluate_ml_model
    # Phase 4:   No global best timeframe -- both TFs export independently
    # Phase 6-7: Pipeline-scoped registry instead of global variables
    # Phase 8:   Validation loop uses pre-trained models (no retraining)
    # Phase 5:   Consumes split from pipeline container (split_dataset)
    # -----------------------------
    import warnings

    def _safe_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _safe_int(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _normalize_metrics(met, ending_balance):
        out = dict(met)
        out["max_drawdown_pct"] = _safe_float(met.get("max_drawdown", met.get("max_drawdown_pct", 0.0)), 0.0)
        out["n_trades"] = _safe_int(met.get("trade_count", met.get("n_trades", 0)), 0)
        out["ending_balance_cents"] = _safe_float(ending_balance, 0.0)
        return out


    # -----------------------------
    # Phase 2-3: train_ml_model and evaluate_ml_model
    # -----------------------------
    def train_ml_model(train_data, exec_tf):
        "Train an HMMXGBComposite model on training data ONCE."
        warnings.filterwarnings("ignore")
        tf = exec_tf.upper()
        if ML_TARGET_PARAMS.get(tf) is None:
            raise RuntimeError("No bridge baseline for %s" % tf)

        feat = build_features(train_data, tf)
        hmm_feats = hmm_feature_columns(feat)

        model = HMMXGBComposite(random_state=RANDOM_STATE)
        model.fit(feat, feat["tb_label"], hmm_features=hmm_feats)
        print("  [%s] Model trained on %d samples, %d HMM features" % (tf, len(feat), len(hmm_feats)))
        return model, feat


    def evaluate_ml_model(trained_model, data, exec_tf, xgb_threshold):
        "Evaluate a PRE-TRAINED model on provided data. NO training occurs here."
        warnings.filterwarnings("ignore")
        tf = exec_tf.upper()
        if ML_TARGET_PARAMS.get(tf) is None:
            raise RuntimeError("No bridge baseline for %s" % tf)

        base_params = dict(ML_TARGET_PARAMS[tf]["parameters"])
        base_params["exit_model"] = ML_TARGET_PARAMS[tf]["exit_model"]

        feat = build_features(data, tf)

        # Inference only -- no model.fit() call
        probs = trained_model.predict_proba_raw(feat)

        trades, met = run_ml_filtered_backtest(tf, feat, probs, base_params, float(xgb_threshold))
        ending_balance = _safe_float(INITIAL_BALANCE_CENTS + _safe_float(met.get("net_profit", 0.0)), INITIAL_BALANCE_CENTS)
        return trades, _normalize_metrics(met, ending_balance)


    def _select_center_with_fallback(tf_fine):
        "Plateau-center selection with the original fallback logic, per TF."
        center = select_plateau_center(tf_fine, min_mean_sharpe=-1e9)
        if center is not None:
            return center

        print("Standard center selection failed. Attempting fallback...")
        d = tf_fine.copy()
        for col in ["xgb_threshold", "mean_sharpe", "stability_adjusted_sharpe"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")

        d = d[np.isfinite(d["mean_sharpe"]) & np.isfinite(d["stability_adjusted_sharpe"])]
        if d.empty:
            raise RuntimeError("No valid fine results found after filtering.")

        row = d.sort_values(
            ["stability_adjusted_sharpe", "mean_sharpe"],
            ascending=False,
            na_position="last",
        ).iloc[0]

        center = {
            "xgb_threshold": float(row["xgb_threshold"]),
            "mean_sharpe": float(row["mean_sharpe"]),
            "stability_adjusted_sharpe": float(row["stability_adjusted_sharpe"]),
            "selection_score": float(row["stability_adjusted_sharpe"]),
        }
        print("Fallback center selected:")
        print(center)
        return center


    # ---------------------------------------------------------------
    # 1. Select the best plateau center PER timeframe (Phase 4)
    #    NO global "best timeframe" selection.
    #    Each TF stores its own threshold and plateau in the pipeline.
    # ---------------------------------------------------------------
    for tf in TIMEFRAMES:
        tf_fine = fine_results[fine_results["timeframe"] == tf].copy()
        if tf_fine.empty:
            raise RuntimeError("No fine results for timeframe %s." % tf)

        print("\n[%s] Selected plateau center:" % tf)
        center = _select_center_with_fallback(tf_fine)
        print(center)

        thr = _safe_float(center.get("xgb_threshold"), np.nan)
        if np.isnan(thr):
            raise RuntimeError("Invalid xgb_threshold selected for %s: %r" % (tf, center.get('xgb_threshold')))

        pipeline[tf].plateau = center
        pipeline[tf].threshold = thr
        print("[%s] Threshold stored in pipeline: %s" % (tf, thr))

    # ---------------------------------------------------------------
    # 2. Train models PER timeframe (Phase 2-3: train once)
    #    Models are stored in pipeline[tf].model.
    #    They will NOT be retrained during evaluation.
    # ---------------------------------------------------------------
    print("\nTraining models (one per timeframe, frozen after this)...")
    for tf in TIMEFRAMES:
        p = pipeline[tf]
        print("\n[%s] Training on IS data (%d rows)..." % (tf, len(p.train_df)))
        model, feat = train_ml_model(p.train_df, tf)
        p.model = model
        p.train_feat = feat
        print("[%s] Model frozen and stored in pipeline" % tf)

    # ---------------------------------------------------------------
    # 3. Evaluate models: IS and OOS using pre-trained models (Phase 8)
    #    Training must never occur here.
    # ---------------------------------------------------------------
    print("\nRunning final validation segments for both timeframes (inference only)...")

    validation_rows = []

    for tf in TIMEFRAMES:
        p = pipeline[tf]
        thr = float(p.threshold)

        print("\n[%s] threshold=%s  (train rows=%d, oos rows=%d)" % (tf, thr, len(p.train_df), len(p.oos_df)))

        # IS evaluation -- uses pre-trained model
        trades_is, met_is = evaluate_ml_model(p.model, p.train_df, tf, thr)
        p.trades_is = trades_is
        p.metrics_is = met_is

        # OOS evaluation -- uses SAME pre-trained model (no retraining!)
        trades_oos, met_oos = evaluate_ml_model(p.model, p.oos_df, tf, thr)
        p.trades_oos = trades_oos
        p.metrics_oos = met_oos

        score_is = score_metrics(met_is)
        score_oos = score_metrics(met_oos)
        rel_drop = (score_is - score_oos) / (abs(score_is) + 1e-9)

        validation_rows.extend([
            {
                "timeframe": tf,
                "scenario": "IS",
                "xgb_threshold": thr,
                "score": score_is,
                "win_rate": met_is.get("win_rate", 0),
                "profit_factor": met_is.get("profit_factor", 0),
                "max_drawdown_pct": met_is.get("max_drawdown_pct", 0),
                "n_trades": met_is.get("n_trades", 0),
                "ending_balance_cents": met_is.get("ending_balance_cents", 0),
                "net_return_pct": met_is.get("net_return_pct", 0),
                "net_profit_cents": met_is.get("net_profit", 0),
                "profit_per_trade_cents": met_is.get("profit_per_trade", 0),
                "profit_per_trade_usd": met_is.get("profit_per_trade", 0) / 100.0,
                "relative_score_drop_is_to_oos": rel_drop,
            },
            {
                "timeframe": tf,
                "scenario": "OOS",
                "xgb_threshold": thr,
                "score": score_oos,
                "win_rate": met_oos.get("win_rate", 0),
                "profit_factor": met_oos.get("profit_factor", 0),
                "max_drawdown_pct": met_oos.get("max_drawdown_pct", 0),
                "n_trades": met_oos.get("n_trades", 0),
                "ending_balance_cents": met_oos.get("ending_balance_cents", 0),
                "net_return_pct": met_oos.get("net_return_pct", 0),
                "net_profit_cents": met_oos.get("net_profit", 0),
                "profit_per_trade_cents": met_oos.get("profit_per_trade", 0),
                "profit_per_trade_usd": met_oos.get("profit_per_trade", 0) / 100.0,
                "relative_score_drop_is_to_oos": rel_drop,
            },
        ])

    # 4. Output summary
    validation_summary = pd.DataFrame(validation_rows)

    print("\nValidation summary (both timeframes):")
    _show(validation_summary)

    print("\nRelative score drop IS->OOS:")
    for tf in TIMEFRAMES:
        row = validation_summary[
            (validation_summary["timeframe"] == tf) & (validation_summary["scenario"] == "OOS")
        ].iloc[0]
        drop = float(row["relative_score_drop_is_to_oos"])
        warn = "  <- WARNING: >30% drop" if drop > 0.30 else ""
        print("  %s: %.2f%%%s" % (tf, drop * 100, warn))

    if not validation_summary.empty:
        validation_summary.to_csv("reports/final_is_oos_summary.csv", index=False)
        print("\nSaved: notebooks/reports/final_is_oos_summary.csv")

    # -- cell 18 --
    # -----------------------------
    # Dual timeframe validation (consolidated): M15 and M5 independently
    # Phase 4:  Each TF uses its own threshold from pipeline
    # Phase 8:  Uses pre-trained models from pipeline (no retraining)
    # Phase 5:  Uses split_dataset for consistent splits
    # Phase 7:  No globals() calls
    # Phase 11: All data from pipeline container
    # -----------------------------

    def run_mode_v2(tf):
        "Run IS/OOS validation for a single timeframe using pipeline state."
        p = pipeline[tf]
        model = p.model
        thr = float(p.threshold)

        # Re-split from raw_all using centralised split_dataset
        train_df, oos_df, _split_time = split_dataset(p.raw_all, HOLDOUT_FRAC)

        # IS evaluation with pre-trained model
        trades_is, met_is = evaluate_ml_model(model, train_df, tf, thr)
        score_is = score_metrics(met_is)

        # OOS evaluation with SAME pre-trained model
        trades_oos, met_oos = evaluate_ml_model(model, oos_df, tf, thr)
        score_oos = score_metrics(met_oos)

        rel_drop = float((score_is - score_oos) / (abs(score_is) + 1e-9))

        rows = [
            {
                "timeframe_mode": "%s_only" % tf,
                "scenario": "IS",
                "xgb_threshold": thr,
                "score": float(score_is),
                "win_rate": float(met_is.get("win_rate", 0)),
                "profit_factor": float(met_is.get("profit_factor", 0)),
                "max_drawdown_pct": float(met_is.get("max_drawdown_pct", 0)),
                "n_trades": int(met_is.get("trade_count", met_is.get("n_trades", 0))),
                "ending_balance_cents": float(met_is.get("ending_balance_cents", 0)),
                "net_return_pct": float(met_is.get("net_return_pct", 0)),
                "net_profit_cents": float(met_is.get("net_profit", 0)),
                "profit_per_trade_cents": float(met_is.get("profit_per_trade", 0)),
                "profit_per_trade_usd": float(met_is.get("profit_per_trade", 0)) / 100.0,
                "relative_score_drop_is_to_oos": rel_drop,
                "split_time": _split_time,
                "exec_rows": int(len(train_df)),
            },
            {
                "timeframe_mode": "%s_only" % tf,
                "scenario": "OOS",
                "xgb_threshold": thr,
                "score": float(score_oos),
                "win_rate": float(met_oos.get("win_rate", 0)),
                "profit_factor": float(met_oos.get("profit_factor", 0)),
                "max_drawdown_pct": float(met_oos.get("max_drawdown_pct", 0)),
                "n_trades": int(met_oos.get("trade_count", met_oos.get("n_trades", 0))),
                "ending_balance_cents": float(met_oos.get("ending_balance_cents", 0)),
                "net_return_pct": float(met_oos.get("net_return_pct", 0)),
                "net_profit_cents": float(met_oos.get("net_profit", 0)),
                "profit_per_trade_cents": float(met_oos.get("profit_per_trade", 0)),
                "profit_per_trade_usd": float(met_oos.get("profit_per_trade", 0)) / 100.0,
                "relative_score_drop_is_to_oos": rel_drop,
                "split_time": _split_time,
                "exec_rows": int(len(oos_df)),
            },
        ]
        return rows, trades_is, trades_oos


    all_rows = []
    trades_by_mode = {}

    for tf in TIMEFRAMES:
        thr = pipeline[tf].threshold
        print("Running %s validation with threshold=%s" % (tf, thr))
        rows, t_is, t_oos = run_mode_v2(tf)
        all_rows.extend(rows)
        trades_by_mode["%s_only_IS" % tf] = t_is
        trades_by_mode["%s_only_OOS" % tf] = t_oos

    dual_tf_summary = pd.DataFrame(all_rows)
    dual_tf_summary["scenario_order"] = dual_tf_summary["scenario"].map({"IS": 0, "OOS": 1})
    dual_tf_summary = dual_tf_summary.sort_values(["timeframe_mode", "scenario_order"]).drop(columns=["scenario_order"])

    print("\nDual timeframe summary:")
    _show(dual_tf_summary)

    # -- cell 22 --
    # =====================================================================
    # EXPORT DEPLOYED MODEL ARTIFACT  ->  pipeline_verification_bundle/models
    # =====================================================================
    import pickle
    from pathlib import Path

    # --- Locate the bundle folder (repo_root resolved in the first cell) ---
    _BUNDLE_DIR = Path(_repo_root) / "pipeline_verification_bundle"
    _MODEL_DIR = _BUNDLE_DIR / "models"
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_MODEL_PKL = _MODEL_DIR / "goldregimex_live_model.pkl"

    # --- Sanity: models must have been trained (run the training cell first) ---
    _missing = [tf for tf in TIMEFRAMES if getattr(pipeline.get(tf), "model", None) is None]
    if _missing:
        raise RuntimeError(
            "No trained model for %s. Run the training / IS-OOS validation cell "
            "first so pipeline[tf].model is populated." % _missing
        )

    # --- Execution settings taken directly from THIS notebook -----------------
    #     (these are what the live simulation will trade with -- lot size, etc.)
    live_settings = {
        "position_a": float(POSITION_A),                       # leg A lot size
        "position_b": float(POSITION_B),                       # leg B lot size
        "initial_balance_cents": float(INITIAL_BALANCE_CENTS),
        "pip_size_price": float(PIP_SIZE_PRICE),
        "pip_value_cents_per_1lot": float(PIP_VALUE_CENTS_PER_1LOT),
        "slippage_pips": float(SLIPPAGE_PIPS),
        "commission_cents_per_trade": float(COMMISSION_CENTS_PER_TRADE),
        "spread_cap_points": float(SPREAD_CAP_POINTS),
        "lot_cycle_small": list(LOT_CYCLE_SMALL),
        "max_positions_per_cycle": int(MAX_POSITIONS_PER_CYCLE),
        "profit_scale_threshold_cents": float(PROFIT_SCALE_THRESHOLD_CENTS),
        "lot_cycle_scaled_options": list(LOT_CYCLE_SCALED_OPTIONS),
        "live_enable_lot_scaling": bool(LIVE_ENABLE_LOT_SCALING),
        "live_scaled_lot": float(LIVE_SCALED_LOT),
    }

    # --- Per-timeframe strategy parameters (from the Strategy Tester bridge) ---
    live_base_params = {}
    for tf in TIMEFRAMES:
        _bp = dict(ML_TARGET_PARAMS[tf]["parameters"])
        _bp["exit_model"] = ML_TARGET_PARAMS[tf]["exit_model"]
        live_base_params[tf] = _bp

    live_bundle = {
        "schema": "goldregimex_live_v1",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "random_state": int(RANDOM_STATE),
        "timeframes": list(TIMEFRAMES),
        "settings": live_settings,
        "models": {tf: pipeline[tf].model for tf in TIMEFRAMES},
        "thresholds": {tf: float(pipeline[tf].threshold) for tf in TIMEFRAMES},
        "base_params": live_base_params,
        "split_time": {tf: str(getattr(pipeline.get(tf), "split_time", None)) for tf in TIMEFRAMES},
    }

    with open(LIVE_MODEL_PKL, "wb") as _fh:
        pickle.dump(live_bundle, _fh, protocol=pickle.HIGHEST_PROTOCOL)

    _size_kb = LIVE_MODEL_PKL.stat().st_size / 1024.0
    print("Saved live model artifact:")
    print("   path : %s" % LIVE_MODEL_PKL)
    print("   size : %.1f KB" % _size_kb)
    print("   TFs  : %s" % ", ".join(TIMEFRAMES))
    for tf in TIMEFRAMES:
        print("   [%s] threshold=%.4f  lots=(A %.2f / B %.2f)  exit=%s" % (
            tf, live_bundle["thresholds"][tf], live_settings["position_a"],
            live_settings["position_b"], live_base_params[tf].get("exit_model", "fixed_tp")))
    return {"pipeline": pipeline, "coarse_results": coarse_results, "fine_results": fine_results}

def get_best_params(timeframe, broker=None):
    tf=str(timeframe).upper()
    pkl=_REPO_ROOT/"pipeline_verification_bundle"/"models"/"goldregimex_live_model.pkl"
    if not pkl.exists():
        raise FileNotFoundError("Live model bundle not found at %s. Run run_optimization() first." % pkl)
    with open(pkl,"rb") as fh:
        bundle=pickle.load(fh)
    thresholds=bundle.get("thresholds",{}); base=bundle.get("base_params",{})
    if tf not in thresholds:
        raise KeyError("Timeframe %s not in live bundle thresholds: %s" % (tf, list(thresholds)))
    params=dict(base.get(tf,{}) or {}); params["xgb_threshold"]=float(thresholds[tf])
    return params

def extract_consensus_params(timeframe, broker=None):
    return get_best_params(timeframe, broker)

def run_wfa(*a, **k):
    return run_optimization(n_jobs=k.get("n_jobs"))

if __name__ == "__main__":
    run_optimization()
