# ---------------------------------------------------------
# Feature engineering + labels
# Phase 8/9: indicators, session features and feature engineering now come from
# the SINGLE shared implementation (shared.features / shared.session_filter).
# Only the TBM labelling (triple_barrier) stays local and UNCHANGED.
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from numba import njit

from shared.features import (
    ema, true_range, atr, rsi, adx, synth_vix_zscore, build_ml_features, feature_hash,
)
from shared.session_filter import SessionFilter

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