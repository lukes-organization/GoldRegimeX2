"""shared/features.py -- Single Feature Engineering module (Phase 8).

Faithful, centralised extraction of the indicator + feature logic that was
previously duplicated across Strategy_Tester.ipynb (build_features),
GoldRegimeX_Explorer.ipynb (build_features), src/processor.py and
src/validator.py. The maths below is UNCHANGED -- only unified so every
consumer imports the exact same definitions and can assert an identical
`feature_hash`.

The canonical entry point is `build_features(df, timeframe, trend_df=None)`:
  * timeframe == "M15": `df` is the M15 frame and also acts as its own trend
    context (m15_* columns computed on df).
  * timeframe == "M5":  pass the M15 frame as `trend_df` for the higher-TF
    merge_asof context, exactly as the notebooks did.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

SPREAD_CAP_POINTS_DEFAULT = 50.0
BREAKOUT_LOOKBACK_GRID_DEFAULT = (20, 40, 60)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False, min_periods=int(period)).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    atr_v = atr(high, low, close, period=period).replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_v
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_v
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().fillna(0.0)


def build_features(
    df: pd.DataFrame,
    timeframe: str,
    trend_df: Optional[pd.DataFrame] = None,
    breakout_lookback_grid=BREAKOUT_LOOKBACK_GRID_DEFAULT,
    spread_cap_points: float = SPREAD_CAP_POINTS_DEFAULT,
) -> pd.DataFrame:
    tf = str(timeframe).upper()
    if tf not in ("M5", "M15"):
        raise ValueError("Unsupported timeframe: %s" % timeframe)

    exec_df = df.copy()
    trend_src = (trend_df if trend_df is not None else df).copy()

    exec_df["rsi5"] = rsi(exec_df["close"], period=5)
    exec_df["atr14"] = atr(exec_df["high"], exec_df["low"], exec_df["close"], period=14)
    exec_df["atr100"] = atr(exec_df["high"], exec_df["low"], exec_df["close"], period=100)
    exec_df["atr_expansion"] = exec_df["atr14"] / exec_df["atr100"].replace(0.0, np.nan)

    for lb in sorted(set(breakout_lookback_grid)):
        exec_df["roll_high_%d" % lb] = exec_df["high"].rolling(lb, min_periods=lb).max().shift(1)
        exec_df["roll_low_%d" % lb] = exec_df["low"].rolling(lb, min_periods=lb).min().shift(1)

    trend_src["m15_ema50"] = ema(trend_src["close"], period=50)
    trend_src["m15_ema200"] = ema(trend_src["close"], period=200)
    trend_src["m15_adx14"] = adx(trend_src["high"], trend_src["low"], trend_src["close"], period=14)

    if tf == "M5":
        ex = exec_df.reset_index().rename(columns={exec_df.index.name or "index": "time"})
        tr = trend_src.reset_index().rename(columns={trend_src.index.name or "index": "time"})
        merged = pd.merge_asof(
            ex.sort_values("time"),
            tr[["time", "m15_ema50", "m15_ema200", "m15_adx14"]].sort_values("time"),
            on="time",
            direction="backward",
        ).set_index("time")
    else:
        merged = exec_df.copy()
        merged["m15_ema50"] = trend_src["m15_ema50"].reindex(merged.index)
        merged["m15_ema200"] = trend_src["m15_ema200"].reindex(merged.index)
        merged["m15_adx14"] = trend_src["m15_adx14"].reindex(merged.index)

    # Session feature columns come from the single SessionFilter (Phase 9).
    from shared.session_filter import SessionFilter
    merged = SessionFilter().add_session_features(merged)
    if "spread" not in merged.columns:
        merged["spread"] = spread_cap_points
    merged["spread"] = merged["spread"].fillna(spread_cap_points)

    is_trend = (merged["m15_adx14"] > 15.0) & (merged["atr_expansion"] < 1.3)
    is_shock = merged["atr_expansion"] >= 1.3
    merged["regime_str"] = np.where(is_shock, "SHOCK", np.where(is_trend, "TREND", "MR"))
    merged["regime_code"] = np.where(is_shock, 2, np.where(is_trend, 1, 3)).astype(np.int32)

    required = [
        "open", "high", "low", "close", "spread",
        "rsi5", "atr14", "atr100", "atr_expansion",
        "m15_ema50", "m15_ema200", "m15_adx14",
        "session", "regime_str", "regime_code",
        "session_mask_none", "session_mask_london", "session_mask_ny", "session_mask_london_ny",
    ]
    merged = merged.dropna(subset=[c for c in required if c in merged.columns]).copy()
    return merged


def feature_hash(df: pd.DataFrame, columns=None) -> str:
    """Deterministic hash of engineered feature values.

    Strategy Tester, Explorer and Validator must produce identical hashes for
    the same input data + timeframe (Phase 8 success criterion).
    """
    cols = sorted(columns) if columns is not None else sorted(
        c for c in df.columns if df[c].dtype.kind in "fiub"
    )
    sub = df[cols].copy()
    blob = pd.util.hash_pandas_object(sub.round(8), index=True).values.tobytes()
    head = ("|".join(cols)).encode()
    return hashlib.sha256(head + b"::" + blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Explorer ML feature set (Phase 8). This is the SECOND canonical feature
# builder -- the Explorer/Validator ML path genuinely computes a different
# feature set than the Strategy Tester grid-search path (external assets,
# log returns, synth-VIX, cyclical time, macro trend). Centralised here so the
# Explorer notebook and src/validator.py import ONE implementation instead of
# each maintaining a copy. TBM labelling stays with the caller (unchanged).
# ---------------------------------------------------------------------------
def synth_vix_zscore(high, low, close, period: int = 22):
    tr = true_range(high, low, close)
    roll_mean = tr.rolling(period, min_periods=period).mean()
    roll_std = tr.rolling(period, min_periods=period).std().replace(0, np.nan)
    return ((tr - roll_mean) / roll_std).fillna(0.0)


ML_FEATURE_KEEP = [
    "Open", "High", "Low", "Close",
    "log_return", "xag_log_return", "xti_log_return",
    "gold_silver_ratio_z", "gold_oil_ratio_z",
    "atr_20", "atr_normalized", "volatility_20",
    "synth_vix_zscore", "hour_sin", "hour_cos",
    "rsi5", "atr14", "atr100", "atr_expansion",
    "macro_ema50", "macro_ema200", "macro_adx14",
    "spread", "regime_str", "regime_code",
    "session_mask_none", "session_mask_london", "session_mask_ny", "session_mask_london_ny",
]


def build_ml_features(exec_panel: pd.DataFrame, timeframe: str, spread_cap_points: float = SPREAD_CAP_POINTS_DEFAULT) -> pd.DataFrame:
    """Faithful, centralised port of the Explorer notebook's feature block.

    Returns the engineered feature frame (no TBM labels). The caller applies
    its own (unchanged) triple_barrier afterwards, keeping TBM logic local.
    """
    tf = str(timeframe).upper()
    if tf not in ("M5", "M15"):
        raise ValueError("Unsupported timeframe: %s" % timeframe)
    exec_df = exec_panel.copy()

    exec_df["log_return"] = np.log(exec_df["Close"] / exec_df["Close"].shift(1))
    exec_df["xag_log_return"] = np.log(exec_df["XAG_Close"] / exec_df["XAG_Close"].shift(1))
    exec_df["xti_log_return"] = np.log(exec_df["XTI_Close"] / exec_df["XTI_Close"].shift(1))
    exec_df["gold_silver_ratio"] = exec_df["Close"] / exec_df["XAG_Close"]
    exec_df["gold_oil_ratio"] = exec_df["Close"] / exec_df["XTI_Close"]

    rw = 64
    gs_m = exec_df["gold_silver_ratio"].rolling(rw, min_periods=rw).mean()
    gs_s = exec_df["gold_silver_ratio"].rolling(rw, min_periods=rw).std().replace(0, np.nan)
    go_m = exec_df["gold_oil_ratio"].rolling(rw, min_periods=rw).mean()
    go_s = exec_df["gold_oil_ratio"].rolling(rw, min_periods=rw).std().replace(0, np.nan)
    exec_df["gold_silver_ratio_z"] = ((exec_df["gold_silver_ratio"] - gs_m) / gs_s).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    exec_df["gold_oil_ratio_z"] = ((exec_df["gold_oil_ratio"] - go_m) / go_s).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tr = true_range(exec_df["High"], exec_df["Low"], exec_df["Close"])
    exec_df["atr_20"] = tr.rolling(20, min_periods=20).mean()
    exec_df["atr_normalized"] = exec_df["atr_20"] / exec_df["Close"]
    exec_df["volatility_20"] = exec_df["log_return"].rolling(20, min_periods=20).std()
    exec_df["synth_vix_zscore"] = synth_vix_zscore(exec_df["High"], exec_df["Low"], exec_df["Close"], period=22)

    exec_df["hour"] = exec_df.index.hour
    exec_df["hour_sin"] = np.sin(2 * np.pi * exec_df["hour"] / 24.0)
    exec_df["hour_cos"] = np.cos(2 * np.pi * exec_df["hour"] / 24.0)

    exec_df["rsi5"] = rsi(exec_df["Close"], period=5)
    exec_df["atr14"] = atr(exec_df["High"], exec_df["Low"], exec_df["Close"], period=14)
    exec_df["atr100"] = atr(exec_df["High"], exec_df["Low"], exec_df["Close"], period=100)
    exec_df["atr_expansion"] = exec_df["atr14"] / exec_df["atr100"].replace(0.0, np.nan)

    exec_df["macro_ema50"] = ema(exec_df["Close"], period=50)
    exec_df["macro_ema200"] = ema(exec_df["Close"], period=200)
    exec_df["macro_adx14"] = adx(exec_df["High"], exec_df["Low"], exec_df["Close"], period=14)

    from shared.session_filter import SessionFilter
    merged = SessionFilter().add_session_features(exec_df)
    merged["spread"] = spread_cap_points

    is_trend = (merged["macro_adx14"] > 15.0) & (merged["atr_expansion"] < 1.3)
    is_shock = merged["atr_expansion"] >= 1.3
    merged["regime_str"] = np.where(is_shock, "SHOCK", np.where(is_trend, "TREND", "MR"))
    merged["regime_code"] = merged["regime_str"].map({"TREND": 1, "SHOCK": 2, "MR": 3})

    return merged[ML_FEATURE_KEEP].dropna().copy()
