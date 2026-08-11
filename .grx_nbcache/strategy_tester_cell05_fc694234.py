# Data loading, strict 5-year reduction, indicators, and rule-based regimes

def resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    alt = Path.cwd().parent / path
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Path not found: {path}")

def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for key in ["open", "high", "low", "close", "volume", "spread"]:
        if key in cols:
            rename[cols[key]] = key
    out = df.rename(columns=rename)
    missing = [c for c in ["open", "high", "low", "close"] if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    return out

def read_xau_raw(path: Path) -> pd.DataFrame:
    path = resolve_path(path)
    df = pd.read_csv(path, sep=";")
    if "Date" not in df.columns:
        raise ValueError(f"Date column missing in {path}")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = _normalize_ohlc(df)
    if "spread" not in df.columns:
        df["spread"] = SPREAD_CAP_POINTS
    return df

def load_recent_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    last_date = df.index.max()
    start_date = last_date - pd.DateOffset(years=int(years))
    return df.loc[df.index >= start_date].copy()

def enforce_recent_window(df_full: pd.DataFrame, df_trim: pd.DataFrame, years: int, label: str):
    last_date = df_full.index.max()
    target_start = last_date - pd.DateOffset(years=int(years))
    observed_start = df_trim.index.min()
    if observed_start < target_start:
        raise RuntimeError(f"{label} window enforcement failed: observed_start={observed_start}, target_start={target_start}")
    print(f"{label} enforced window: {observed_start} -> {last_date}")

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

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out.index.hour
    london = (hour >= 7) & (hour < 16)
    ny = (hour >= 13) & (hour < 21)
    overlap = (hour >= 13) & (hour < 16)
    out["session"] = np.where(overlap, "OVERLAP", np.where(london, "LONDON", np.where(ny, "NEW_YORK", "ASIA")))
    out["session_mask_none"] = True
    out["session_mask_london"] = london
    out["session_mask_ny"] = ny
    out["session_mask_london_ny"] = london | ny
    return out

def build_features(exec_tf: str, m5_df: pd.DataFrame, m15_df: pd.DataFrame) -> pd.DataFrame:
    exec_tf = exec_tf.upper()
    if exec_tf not in ("M5", "M15"):
        raise ValueError(f"Unsupported timeframe: {exec_tf}")

    exec_df = m5_df.copy() if exec_tf == "M5" else m15_df.copy()
    trend_df = m15_df.copy()

    exec_df["rsi5"] = rsi(exec_df["close"], period=5)
    exec_df["atr14"] = atr(exec_df["high"], exec_df["low"], exec_df["close"], period=14)
    exec_df["atr100"] = atr(exec_df["high"], exec_df["low"], exec_df["close"], period=100)
    exec_df["atr_expansion"] = exec_df["atr14"] / exec_df["atr100"].replace(0.0, np.nan)

    for lb in sorted(set(BREAKOUT_LOOKBACK_GRID)):
        exec_df[f"roll_high_{lb}"] = exec_df["high"].rolling(lb, min_periods=lb).max().shift(1)
        exec_df[f"roll_low_{lb}"] = exec_df["low"].rolling(lb, min_periods=lb).min().shift(1)

    trend_df["m15_ema50"] = ema(trend_df["close"], period=50)
    trend_df["m15_ema200"] = ema(trend_df["close"], period=200)
    trend_df["m15_adx14"] = adx(trend_df["high"], trend_df["low"], trend_df["close"], period=14)

    if exec_tf == "M5":
        ex = exec_df.reset_index().rename(columns={exec_df.index.name or "index": "time"})
        tr = trend_df.reset_index().rename(columns={trend_df.index.name or "index": "time"})
        merged = pd.merge_asof(
            ex.sort_values("time"),
            tr[["time", "m15_ema50", "m15_ema200", "m15_adx14"]].sort_values("time"),
            on="time",
            direction="backward",
        ).set_index("time")
    else:
        merged = exec_df.copy()
        merged["m15_ema50"] = trend_df["m15_ema50"].reindex(merged.index)
        merged["m15_ema200"] = trend_df["m15_ema200"].reindex(merged.index)
        merged["m15_adx14"] = trend_df["m15_adx14"].reindex(merged.index)

    merged = add_session_features(merged)
    merged["spread"] = merged["spread"].fillna(SPREAD_CAP_POINTS)

        # Rule-based regime classification
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

m5_raw_full = read_xau_raw(M5_PATH)
m15_raw_full = read_xau_raw(M15_PATH)

m5_raw = load_recent_years(m5_raw_full, years=RESEARCH_YEARS)
m15_raw = load_recent_years(m15_raw_full, years=RESEARCH_YEARS)

enforce_recent_window(m5_raw_full, m5_raw, years=RESEARCH_YEARS, label="M5")
enforce_recent_window(m15_raw_full, m15_raw, years=RESEARCH_YEARS, label="M15")

FEATURES_BY_TF = {
    "M5": build_features("M5", m5_raw, m15_raw),
    "M15": build_features("M15", m5_raw, m15_raw),
}

print("M5 full rows:", len(m5_raw_full), "| recent rows:", len(m5_raw))
print("M15 full rows:", len(m15_raw_full), "| recent rows:", len(m15_raw))
for tf in TIMEFRAMES:
    print(tf, "feature rows:", len(FEATURES_BY_TF[tf]))
display(FEATURES_BY_TF["M5"].head(3))