# Cell 5: Strategy Definitions & Regime-Based Signal Router (With Macro Filter)
class BaseStrategy:
    name = "base"
    param_grid = {}
    param_cols = []

    def iter_param_dicts(self):
        keys = list(self.param_cols)
        vals = [self.param_grid[k] for k in keys]
        for combo in itertools.product(*vals):
            yield dict(zip(keys, combo))

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        raise NotImplementedError

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

class TrendPullbackStrategy(BaseStrategy):
    name = "trend_pullback"
    param_grid = {
        "adx_threshold": ADX_GRID,
        "pullback_rsi": PULLBACK_RSI_GRID,
        "confirmation_bars": CONFIRMATION_GRID,
        "atr_stop": ATR_STOP_GRID,
        "atr_target": ATR_TARGET_GRID,
        "session_filter": SESSION_FILTER_VALUES,
    }
    param_cols = list(param_grid.keys())

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        adx_threshold = float(params["adx_threshold"])
        pullback_rsi = float(params["pullback_rsi"])
        confirmation_bars = int(params["confirmation_bars"])
        session_col = session_col_from_value(params["session_filter"])

        trend_up_raw = (df["m15_ema50"] > df["m15_ema200"]) & (df["m15_adx14"] > adx_threshold)
        trend_dn_raw = (df["m15_ema50"] < df["m15_ema200"]) & (df["m15_adx14"] > adx_threshold)

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

class VolatilityExpansionStrategy(BaseStrategy):
    name = "volatility_expansion"
    param_grid = {
        "atr_expansion_threshold": ATR_EXPANSION_GRID,
        "breakout_lookback": BREAKOUT_LOOKBACK_GRID,
        "breakout_buffer": BREAKOUT_BUFFER_GRID,
        "atr_stop": ATR_STOP_GRID,
        "atr_target": ATR_TARGET_GRID,
        "session_filter": SESSION_FILTER_VALUES,
    }
    param_cols = list(param_grid.keys())

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        thr = float(params["atr_expansion_threshold"])
        lb = int(params["breakout_lookback"])
        buf_mult = float(params["breakout_buffer"])
        session_col = session_col_from_value(params["session_filter"])

        high_col = f"roll_high_{lb}"
        low_col = f"roll_low_{lb}"
        breakout_buffer = buf_mult * df["atr14"]
        is_expansion = df["atr_expansion"] > thr

        # MACRO TREND FILTER: Ensures entries align with higher timeframe momentum
        macro_up = df["m15_ema50"] > df["m15_ema200"]
        macro_dn = df["m15_ema50"] < df["m15_ema200"]

        long_cond = is_expansion & (df["close"] > (df[high_col] + breakout_buffer)) & macro_up & df[session_col].astype(bool)
        short_cond = is_expansion & (df["close"] < (df[low_col] - breakout_buffer)) & macro_dn & df[session_col].astype(bool)

        sig = pd.Series(0, index=df.index, dtype=np.int8)
        sig.loc[long_cond.fillna(False)] = 1
        sig.loc[short_cond.fillna(False)] = -1
        return sig

STRATEGIES = {
    "trend_pullback": TrendPullbackStrategy(),
    "volatility_expansion": VolatilityExpansionStrategy(),
}

def generate_routed_signals(df: pd.DataFrame, params: dict, strategy_name: str) -> pd.Series:
    raw_signals = STRATEGIES[strategy_name].generate_signals(df, params)
    routed = pd.Series(0, index=df.index, dtype=np.int8)
    if strategy_name == "trend_pullback":
        mask = df["regime_code"] == 1
        routed.loc[mask] = raw_signals.loc[mask]
    elif strategy_name == "volatility_expansion":
        mask = df["regime_code"] == 2
        routed.loc[mask] = raw_signals.loc[mask]
    return routed