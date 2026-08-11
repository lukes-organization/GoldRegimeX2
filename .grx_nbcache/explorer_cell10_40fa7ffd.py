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