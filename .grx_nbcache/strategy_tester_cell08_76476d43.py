# Cell 6: Numba Backtest Engine with Early Eject, Pyramiding (Scale-In), and Asymmetric Guard

from numba import njit
import numpy as np
import pandas as pd
import math

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
    
    # RISK MANAGEMENT: true peak-to-trough drawdown on raw realized equity
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

@njit(cache=True)
def _run_backtest_numba(
    sig, high, low, close, spread, atr14, regime_code, time_minutes, 
    entry_atr_stop, entry_atr_target, leg_a_atr_target, exit_mode_code, 
    time_stop_minutes, trail_mult, initial_balance, pip_size_price, 
    pip_value_per_1lot, slippage_pips, spread_cap_points, commission_cents, is_m5
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

    eq = np.empty(max_trades + 1, dtype=np.float64)
    eq[0] = initial_balance
    eq_count = 1

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
                    tp_dist = entry_atr_target * atr_now

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
                    legs[0, 0], legs[0, 1], legs[0, 2], legs[0, 3], legs[0, 4], legs[0, 5], legs[0, 6] = 1.0, POSITION_A, side, entry_px, stop_px, leg_a_tp_px, i
                    legs[1, 0], legs[1, 1], legs[1, 2], legs[1, 3], legs[1, 4], legs[1, 5], legs[1, 6] = 1.0, POSITION_B, side, entry_px, stop_px, fixed_tp_px if enable_fixed_tp else np.nan, i

                elif can_scale_in:
                    leg_c_lot = POSITION_A if leg_a_closed_b_open else POSITION_B
                    
                    tighter_stop = 0.5 * atr_now
                    actual_stop_dist = tighter_stop * guard_factor
                    tighter_tp = 0.5 * atr_now

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

            # 1. Stop Loss Hit
            stop_hit = (bar_low <= leg_stop) if leg_side == 1 else (bar_high >= leg_stop)
            if stop_hit:
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
                            trade_count += 1
                            balance += p_cents
                            legs[l_id, 0] = 0.0
                    eq[eq_count] = balance
                    eq_count += 1
                    leg_a_profit_hit = 0
                    leg_b_profit_hit = 0
                    break

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
                trade_count += 1
                balance += p_cents
        eq[eq_count] = balance
        eq_count += 1

    return (
        trade_entry_idx, trade_exit_idx, trade_leg, trade_side, trade_entry_px,
        trade_exit_px, trade_move_pips, trade_pnl_cents, trade_reason,
        trade_entry_regime, trade_count, eq, eq_count, balance
    )

def _safe_float(value, default):
    return float(default) if value is None else float(value)

def run_backtest(timeframe: str, df: pd.DataFrame, signals: pd.Series, entry_params: dict, exit_model: str, exit_params: dict):
    exit_model_map = {
        "fixed_tp": 0, "mr_exit": 1, "fixed_tp_plus_mr": 2, 
        "partial_tp_plus_mr": 3, "partial_tp_mr_time_stop": 4
    }
    exit_mode_code = int(exit_model_map[exit_model])
    leg_a_atr_target = _safe_float(exit_params.get("leg_a_atr_target", -1.0), -1.0) if exit_params else -1.0
    time_stop_minutes = _safe_float(exit_params.get("time_stop_minutes", -1.0), -1.0) if exit_params else -1.0
    trail_mult = _safe_float(exit_params.get("trail_mult", 0.0), 0.0) if exit_params else 0.0

    idx = df.index
    time_minutes = (idx.view("int64") // 60000000000).astype(np.int64)

    sig = signals.reindex(idx).fillna(0).astype(np.int8).to_numpy()
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    spread = df["spread"].fillna(SPREAD_CAP_POINTS).to_numpy(dtype=np.float64)
    atr14 = df["atr14"].to_numpy(dtype=np.float64)
    regime_code = df["regime_code"].fillna(3).to_numpy(dtype=np.int32)
    
    is_m5 = 1 if timeframe == "M5" else 0

    (
        trade_entry_idx, trade_exit_idx, trade_leg, trade_side, trade_entry_px,
        trade_exit_px, trade_move_pips, trade_pnl_cents, trade_reason,
        trade_entry_regime, trade_count, eq, eq_count, ending_balance
    ) = _run_backtest_numba(
        sig, high, low, close, spread, atr14, regime_code, time_minutes,
        float(entry_params["atr_stop"]), float(entry_params["atr_target"]),
        leg_a_atr_target, exit_mode_code, time_stop_minutes, trail_mult,
        float(INITIAL_BALANCE_CENTS), float(PIP_SIZE_PRICE), float(PIP_VALUE_CENTS_PER_1LOT),
        float(SLIPPAGE_PIPS), float(SPREAD_CAP_POINTS), float(COMMISSION_CENTS_PER_TRADE),
        int(is_m5)
    )

    if trade_count == 0:
        return pd.DataFrame(), compute_metrics(pd.DataFrame(), [], INITIAL_BALANCE_CENTS, float(ending_balance))

    reasons = {1: "STOP", 2: "PROFIT_TARGET", 3: "FIXED_TP", 4: "MR_EXIT", 5: "TIME_STOP", 6: "EOD_CLOSE", 7: "REGIME_SHIFT"}
    regime_map = {1: "TREND", 2: "SHOCK", 3: "MR"}

    trades_df = pd.DataFrame(
        {
            "entry_time": idx[trade_entry_idx[:trade_count]],
            "exit_time": idx[trade_exit_idx[:trade_count]],
            "leg": np.where(trade_leg[:trade_count] == 0, "A", np.where(trade_leg[:trade_count] == 1, "B", "C_SCALE")),
            "side": np.where(trade_side[:trade_count] == 1, "BUY", "SELL"),
            "entry_price": trade_entry_px[:trade_count],
            "exit_price": trade_exit_px[:trade_count],
            "move_pips": trade_move_pips[:trade_count],
            "pnl_cents": trade_pnl_cents[:trade_count],
            "exit_reason": [reasons[int(x)] for x in trade_reason[:trade_count]],
            "entry_regime": [regime_map[int(x)] for x in trade_entry_regime[:trade_count]],
            "exit_model": exit_model,
        }
    )

    return trades_df, compute_metrics(trades_df, eq[:eq_count].tolist(), INITIAL_BALANCE_CENTS, float(ending_balance))