import numpy as np
import pandas as pd
from src.logger import setup_logger
from src.risk_manager import BROKER_CONFIGS

logger = setup_logger(__name__)

ANNUALIZATION_FACTORS = {
    "H1":  np.sqrt(252 * 24),
    "M15": np.sqrt(252 * 96),    # 96 M15 bars per day
    "M5":  np.sqrt(252 * 288),   # 288 M5 bars per day
}
ANNUALIZATION_FACTOR = ANNUALIZATION_FACTORS["H1"]   # default

# Minimum ATR/spread ratio required before a signal is allowed through.
# M5: 4.5× — filters the worst noise bars (ATR < $1.80 on $2000 gold) while
#     keeping the bulk of mean-reversion opportunities.  The previous 7.0×
#     threshold matched avg_efficiency 4-6× seen in most trials, blocking
#     the majority of valid M5 signals and starving the Activity Bonus.
# H1/M15/default: 1.25× — standard threshold that blocks only the most
#     illiquid bars while leaving the majority of trend sessions open.
TF_MIN_EFFICIENCY = {"M5": 4.5}

RISK_PER_TRADE = 0.01

# ── Strict Probability Thresholds (mirrored from mt5_trader.py) ──────────────
TF_PROB_THRESHOLD  = {"M5": 0.55, "M15": 0.55, "H1": 0.55}
TF_SHORT_THRESHOLD = {"M5": 0.45, "M15": 0.45, "H1": 0.45}

BULL_STATE     = 0     # HMM Bull regime
BEAR_STATE     = 1     # HMM Bear regime
CHOP_STATE     = 2     # HMM Chop regime (no signals)

# ── Live Execution Configs (mirrored from mt5_trader.py) ─────────────────────
MAX_HOLD_BARS = 12

ATR_TRAIL_CONFIG: dict = {
    "H1":  {"activation_pnl": 1.50, "trail_mult": 2.5, "partial_close": True},
    "M15": {"activation_pnl": 1.50, "trail_mult": 1.5, "partial_close": True},
    "M5":  {"activation_pnl": 1.00, "trail_mult": 1.5, "partial_close": False,
            "scalp_target": 4.00},
}

TP_SL_CONFIG: dict = {
    "H1":  {
        "trend": {"tp1_mult": 1.5, "tp2_mult": 3.0,  "sl_mult": 2.0},
        "chop":  {"tp1_mult": 1.0, "tp2_mult": None,  "sl_mult": 1.4},
    },
    "M15": {
        "trend": {"tp1_mult": 1.2, "tp2_mult": 2.5,  "sl_mult": 2.0},
        "chop":  {"tp1_mult": 0.8, "tp2_mult": None,  "sl_mult": 1.4},
    },
    "M5":  {
        "trend": {"tp1_mult": 0.8, "tp2_mult": 1.5,  "sl_mult": 1.5},
        "chop":  {"tp1_mult": 0.5, "tp2_mult": None,  "sl_mult": 1.05},
    },
}


def format_payout(total_return: float, account_size: float, broker: str) -> str:
    """Format net payout in broker-appropriate currency units.

    Headway cent accounts show Cents (×100) alongside the USD equivalent so
    the $15 account balance is legible as 1,500 Cents.
    Standard accounts show USD only.
    """
    net_usd = total_return * account_size
    if broker == "headway_cent":
        return f"{net_usd * 100:.2f} Cents ({net_usd:.4f} USD)"
    return f"${net_usd:.4f} USD"


def _compute_audit_metrics(
    signals: np.ndarray,
    atr_norm: np.ndarray,
    spread_frac: float,
    gross_returns: np.ndarray,
    costs: np.ndarray,
    total_return: float,
    account_size: float,
) -> dict:
    """Broker-cost and efficiency audit metrics for a given array slice.

    avg_efficiency  — mean ATR/spread on active-trade bars; >1.25 means the
                      model is trading when moves cover the bid/ask.
    cost_efficiency — fraction of gross profit retained after broker costs;
                      <0.50 signals the broker is taking >50% of the edge.
    total_net_payout — absolute dollar profit (total_return × account_size).
    """
    active      = signals != 0
    avg_eff     = float(np.mean(atr_norm[active] / spread_frac)) if active.any() else 0.0
    gross_pos   = float(np.sum(gross_returns[gross_returns > 0]))
    total_costs = float(np.sum(np.abs(costs)))
    cost_eff    = 1.0 - (total_costs / gross_pos) if gross_pos > 0 else 0.0
    return {
        "avg_efficiency":   avg_eff,
        "cost_efficiency":  cost_eff,
        "total_net_payout": total_return * account_size,
    }


def compute_position_sizes(
    signals,
    atr_normalized,
    atr_multiplier=2.0,
    risk_per_trade=RISK_PER_TRADE,
    pos_per_trade: int = 1,
):
    """Size positions using 1% risk / (2×ATR) with optional dual-position multiplier.

    ``signals`` may be -1 (SELL), 0 (hold), or 1 (BUY).  The sign is
    preserved so that short positions produce negative sizes, which when
    multiplied by positive next_returns yield negative (loss) gross returns
    — correctly reflecting a losing short trade when price rises.

    Args:
        pos_per_trade: 1 for small accounts (single), 2 for growth accounts (dual/hedging).
    """
    stop_distance = atr_normalized * atr_multiplier
    stop_distance = np.where(stop_distance > 0, stop_distance, 1e-8)
    sizes = (risk_per_trade / stop_distance) * signals * pos_per_trade
    return sizes


def compute_signals(
    df: "pd.DataFrame",
    probabilities: np.ndarray,
    hmm_states: np.ndarray,
    tf: str = "H1",
    broker: str = "standard",
    account_size: float = 15.0,
    hmm_transmat=None,
    **_kwargs,  # absorb legacy zscore/regime_stats/threshold arguments
) -> np.ndarray:
    """Return the +1/0/-1 signal array from the SignalEngine bar loop.

    Thin public wrapper around ``_run_bar_loop`` for use by visualizer and
    audit tools.  All deprecated Z-score / regime-stats arguments are silently
    ignored via ``**_kwargs``.
    """
    closes = df["Close"].values
    highs  = df["High"].values if "High" in df.columns else closes
    lows   = df["Low"].values  if "Low"  in df.columns else closes
    _tr = np.maximum(highs - lows,
          np.maximum(np.abs(highs - np.roll(closes, 1)),
                     np.abs(lows  - np.roll(closes, 1))))
    raw_atr_arr = pd.Series(_tr).rolling(14, min_periods=1).mean().bfill().values
    signals, *_ = _run_bar_loop(
        df, probabilities, hmm_states, hmm_transmat,
        tf=tf, broker=broker, account_size=account_size,
        raw_atr_arr=raw_atr_arr,
    )
    return signals


def compute_trade_costs(signals: np.ndarray, broker: str = "standard") -> np.ndarray:
    """Apply spread + commission only at position transitions (entry / exit / reversal).

    A transition is any bar where the signal changes from the previous bar:
      0 → ±1   = entry (1× cost)
      ±1 → 0   = exit  (1× cost)
      +1 → −1  = simultaneous exit + entry (2× cost)

    Holding bars (same non-zero signal as previous) incur no additional cost,
    matching live MT5 behaviour where a held position has no per-bar charge.
    """
    config = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["standard"])
    cost_per_trade = config["spread_frac"] + config["commission_frac"]
    prev = np.concatenate([[0], signals[:-1]])
    changed = signals != prev                              # any transition
    reversal = changed & (signals != 0) & (prev != 0)     # direction flip = 2 costs
    cost_mult = changed.astype(float) + reversal.astype(float)   # 1 or 2
    return cost_mult * cost_per_trade


def _compute_metrics(strategy_returns, signals, tf: str = "H1"):
    ann_factor = ANNUALIZATION_FACTORS.get(tf.upper(), ANNUALIZATION_FACTOR)
    mean_ret = np.mean(strategy_returns)
    std_ret  = np.std(strategy_returns)
    sharpe   = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0.0

    cumulative  = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns   = running_max - cumulative
    max_dd      = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Count entries (signal transitions to non-zero) — matches live behaviour where
    # a held position is ONE trade until direction changes or the bar exits.
    prev_s   = np.concatenate([[0], signals[:-1]])
    is_entry = (signals != 0) & (signals != prev_s)
    n_trades = int(np.sum(is_entry))

    # Win rate computed per trade (entry-to-exit P&L), not per active bar.
    # Assign each position bar to its originating trade via cumsum of entries,
    # then sum strategy_returns within each trade to determine wins vs losses.
    if n_trades > 0:
        trade_idx = np.cumsum(is_entry)  # 0 before first entry, then 1, 2, …
        in_trade  = signals != 0
        per_trade = np.bincount(trade_idx[in_trade],
                                weights=strategy_returns[in_trade],
                                minlength=n_trades + 1)[1:]  # drop index-0 bucket
        win_rate   = float(np.sum(per_trade > 0) / n_trades)
        wins_sum   = float(np.sum(per_trade[per_trade > 0]))
        losses_sum = float(abs(np.sum(per_trade[per_trade < 0])))
        profit_factor   = min(wins_sum / losses_sum, 10.0) if losses_sum > 0 else (10.0 if wins_sum > 0 else 1.0)
        expected_payoff = float(np.sum(per_trade) / n_trades)
    else:
        win_rate        = 0.0
        profit_factor   = 1.0
        expected_payoff = 0.0

    total_return = float(np.exp(np.sum(strategy_returns)) - 1)

    if max_dd > 0:
        recovery_factor = float(min(total_return / max_dd, 20.0))
    elif total_return > 0:
        recovery_factor = 20.0
    else:
        recovery_factor = 0.0

    # Return Consistency — weekly P&L stability.
    # Bucket strategy_returns into approximate trading-week windows and compute
    # how stable the weekly income is.  A model that earns steadily every week
    # scores higher than one that earns the same total from 2 lucky streaks.
    #
    # Formula: consistency = 1 - (std / (std + |mean|))
    #   → 1.0 when std ≈ 0 (all weeks identical)  → 0.0 when std >> mean (erratic)
    # Only computed when ≥4 complete windows are available; 0.0 otherwise.
    WEEKLY_BARS = {"H1": 40, "M15": 160, "M5": 480}   # ~5 trading days per window
    _bpw = WEEKLY_BARS.get(tf.upper(), 40)
    if n_trades >= 5 and len(strategy_returns) >= _bpw * 4:
        _n       = (len(strategy_returns) // _bpw) * _bpw
        _weekly  = strategy_returns[:_n].reshape(-1, _bpw).sum(axis=1)
        _w_std   = float(np.std(_weekly))
        _w_mean  = float(np.mean(_weekly))
        _denom   = _w_std + abs(_w_mean)
        return_consistency = float(1.0 - _w_std / _denom) if _denom > 1e-9 else 0.0
    else:
        return_consistency = 0.0

    return {
        "sharpe_ratio":       float(sharpe),
        "max_drawdown":       max_dd,
        "win_rate":           win_rate,
        "total_return":       total_return,
        "n_trades":           n_trades,
        "recovery_factor":    recovery_factor,
        "profit_factor":      profit_factor,
        "expected_payoff":    expected_payoff,
        "return_consistency": return_consistency,
    }


def _compute_floating_drawdown(
    df: pd.DataFrame,
    signals: np.ndarray,
    sizes: np.ndarray,
    strategy_returns: np.ndarray,
) -> float:
    """Peak-to-trough drawdown on the intra-bar floating equity curve.

    For each bar with an open position the worst-case adverse price move from
    the trade entry is computed:

        BUY  bar: adverse_frac = max(0, (entry_price - Low[i])  / entry_price)
        SELL bar: adverse_frac = max(0, (High[i] - entry_price) / entry_price)

    The floating equity at bar i is:
        closed_equity[i]  -  |sizes[i]| × adverse_frac[i]

    The reported drawdown is the largest peak (closed equity) to trough
    (floating equity) drop, capturing trades that dip deep before recovering.

    Falls back to closed-bar drawdown when High/Low columns are absent.
    """
    if not {"High", "Low", "Close"}.issubset(df.columns) or len(signals) == 0:
        cumulative  = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        return float(np.max(running_max - cumulative)) if len(cumulative) > 0 else 0.0

    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    n     = len(signals)

    # ── Track entry price for each in-trade bar ───────────────────────────────
    # When a trade opens (signal becomes non-zero or reverses), record Close[i]
    # as the entry price and carry it forward until the position closes.
    entry_prices  = np.zeros(n)
    current_entry = 0.0
    prev_sig      = 0
    for i in range(n):
        sig = signals[i]
        if sig != 0 and (prev_sig == 0 or sig != prev_sig):
            current_entry = close[i]   # entry at this bar's close
        elif sig == 0:
            current_entry = 0.0
        entry_prices[i] = current_entry if sig != 0 else 0.0
        prev_sig = sig

    # ── Adverse excursion fraction ─────────────────────────────────────────────
    adverse_frac = np.zeros(n)
    buy_on       = (signals ==  1) & (entry_prices > 0)
    sell_on      = (signals == -1) & (entry_prices > 0)
    if buy_on.any():
        ep = entry_prices[buy_on]
        adverse_frac[buy_on]  = np.maximum(0.0, (ep - low[buy_on])  / ep)
    if sell_on.any():
        ep = entry_prices[sell_on]
        adverse_frac[sell_on] = np.maximum(0.0, (high[sell_on] - ep) / ep)

    # ── Floating equity series ─────────────────────────────────────────────────
    # Closed equity: cumulative realised P&L at each bar close.
    # Floating equity: worst-case equity during each bar (can dip below closed).
    closed_equity   = np.cumsum(strategy_returns)
    adverse_equity  = np.abs(sizes) * adverse_frac
    floating_equity = closed_equity - adverse_equity

    # Running max anchored to bar closes; trough is min of floating vs closed.
    running_max = np.maximum.accumulate(closed_equity)
    trough      = np.minimum(floating_equity, closed_equity)
    drawdowns   = running_max - trough
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def _compute_balance_curve(
    equity_arr: np.ndarray,
    signals: np.ndarray,
    account_size: float,
) -> np.ndarray:
    """Step-function balance: constant while a trade is open, steps at close.

    MT5 Strategy Tester style — Balance only moves when a position is fully
    closed.  Equity is continuous (includes floating P&L between entries).
    """
    in_trade    = signals != 0
    balance_arr = equity_arr.copy().astype(float)
    balance_arr[in_trade] = np.nan            # mask in-trade bars
    if in_trade[0]:
        balance_arr[0] = account_size         # seed before first trade
    # Forward-fill: hold the last closed balance across in-trade bars
    prev = np.nan
    for i in range(len(balance_arr)):
        if np.isnan(balance_arr[i]):
            balance_arr[i] = prev
        else:
            prev = balance_arr[i]
    return balance_arr


def _mr_attribution(
    signals: np.ndarray,
    hmm_states: np.ndarray,
    strategy_returns: np.ndarray,
) -> dict:
    """Compute MR trade statistics separately from trend trades.

    Classifies each trade entry as 'trend' (HMM state < 2) or 'mr' (state >= 2),
    then computes count, win rate, and total log-return for MR trades.

    Returns dict with keys: mr_trades, mr_win_rate, mr_pnl.
    """
    prev_s   = np.concatenate([[0], signals[:-1]])
    is_entry = (signals != 0) & (signals != prev_s)
    if not is_entry.any():
        return {"mr_trades": 0, "mr_win_rate": 0.0, "mr_pnl": 0.0}

    trade_idx    = np.cumsum(is_entry)
    entry_is_mr  = {int(trade_idx[i]): bool(hmm_states[i] >= CHOP_STATE)
                    for i in np.where(is_entry)[0]}

    in_trade  = signals != 0
    mr_pnls: list[float] = []
    for tid, is_mr in entry_is_mr.items():
        if is_mr:
            mask = (trade_idx == tid) & in_trade
            if mask.any():
                mr_pnls.append(float(strategy_returns[mask].sum()))

    n = len(mr_pnls)
    return {
        "mr_trades":   n,
        "mr_win_rate": float(sum(1 for r in mr_pnls if r > 0) / n) if n else 0.0,
        "mr_pnl":      float(sum(mr_pnls)),
    }


def _compute_bb_positions(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Normalised position within rolling high-low band (0=at low, 1=at high)."""
    out = np.full(len(close), 0.5)
    for i in range(window, len(close)):
        w = close[i - window : i + 1]
        lo, hi = w.min(), w.max()
        out[i] = (close[i] - lo) / (hi - lo + 1e-9)
    return out


def _run_bar_loop(
    df_aligned: pd.DataFrame,
    probabilities: np.ndarray,
    states: np.ndarray,
    hmm_transmat,
    tf: str,
    broker: str,
    account_size: float,
    split_idx=None,
    return_trades: bool = False,
    test_mask=None,
    raw_atr_arr=None,
) -> tuple:
    from src.signal_engine import SignalEngine

    n = len(df_aligned)
    engine = SignalEngine(tf=tf)

    signals          = np.zeros(n, dtype=np.int8)
    strategy_returns = np.zeros(n, dtype=np.float64)
    gross_returns    = np.zeros(n, dtype=np.float64)
    costs_arr        = np.zeros(n, dtype=np.float64)
    sizes_arr        = np.zeros(n, dtype=np.float64)
    trades           = [] if return_trades else None

    # ── Core price arrays ─────────────────────────────────────────────────────
    closes      = df_aligned["Close"].values
    highs       = df_aligned["High"].values if "High" in df_aligned.columns else closes
    lows        = df_aligned["Low"].values  if "Low"  in df_aligned.columns else closes
    log_returns = df_aligned["log_return"].values

    # ── Raw ATR fallback (if not pre-computed by caller) ─────────────────────
    if raw_atr_arr is None:
        _tr = np.maximum(highs - lows,
               np.maximum(np.abs(highs - np.roll(closes, 1)),
                          np.abs(lows  - np.roll(closes, 1))))
        raw_atr_arr = pd.Series(_tr).rolling(14, min_periods=1).mean().bfill().values

    _broker_cfg = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["standard"])
    spread_cost = _broker_cfg.get("spread_frac", 0.0004) + _broker_cfg.get("commission_frac", 0.0)

    # ── Per-TF configs ────────────────────────────────────────────────────────
    _tf_up       = tf.upper()
    prob_thresh  = TF_PROB_THRESHOLD.get(_tf_up, 0.55)
    short_thresh = TF_SHORT_THRESHOLD.get(_tf_up, 0.45)
    _trail_cfg   = ATR_TRAIL_CONFIG.get(_tf_up, ATR_TRAIL_CONFIG["H1"])
    _tp_sl_cfg   = TP_SL_CONFIG.get(_tf_up, TP_SL_CONFIG["H1"])

    # ── Trade state ───────────────────────────────────────────────────────────
    in_trade      = False
    direction     = 0
    size_mult     = 1.0
    cum_return    = 0.0
    max_fav_pnl   = 0.0
    bars_in_trade = 0
    trade_entry_i = 0
    cooldown_bars = 0

    # ── Live-mirror execution tracker ─────────────────────────────────────────
    entry_price   = 0.0
    entry_atr     = 0.0
    current_sl    = 0.0
    tp1_level     = 0.0
    signal_type   = "trend"
    guard_hit     = False   # 70 % profit guard fired → SL moved to entry+buffer
    atr_activated = False   # ATR trail arm tripped
    partial_done  = False   # half-close already executed

    COOLDOWN_PERIOD = 3     # bars locked after any trade exit (all TFs)

    for i in range(n):
        regime_info  = engine.update_regime(int(states[i]), hmm_transmat)
        _in_test     = test_mask is None or bool(test_mask[i])
        current_prob = float(probabilities[i])

        # Cooldown always ticks down, even while in a trade (ensures the 3-bar
        # gap is measured from the *close* of the exiting trade, not re-entry).
        if cooldown_bars > 0:
            cooldown_bars -= 1

        if in_trade:
            bars_in_trade += 1
            bar_high  = float(highs[i])
            bar_low   = float(lows[i])
            bar_close = float(closes[i])

            # ── 1. Intra-bar SL / TP hit check (Pessimistic Execution) ────────
            # Pessimistic: if both would be hit, assume SL hit first (worst case).
            # When a level is breached, record the exact SL/TP price so bar_gross
            # reflects the actual fill rather than the close-to-close return.
            hit_sl = hit_tp1 = False
            exit_price = bar_close  # default — overridden when a level is breached

            if direction == 1:
                if bar_low <= current_sl:
                    hit_sl     = True
                    exit_price = current_sl
                elif bar_high >= tp1_level:
                    hit_tp1    = True
                    exit_price = tp1_level
            else:
                if bar_high >= current_sl:
                    hit_sl     = True
                    exit_price = current_sl
                elif bar_low <= tp1_level:
                    hit_tp1    = True
                    exit_price = tp1_level

            # ── 2. Precise PnL accounting ─────────────────────────────────────
            # When a SL/TP is hit use log(exit_price / prev_close) so the return
            # segment stops at the actual fill price, not the bar close.
            prev_close = float(closes[i - 1]) if i > 0 else float(closes[i])
            if (hit_sl or hit_tp1) and prev_close > 0:
                actual_return = np.log(exit_price / prev_close)
                bar_gross = direction * actual_return * size_mult
            else:
                bar_gross = direction * log_returns[i] * size_mult

            gross_returns[i] = bar_gross
            # Accumulate the raw directional log-return (un-sized) so cum_return
            # tracks the true price move from entry regardless of partial closes.
            cum_return += (bar_gross / size_mult) if size_mult > 0 else 0.0
            cur_pnl      = cum_return * account_size
            max_fav_pnl  = max(max_fav_pnl, cur_pnl)
            signals[i]   = direction
            sizes_arr[i] = size_mult

            # ── 3. 70 % Profit Guard ─────────────────────────────────────────
            # Once price has covered 70 % of the TP1 distance, move SL to
            # entry + 2× spread so the trade can't turn into a loss.
            # Skip if an intra-bar exit already fired this bar.
            if not guard_hit and not hit_tp1 and not hit_sl:
                guard_dist = abs(tp1_level - entry_price) * 0.70
                if direction == 1 and bar_high >= entry_price + guard_dist:
                    current_sl = entry_price + spread_cost * 2
                    guard_hit  = True
                elif direction == -1 and bar_low <= entry_price - guard_dist:
                    current_sl = entry_price - spread_cost * 2
                    guard_hit  = True

            # ── 4. ATR Trail Activation & Partial Close ───────────────────────
            if not atr_activated and cur_pnl >= _trail_cfg["activation_pnl"]:
                atr_activated = True
                if _trail_cfg.get("partial_close") and not partial_done:
                    size_mult    *= 0.50
                    partial_done  = True
                    costs_arr[i] += spread_cost * size_mult   # cost of halving

            # ── 5. Ratchet Trail SL ───────────────────────────────────────────
            if atr_activated:
                trail_dist = _trail_cfg["trail_mult"] * entry_atr
                if direction == 1:
                    new_sl = bar_close - trail_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
                else:
                    new_sl = bar_close + trail_dist
                    if new_sl < current_sl:
                        current_sl = new_sl

            # ── 6. End-of-bar exit checks ─────────────────────────────────────
            hit_time_stop = bars_in_trade >= MAX_HOLD_BARS

            _hmm_now = int(states[i])
            regime_exit = (
                (signal_type == "trend"          and _hmm_now >= CHOP_STATE) or
                (signal_type == "mean_reversion" and _hmm_now <  CHOP_STATE)
            )

            engine_exit, _exit_reason = engine.should_exit(
                regime_info, cur_pnl, max_fav_pnl, bars_in_trade
            )

            _scalp_hit = (
                _tf_up == "M5"
                and not _trail_cfg.get("partial_close")
                and cur_pnl >= _trail_cfg.get("scalp_target", 4.00)
            )

            _force_exit = (
                test_mask is not None
                and (i + 1 >= n or not bool(test_mask[i + 1]))
            )

            exit_now = hit_sl or hit_tp1 or hit_time_stop or regime_exit or engine_exit or _scalp_hit or _force_exit

            if exit_now:
                costs_arr[i]        += spread_cost * size_mult
                strategy_returns[i]  = bar_gross - spread_cost * size_mult

                if return_trades:
                    _is_end = split_idx if split_idx is not None else n
                    if trade_entry_i < _is_end:
                        trades.append({
                            "bar_idx":     trade_entry_i,
                            "regime":      int(states[trade_entry_i]),
                            "signal":      direction,
                            "prob":        float(probabilities[trade_entry_i]),
                            "pnl":         cur_pnl,
                            "exit_reason": "SL"     if hit_sl        else
                                           "TP"     if hit_tp1        else
                                           "TIME"   if hit_time_stop  else
                                           "REGIME" if regime_exit    else
                                           "SCALP"  if _scalp_hit     else
                                           _exit_reason if engine_exit else "FORCE",
                        })

                in_trade      = False
                guard_hit     = False
                atr_activated = False
                partial_done  = False
                cooldown_bars = COOLDOWN_PERIOD
                engine.on_trade_closed()
                cum_return, max_fav_pnl, bars_in_trade = 0.0, 0.0, 0
            else:
                strategy_returns[i] = bar_gross

        else:
            if cooldown_bars == 0 and _in_test:
                valid_long  = current_prob >= prob_thresh
                valid_short = current_prob <= short_thresh

                if valid_long or valid_short:
                    # ── Spread Efficiency Guard ────────────────────────────────
                    # Reject the signal if the bar's raw ATR cannot cover the
                    # broker spread by the required multiple.  Mirrors the live
                    # TF_MIN_EFFICIENCY check and prevents trading illiquid bars.
                    _bar_atr = max(float(raw_atr_arr[i]), 1e-6)
                    _min_eff = TF_MIN_EFFICIENCY.get(_tf_up, 1.25)
                    if (_bar_atr / spread_cost) < _min_eff:
                        continue

                    entry = engine.should_enter(regime_info, current_prob, 1, 0.5)

                    if entry:
                        _dir = 1 if entry["signal"] in ("BUY", "MR_BUY") else -1
                        if (_dir == 1 and valid_long) or (_dir == -1 and valid_short):
                            direction     = _dir
                            size_mult     = entry["size_multiplier"]
                            in_trade      = True
                            signals[i]    = direction
                            sizes_arr[i]  = size_mult
                            trade_entry_i = i
                            signal_type   = "mean_reversion" if "MR" in entry["signal"] else "trend"

                            # ── Raw ATR-based SL/TP at entry ─────────────────
                            entry_price = float(closes[i])
                            entry_atr   = max(float(raw_atr_arr[i]), 1e-6)

                            _regime_key = "chop" if int(states[i]) >= CHOP_STATE else "trend"
                            _sl_tp      = _tp_sl_cfg.get(_regime_key, _tp_sl_cfg["trend"])
                            sl_dist     = _sl_tp["sl_mult"]  * entry_atr
                            tp1_dist    = _sl_tp["tp1_mult"] * entry_atr

                            if direction == 1:
                                current_sl = entry_price - sl_dist
                                tp1_level  = entry_price + tp1_dist
                            else:
                                current_sl = entry_price + sl_dist
                                tp1_level  = entry_price - tp1_dist

                            engine.on_trade_entered(int(states[i]))
                            costs_arr[i]        = spread_cost * size_mult
                            strategy_returns[i] = -spread_cost * size_mult

    return signals, strategy_returns, gross_returns, costs_arr, sizes_arr, trades


def vectorized_backtest(
    df,
    probabilities,
    hmm_states,
    split_idx=None,
    account_size: float = 15.0,
    broker: str = "standard",
    tf: str = "H1",
    prob_threshold: float = None,   # noqa: ARG001 — kept for backward compat
    short_threshold: float = None,  # noqa: ARG001 — kept for backward compat
    regime_stats: dict = None,      # noqa: ARG001 — kept for backward compat
    evaluator_config: dict = None,  # noqa: ARG001 — kept for backward compat
    use_tiered: bool = False,       # noqa: ARG001 — kept for backward compat
    return_trades: bool = False,
    hmm_transmat=None,
    test_mask=None,
):
    """Bar-by-bar backtest using SignalEngine for entry/exit decisions.

    Args:
        df: Processed DataFrame with ``log_return`` and ``atr_normalized``.
        probabilities: XGB probability array aligned with df.
        hmm_states: HMM state array aligned with df.
        split_idx: Bar index for IS/OOS split (None = full-period only).
        account_size: Account balance in USD.
        broker: Broker profile key from ``BROKER_CONFIGS``.
        tf: Timeframe string used for correct annualization ("H1", "M15", "M5").
        prob_threshold/short_threshold/regime_stats/evaluator_config/use_tiered:
            Retained for backward compatibility — ignored by the SignalEngine path.
        return_trades: When True, add ``'trades_df'`` to the result.
        hmm_transmat: HMM transition matrix — passed to SignalEngine for
                      persistence-collapse exit detection.
    """
    _ = (prob_threshold, short_threshold, regime_stats, evaluator_config, use_tiered)
    atr_norm    = df["atr_normalized"].values

    # ── RAW ATR CALCULATION (CRITICAL FIX) ───────────────────────────────────
    # Using standard-scaled atr_normalized for SL/TP distances causes negative
    # distances when volatility is below mean, instantly inverting stops.
    # Always recompute from True Range using raw price columns.
    closes = df["Close"].values
    highs  = df["High"].values if "High" in df.columns else closes
    lows   = df["Low"].values  if "Low"  in df.columns else closes
    _tr    = np.maximum(highs - lows,
             np.maximum(np.abs(highs - np.roll(closes, 1)),
                        np.abs(lows  - np.roll(closes, 1))))
    raw_atr_arr = pd.Series(_tr).rolling(14, min_periods=1).mean().bfill().values

    # ── Signal generation via bar-by-bar SignalEngine ─────────────────────────
    signals, strategy_returns, gross_returns, costs, sizes, _trades = _run_bar_loop(
        df, probabilities, hmm_states, hmm_transmat,
        tf=tf, broker=broker, account_size=account_size,
        split_idx=split_idx, return_trades=return_trades,
        test_mask=test_mask, raw_atr_arr=raw_atr_arr,
    )

    _spread_frac = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["standard"]).get("spread_frac", 0.0004)

    result = _compute_metrics(strategy_returns, signals, tf=tf)
    result["floating_max_drawdown"] = _compute_floating_drawdown(
        df, signals, sizes, strategy_returns
    )
    result.update(_compute_audit_metrics(
        signals, atr_norm, _spread_frac, gross_returns, costs,
        result["total_return"], account_size,
    ))
    result.update({
        "account_size":  account_size,
        "broker":        broker,
        "tf":            tf,
        "pos_per_trade": 1,
    })
    # Full-period MR attribution
    result.update(_mr_attribution(signals, hmm_states, strategy_returns))

    if split_idx is not None and 0 < split_idx < len(strategy_returns) and test_mask is None:
        is_m  = _compute_metrics(strategy_returns[:split_idx], signals[:split_idx], tf)
        oos_m = _compute_metrics(strategy_returns[split_idx:], signals[split_idx:], tf)
        # Floating drawdown per IS/OOS slice — df index must be sliced to match
        is_m["floating_max_drawdown"]  = _compute_floating_drawdown(
            df.iloc[:split_idx], signals[:split_idx], sizes[:split_idx], strategy_returns[:split_idx],
        )
        oos_m["floating_max_drawdown"] = _compute_floating_drawdown(
            df.iloc[split_idx:], signals[split_idx:], sizes[split_idx:], strategy_returns[split_idx:],
        )
        # Audit metrics per slice
        is_m.update(_compute_audit_metrics(
            signals[:split_idx], atr_norm[:split_idx], _spread_frac,
            gross_returns[:split_idx], costs[:split_idx],
            is_m["total_return"], account_size,
        ))
        oos_m.update(_compute_audit_metrics(
            signals[split_idx:], atr_norm[split_idx:], _spread_frac,
            gross_returns[split_idx:], costs[split_idx:],
            oos_m["total_return"], account_size,
        ))
        result["split_idx"] = split_idx
        for k, v in is_m.items():
            result[f"is_{k}"] = v
        for k, v in oos_m.items():
            result[f"oos_{k}"] = v
        # MR attribution per slice
        oos_attr = _mr_attribution(
            signals[split_idx:], hmm_states[split_idx:], strategy_returns[split_idx:]
        )
        for k, v in oos_attr.items():
            result[f"oos_{k}"] = v
        logger.info(
            "Backtest IS  [%s]: Sharpe=%.3f | FloatDD=%.4f | WR=%.3f | Trades=%d | Eff=%.2fx | CostEff=%.1f%%",
            tf, is_m["sharpe_ratio"], is_m["floating_max_drawdown"], is_m["win_rate"],
            is_m["n_trades"], is_m["avg_efficiency"], is_m["cost_efficiency"] * 100,
        )
        logger.info(
            "Backtest OOS [%s]: Sharpe=%.3f | FloatDD=%.4f | WR=%.3f | Trades=%d | Eff=%.2fx | CostEff=%.1f%%",
            tf, oos_m["sharpe_ratio"], oos_m["floating_max_drawdown"], oos_m["win_rate"],
            oos_m["n_trades"], oos_m["avg_efficiency"], oos_m["cost_efficiency"] * 100,
        )
    else:
        logger.info(
            "Backtest [%s]: Sharpe=%.3f | MaxDD=%.3f | WR=%.3f | Return=%.1f%% | Trades=%d",
            tf, result["sharpe_ratio"], result["max_drawdown"], result["win_rate"],
            result["total_return"] * 100, result["n_trades"],
        )

    logger.info(
        "%s: account=$%.2f | costs=%.4f/trade",
        broker,
        account_size,
        BROKER_CONFIGS.get(broker, {}).get("spread_frac", 0)
        + BROKER_CONFIGS.get(broker, {}).get("commission_frac", 0),
    )

    # ── Equity / Balance series for MT5-style chart ───────────────────────────
    _cumulative     = np.cumsum(strategy_returns)
    _equity_arr     = account_size * np.exp(_cumulative)
    _balance_arr    = _compute_balance_curve(_equity_arr, signals, account_size)
    _deposit_load   = (signals != 0).astype(np.float32)   # 1 = in-trade bar
    result["equity_timestamps"] = np.array(df.index)
    result["equity_values"]     = _equity_arr
    result["balance_values"]    = _balance_arr
    result["deposit_load"]      = _deposit_load

    # ── Per-trade records (collected by _run_bar_loop when return_trades=True) ──
    if return_trades and _trades is not None:
        _empty_cols = ["bar_idx", "regime", "signal", "prob", "volatility",
                       "spread", "gmm_vol_cluster", "pnl"]
        result["trades_df"] = (
            pd.DataFrame(_trades) if _trades else pd.DataFrame(columns=_empty_cols)
        )

    return result


def run_walk_forward(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    states: np.ndarray,
    train_days: int        = 365,
    test_days: int         = 90,
    account_size: float    = 15.0,
    broker: str            = "headway_cent",
    tf: str                = "H1",
    prob_threshold: float  = None,
    short_threshold: float = None,
    regime_stats: dict     = None,
) -> dict:
    """Roll a fixed-weight model across time to compute Walk-Forward Efficiency.

    Uses pre-computed *probabilities* and *states* — no model retraining.
    Measures whether signals are consistent across all time periods rather
    than concentrated in a few favourable years.

    Walk-Forward Efficiency (WFE):
        WFE = mean(OOS Sharpe) / mean(IS Sharpe)
        WFE > 0.50 means the strategy retains at least 50% of its IS
        performance on unseen forward-walk windows.

    Args:
        df:             Full featurised DataFrame (must have a DatetimeIndex).
        probabilities:  XGBoost output array aligned with *df*.
        states:         HMM state array aligned with *df*.
        train_days:     IS window length in calendar days.
        test_days:      OOS step/window size in calendar days.
        account_size, broker, tf, prob_threshold, short_threshold:
                        Forwarded to vectorized_backtest unchanged.

    Returns a dict with keys:
        wfe_ratio        — Walk-Forward Efficiency (OOS/IS Sharpe ratio)
        mean_is_sharpe   — average IS Sharpe across all windows
        mean_oos_sharpe  — average OOS Sharpe across all windows
        n_windows        — number of walk-forward windows evaluated
        windows          — list of per-window result dicts (IS+OOS metrics)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for walk-forward windowing.")

    train_td = pd.Timedelta(days=train_days)
    test_td  = pd.Timedelta(days=test_days)
    start    = df.index[0]
    end      = df.index[-1]
    cursor   = start
    windows: list[dict] = []

    while cursor + train_td + test_td <= end:
        is_mask  = (df.index >= cursor) & (df.index < cursor + train_td)
        oos_mask = (df.index >= cursor + train_td) & (df.index < cursor + train_td + test_td)
        n_is  = int(is_mask.sum())
        n_oos = int(oos_mask.sum())

        # Skip windows that are too thin for meaningful statistics
        if n_is < 20 or n_oos < 5:
            cursor += test_td
            continue

        window_mask = is_mask | oos_mask
        idx         = np.where(window_mask)[0]
        df_w     = df.iloc[idx]
        probs_w  = probabilities[idx]
        states_w = states[idx]

        try:
            result_w = vectorized_backtest(
                df_w, probs_w, states_w,
                split_idx=n_is,
                account_size=account_size,
                broker=broker,
                tf=tf,
                regime_stats=regime_stats,
                prob_threshold=prob_threshold,
                short_threshold=short_threshold,
            )
            windows.append({
                "is_start":  cursor,
                "oos_start": cursor + train_td,
                "oos_end":   cursor + train_td + test_td,
                **result_w,
            })
        except Exception as exc:
            logger.warning(
                "WFA window %s skipped: %s",
                cursor.strftime("%Y-%m"), exc,
            )
        cursor += test_td

    if not windows:
        logger.warning(
            "Walk-Forward: no windows produced. "
            "Dataset may be too short for train_days=%d + test_days=%d.",
            train_days, test_days,
        )
        return {
            "wfe_ratio": 0.0, "mean_is_sharpe": 0.0,
            "mean_oos_sharpe": 0.0, "n_windows": 0, "windows": [],
        }

    # Only count windows with enough trades for a meaningful Sharpe
    valid = [
        w for w in windows
        if w.get("is_n_trades", 0) >= 5 and w.get("oos_n_trades", 0) >= 1
    ]
    is_sharpes  = [w["is_sharpe_ratio"]  for w in valid]
    oos_sharpes = [w["oos_sharpe_ratio"] for w in valid]

    mean_is  = float(np.mean(is_sharpes))  if is_sharpes  else 0.0
    mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    wfe      = mean_oos / mean_is          if mean_is > 0 else 0.0

    logger.info(
        "Walk-Forward [%s/%s]: %d windows (%d valid) | IS=%.3f | OOS=%.3f | WFE=%.1f%%",
        tf, broker, len(windows), len(valid), mean_is, mean_oos, wfe * 100,
    )
    return {
        "wfe_ratio":         wfe,
        "mean_is_sharpe":    mean_is,
        "mean_oos_sharpe":   mean_oos,
        "n_windows":         len(windows),
        "n_valid_windows":   len(valid),
        "windows":           windows,
    }


def compare_timeframes(m15_result: dict, h1_result: dict) -> tuple[dict, str]:
    """Compare M15 vs H1 backtest results and recommend the better timeframe.

    Comparison is based on the OOS Sharpe ratio when available, otherwise
    full-period Sharpe.

    Returns:
        results: Dict mapping tf label to its result dict.
        winner: ``"M15"`` or ``"H1"``, whichever has the higher Sharpe.
    """
    results = {"M15": m15_result, "H1": h1_result}

    def _oos_sharpe(r):
        return r.get("oos_sharpe_ratio", r.get("sharpe_ratio", 0.0))

    winner = max(results, key=lambda k: _oos_sharpe(results[k]))

    logger.info("Timeframe comparison:")
    for tf, r in results.items():
        logger.info(
            "  %s  Sharpe=%.3f (OOS=%.3f) | DD=%.1f%% | WR=%.1f%% | Trades=%d",
            tf,
            r["sharpe_ratio"],
            r.get("oos_sharpe_ratio", r["sharpe_ratio"]),
            r["max_drawdown"] * 100,
            r["win_rate"] * 100,
            r["n_trades"],
        )
    logger.info("Winner: %s", winner)
    return results, winner
