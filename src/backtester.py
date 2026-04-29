import numpy as np
import pandas as pd
from src.logger import setup_logger
from src.risk_manager import BROKER_CONFIGS, AdaptiveRiskManager
from src.signal_evaluator import SignalEvaluator

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
PROB_THRESHOLD = 0.65
BULL_STATE     = 0     # HMM Bull regime
BEAR_STATE     = 1     # HMM Bear regime
CHOP_STATE     = 2     # HMM Chop regime (no signals)


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


def compute_signals(probabilities, hmm_states, threshold=PROB_THRESHOLD,
                    short_threshold=None):
    """Generate directional signals: 1=BUY, -1=SELL, 0=no trade.

    Two-regime signal logic:

    Trend regime (Bull=0 / Bear=1) — trade WITH the model's conviction:
      BUY  when prob > threshold        AND state == Bull
      SELL when prob < short_threshold  AND state == Bear

    Mean Reversion regime (Chop=2/3) — trade AGAINST extreme readings,
    betting price will snap back to the local mean ("Snap-Back" logic):
      BUY  when prob < (short_threshold - 0.10)  AND state is Chop
      SELL when prob > (threshold + 0.10)         AND state is Chop

    ``short_threshold`` defaults to ``1 - threshold`` when not supplied.
    Note: short_threshold must be < threshold; Optuna penalises crossovers.
    Trend and MR signals are mutually exclusive (different state conditions).
    """
    if short_threshold is None:
        short_threshold = 1.0 - threshold

    bull_regime = (hmm_states == BULL_STATE)
    bear_regime = (hmm_states == BEAR_STATE)
    chop_regime = (hmm_states >= CHOP_STATE)   # state 2 or 3

    # Trend signals (regime-aligned)
    buy  = ((probabilities > threshold)       & bull_regime).astype(int)
    sell = ((probabilities < short_threshold) & bear_regime).astype(int)

    # Mean reversion signals (Chop snap-back)
    mr_buy  = ((probabilities < (short_threshold - 0.10)) & chop_regime).astype(int)
    mr_sell = ((probabilities > (threshold + 0.10))       & chop_regime).astype(int)

    return (buy - sell + mr_buy - mr_sell).clip(-1, 1)


def compute_signals_zscore(
    probabilities: np.ndarray,
    hmm_states:    np.ndarray,
    regime_stats:  dict,
    gmm_clusters:  np.ndarray | None = None,
    tf:            str = "H1",
    evaluator_config: dict | None = None,
    use_tiered:    bool = False,
) -> np.ndarray:
    """Vectorised Z-Score signal generation — used when regime_stats are available.

    No live-safety gates (consecutive-bars / transition-prob / Bollinger Bands)
    are applied here.  Those are execution safeguards added only by the live
    bridge.  This function evaluates the *pure signal edge* so the backtester
    and optimiser score the same logic the SignalEvaluator implements.

    Signal encoding matches :func:`compute_signals`: ``1=BUY, -1=SELL, 0=no trade``.
    MR_BUY and MR_SELL are collapsed to +1 / −1 respectively.
    """
    evaluator = SignalEvaluator(regime_stats, tf=tf, config=evaluator_config)
    n         = len(probabilities)
    signals   = np.zeros(n, dtype=np.int8)
    for i in range(n):
        gmc = int(gmm_clusters[i]) if gmm_clusters is not None else 1
        sig, _ = evaluator.evaluate_signal_fast(
            prob_buy=float(probabilities[i]),
            hmm_state=int(hmm_states[i]),
            gmm_cluster=gmc,
            use_tiered=use_tiered,
        )
        if sig in ("BUY", "MR_BUY"):
            signals[i] = 1
        elif sig in ("SELL", "MR_SELL"):
            signals[i] = -1
    return signals


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


def vectorized_backtest(
    df,
    probabilities,
    hmm_states,
    split_idx=None,
    account_size: float = 15.0,
    broker: str = "standard",
    tf: str = "H1",
    prob_threshold: float = None,
    short_threshold: float = None,
    regime_stats: dict = None,
    evaluator_config: dict = None,
    use_tiered: bool = False,
    return_trades: bool = False,
):
    """Vectorized backtest with adaptive session limits and broker costs.

    Args:
        df: Processed DataFrame with ``log_return`` and ``atr_normalized``.
        probabilities: XGB probability array aligned with df.
        hmm_states: HMM state array aligned with df.
        split_idx: Bar index for IS/OOS split (None = full-period only).
        account_size: Account balance in USD; drives adaptive session limits.
        broker: Broker profile key from ``BROKER_CONFIGS``.
        tf: Timeframe string used for correct annualization ("H1" or "M15").
        prob_threshold: BUY threshold (legacy fixed-threshold mode).  Ignored
                        when *regime_stats* is supplied.
        short_threshold: SELL threshold (legacy).  Ignored when *regime_stats*
                         is supplied.
        regime_stats: Per-HMM-state probability statistics produced by
                      :func:`~src.engine_xgb.compute_regime_stats`.  When
                      provided, signals are generated via Z-Score calibration
                      (:func:`compute_signals_zscore`) instead of the legacy
                      fixed-threshold method.
        return_trades: When True, add ``'trades_df'`` to the result — a
                       DataFrame of IS-bar records used to calibrate the
                       :class:`~src.rcev_scorer.RCEVScorer`.  Has no effect
                       when ``split_idx`` is None (uses all bars).
    """
    log_returns = df["log_return"].values
    atr_norm    = df["atr_normalized"].values

    # ── Signal generation ─────────────────────────────────────────────────
    if regime_stats:
        gmm_clusters = (
            df["gmm_vol_cluster"].values
            if "gmm_vol_cluster" in df.columns
            else None
        )
        raw_signals = compute_signals_zscore(
            probabilities, hmm_states, regime_stats, gmm_clusters, tf=tf,
            evaluator_config=evaluator_config,
            use_tiered=use_tiered,
        )
    else:
        buy_th = prob_threshold if prob_threshold is not None else PROB_THRESHOLD
        raw_signals = compute_signals(
            probabilities, hmm_states,
            threshold=buy_th,
            short_threshold=short_threshold,  # None → symmetric default inside
        )

    # Spread-Aware Adaptive Filter: suppress signals where ATR / spread is below
    # the TF-specific minimum efficiency threshold.
    # M5: 7.0× — ensures MR trades only fire when volatility comfortably exceeds
    #     the spread, filtering expensive noise in quiet night sessions.
    # H1/M15: 1.25× — allows standard market conditions while blocking dead sessions.
    _spread_frac = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["standard"]).get("spread_frac", 0.0004)
    _min_eff     = TF_MIN_EFFICIENCY.get(tf.upper(), 1.25)
    _er_mask     = (atr_norm / _spread_frac) >= _min_eff
    signals = np.where(_er_mask, raw_signals, 0)

    # Determine pos_per_trade from the adaptive risk manager
    arm           = AdaptiveRiskManager(account_size, broker=broker)
    base_limits   = arm.get_trade_limits(tf=tf)
    pos_per_trade = base_limits["pos_per_trade"]

    sizes = compute_position_sizes(signals, atr_norm, pos_per_trade=pos_per_trade)

    next_returns     = np.roll(log_returns, -1)
    next_returns[-1] = 0.0
    gross_returns    = sizes * next_returns
    costs            = compute_trade_costs(signals, broker=broker)
    strategy_returns = gross_returns - costs

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
        "pos_per_trade": pos_per_trade,
    })
    # Full-period MR attribution
    result.update(_mr_attribution(signals, hmm_states, strategy_returns))

    if split_idx is not None and 0 < split_idx < len(strategy_returns):
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
        "%s: account=$%.2f | tier=%s | pos_per_trade=%d | costs=%.4f/trade",
        broker,
        account_size,
        "small" if arm.is_small_account else "growth",
        pos_per_trade,
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

    # ── Per-trade records for RCEV calibration ────────────────────────────────
    if return_trades:
        _spread_val = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["standard"]).get("spread_frac", 0.0004)
        _is_end = split_idx if split_idx is not None else len(signals)
        _sig_is = signals[:_is_end]
        _active_idx = np.where(_sig_is != 0)[0]
        _gmm = (
            df["gmm_vol_cluster"].values[:_is_end]
            if "gmm_vol_cluster" in df.columns
            else np.ones(_is_end, dtype=int)
        )
        trades_list = [
            {
                "bar_idx":         int(i),
                "regime":          int(hmm_states[i]),
                "signal":          int(_sig_is[i]),
                "prob":            float(probabilities[i]),
                "volatility":      float(atr_norm[i]),
                "spread":          float(_spread_val),
                "gmm_vol_cluster": int(_gmm[i]),
                "pnl":             float(strategy_returns[i]),
            }
            for i in _active_idx
        ]
        _empty_cols = ["bar_idx", "regime", "signal", "prob", "volatility",
                       "spread", "gmm_vol_cluster", "pnl"]
        result["trades_df"] = (
            pd.DataFrame(trades_list) if trades_list else pd.DataFrame(columns=_empty_cols)
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
