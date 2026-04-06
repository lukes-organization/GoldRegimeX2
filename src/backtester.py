import numpy as np
import pandas as pd
from src.logger import setup_logger
from src.risk_manager import BROKER_CONFIGS, AdaptiveRiskManager

logger = setup_logger(__name__)

ANNUALIZATION_FACTORS = {
    "H1":  np.sqrt(252 * 24),
    "M15": np.sqrt(252 * 96),    # 96 M15 bars per day
    "M5":  np.sqrt(252 * 288),   # 288 M5 bars per day
}
ANNUALIZATION_FACTOR = ANNUALIZATION_FACTORS["H1"]   # default

RISK_PER_TRADE = 0.01
PROB_THRESHOLD = 0.65
CHOP_STATE     = 2


def compute_signals(probabilities, hmm_states, threshold=PROB_THRESHOLD,
                    short_threshold=None):
    """Generate directional signals: 1=BUY, -1=SELL, 0=no trade.

    BUY  when prob > threshold       and state != Chop
    SELL when prob < short_threshold and state != Chop

    ``short_threshold`` defaults to ``1 - threshold`` when not supplied,
    which gives a symmetric no-trade zone around 0.5.
    Note: short_threshold must be < threshold to avoid a no-trade zone gap
    inversion.  If they cross Optuna penalises the trial via the objective.
    """
    if short_threshold is None:
        short_threshold = 1.0 - threshold
    not_chop = (hmm_states != CHOP_STATE)
    buy  = ((probabilities > threshold)       & not_chop).astype(int)
    sell = ((probabilities < short_threshold) & not_chop).astype(int)
    return buy - sell   # 1=BUY, -1=SELL, 0=hold


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


def apply_session_limits(
    signals: np.ndarray,
    dates,
    account_size: float = 15.0,
    hmm_states: np.ndarray = None,
    tf: str = "H1",
) -> np.ndarray:
    """Cap signals to the adaptive daily limit based on account size, HMM state and TF.

    Counts both BUY (+1) and SELL (-1) signals toward the daily cap.

    For growth accounts (> $50), the limit is market-state-dependent:
    - Bull / Bear: 3 signals/day
    - Chop: 2 signals/day

    M5 small accounts get a higher cap (4/day) to match the 288-bar/day
    frequency and allow the optimizer to generate enough OOS trades.

    Uses the majority HMM state within each calendar day to decide the cap.
    """
    arm = AdaptiveRiskManager(account_size)
    result = signals.copy()
    day_labels = pd.DatetimeIndex(dates).normalize().values

    unique_days, day_starts = np.unique(day_labels, return_index=True)
    day_ends = np.append(day_starts[1:], len(signals))

    for start, end in zip(day_starts, day_ends):
        # Market-state-aware limit for growth accounts
        if hmm_states is not None and not arm.is_small_account:
            day_states = hmm_states[start:end].astype(int)
            day_mode = int(np.bincount(day_states).argmax())
            limits = arm.get_trade_limits(day_mode, tf=tf)
        else:
            limits = arm.get_trade_limits(tf=tf)

        max_daily = limits["max_daily_trades"]
        seg = result[start:end]
        # Count BUY (+1) and SELL (-1) trades together toward the daily cap
        trade_indices = np.where(np.abs(seg) == 1)[0]
        if len(trade_indices) > max_daily:
            seg[trade_indices[max_daily:]] = 0
        result[start:end] = seg

    original = int(np.sum(np.abs(signals)))
    limited  = int(np.sum(np.abs(result)))
    if original != limited:
        logger.debug(
            "Session limit (acct=$%.0f): %d -> %d trades",
            account_size, original, limited,
        )
    return result


def compute_trade_costs(signals: np.ndarray, broker: str = "standard") -> np.ndarray:
    """Apply spread + commission cost to every active trade (BUY or SELL)."""
    config = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["standard"])
    cost_per_trade = config["spread_frac"] + config["commission_frac"]
    return np.abs(signals) * cost_per_trade   # cost is always positive


def _compute_metrics(strategy_returns, signals, tf: str = "H1"):
    ann_factor = ANNUALIZATION_FACTORS.get(tf.upper(), ANNUALIZATION_FACTOR)
    mean_ret = np.mean(strategy_returns)
    std_ret  = np.std(strategy_returns)
    sharpe   = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0.0

    cumulative  = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns   = running_max - cumulative
    max_dd      = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Count both BUY (+1) and SELL (-1) as active trades
    active         = np.abs(signals) == 1
    active_returns = strategy_returns[active[:len(strategy_returns)]]
    n_trades       = int(np.sum(active))
    win_rate       = (float(np.sum(active_returns > 0) / len(active_returns))
                      if len(active_returns) > 0 else 0.0)
    total_return   = float(np.exp(np.sum(strategy_returns)) - 1)

    return {
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "win_rate":     win_rate,
        "total_return": total_return,
        "n_trades":     n_trades,
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
        prob_threshold: BUY threshold — signal when prob > this value.
                        Defaults to PROB_THRESHOLD (0.65) if not supplied.
        short_threshold: SELL threshold — signal when prob < this value.
                         Defaults to ``1 - prob_threshold`` if not supplied,
                         giving a symmetric no-trade zone around 0.5.
    """
    log_returns = df["log_return"].values
    atr_norm    = df["atr_normalized"].values

    buy_th = prob_threshold  if prob_threshold  is not None else PROB_THRESHOLD
    raw_signals = compute_signals(
        probabilities, hmm_states,
        threshold=buy_th,
        short_threshold=short_threshold,   # None → symmetric default inside
    )
    signals = apply_session_limits(raw_signals, df.index, account_size, hmm_states, tf=tf)

    # Determine pos_per_trade from the adaptive risk manager
    arm           = AdaptiveRiskManager(account_size)
    base_limits   = arm.get_trade_limits(tf=tf)
    pos_per_trade = base_limits["pos_per_trade"]

    sizes = compute_position_sizes(signals, atr_norm, pos_per_trade=pos_per_trade)

    next_returns     = np.roll(log_returns, -1)
    next_returns[-1] = 0.0
    gross_returns    = sizes * next_returns
    costs            = compute_trade_costs(signals, broker=broker)
    strategy_returns = gross_returns - costs

    result = _compute_metrics(strategy_returns, signals, tf=tf)
    result.update({
        "account_size":      account_size,
        "broker":            broker,
        "tf":                tf,
        "session_max_trades": arm.get_trade_limits(tf=tf)["max_daily_trades"],
        "pos_per_trade":     pos_per_trade,
    })

    if split_idx is not None and 0 < split_idx < len(strategy_returns):
        is_m  = _compute_metrics(strategy_returns[:split_idx], signals[:split_idx], tf)
        oos_m = _compute_metrics(strategy_returns[split_idx:], signals[split_idx:], tf)
        result["split_idx"] = split_idx
        for k, v in is_m.items():
            result[f"is_{k}"] = v
        for k, v in oos_m.items():
            result[f"oos_{k}"] = v
        logger.info(
            "Backtest IS  [%s]: Sharpe=%.3f | MaxDD=%.3f | WinRate=%.3f | Trades=%d",
            tf, is_m["sharpe_ratio"], is_m["max_drawdown"], is_m["win_rate"], is_m["n_trades"],
        )
        logger.info(
            "Backtest OOS [%s]: Sharpe=%.3f | MaxDD=%.3f | WinRate=%.3f | Trades=%d",
            tf, oos_m["sharpe_ratio"], oos_m["max_drawdown"], oos_m["win_rate"], oos_m["n_trades"],
        )
    else:
        logger.info(
            "Backtest [%s]: Sharpe=%.3f | MaxDD=%.3f | WR=%.3f | Return=%.1f%% | Trades=%d",
            tf, result["sharpe_ratio"], result["max_drawdown"], result["win_rate"],
            result["total_return"] * 100, result["n_trades"],
        )

    logger.info(
        "%s: account=$%.2f | tier=%s | session_limit=%d/day | pos_per_trade=%d | costs=%.4f/trade",
        broker,
        account_size,
        "small" if arm.is_small_account else "growth",
        arm.get_trade_limits(tf=tf)["max_daily_trades"],
        pos_per_trade,
        BROKER_CONFIGS.get(broker, {}).get("spread_frac", 0)
        + BROKER_CONFIGS.get(broker, {}).get("commission_frac", 0),
    )
    return result


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
