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
BULL_STATE     = 0     # HMM Bull regime
BEAR_STATE     = 1     # HMM Bear regime
CHOP_STATE     = 2     # HMM Chop regime (no signals)


def compute_signals(probabilities, hmm_states, threshold=PROB_THRESHOLD,
                    short_threshold=None):
    """Generate directional signals: 1=BUY, -1=SELL, 0=no trade.

    Regime-aligned rules (applied consistently in backtester and live trader):
      BUY  when prob > threshold        AND state == Bull
      SELL when prob < short_threshold  AND state == Bear
      Chop state generates no signals regardless of probability.

    ``short_threshold`` defaults to ``1 - threshold`` when not supplied.
    Note: short_threshold must be < threshold; Optuna penalises crossovers.
    """
    if short_threshold is None:
        short_threshold = 1.0 - threshold
    bull_regime = (hmm_states == BULL_STATE)
    bear_regime = (hmm_states == BEAR_STATE)
    buy  = ((probabilities > threshold)       & bull_regime).astype(int)
    sell = ((probabilities < short_threshold) & bear_regime).astype(int)
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
    broker: str = "standard",
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
    arm = AdaptiveRiskManager(account_size, broker=broker)
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
        win_rate = float(np.sum(per_trade > 0) / n_trades)
    else:
        win_rate = 0.0

    total_return = float(np.exp(np.sum(strategy_returns)) - 1)

    if max_dd > 0:
        recovery_factor = float(min(total_return / max_dd, 20.0))
    elif total_return > 0:
        recovery_factor = 20.0
    else:
        recovery_factor = 0.0

    return {
        "sharpe_ratio":    float(sharpe),
        "max_drawdown":    max_dd,
        "win_rate":        win_rate,
        "total_return":    total_return,
        "n_trades":        n_trades,
        "recovery_factor": recovery_factor,
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

    # Spread efficiency filter: suppress signals where ATR / spread < 1.8.
    # Both atr_norm and spread_frac are expressed as fractions of price so
    # the ratio is scale-free and consistent across all timeframes.
    _spread_frac = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["standard"]).get("spread_frac", 0.0004)
    _er_mask     = (atr_norm / _spread_frac) >= 1.8
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
    result.update({
        "account_size":  account_size,
        "broker":        broker,
        "tf":            tf,
        "pos_per_trade": pos_per_trade,
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
        "%s: account=$%.2f | tier=%s | pos_per_trade=%d | costs=%.4f/trade",
        broker,
        account_size,
        "small" if arm.is_small_account else "growth",
        pos_per_trade,
        BROKER_CONFIGS.get(broker, {}).get("spread_frac", 0)
        + BROKER_CONFIGS.get(broker, {}).get("commission_frac", 0),
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
