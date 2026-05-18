"""Monte Carlo stress-testing suite for GoldRegime_X.

Two simulation modes:
  1. Trade reshuffling  — randomly reorders the chronological PnL sequence
     10,000 times to measure the 95th-percentile MaxDD and Risk of Ruin.
  2. Price perturbation — implemented in backtester.vectorized_backtest via
     the ``noise_std`` parameter; this module only handles trade reshuffling.
"""
import numpy as np

from src.logger import setup_logger

logger = setup_logger(__name__)


def run_trade_reshuffle(
    trades: list[dict],
    initial_balance: float,
    iterations: int = 10_000,
    ruin_fraction: float = 0.50,
) -> dict:
    """Randomly reshuffles the chronological sequence of trade PnLs to simulate
    alternative realities and calculate the 95th-percentile Maximum Drawdown.

    Args:
        trades:          List of trade dicts, each containing a ``'pnl'`` key
                         (absolute dollar PnL for the trade).
        initial_balance: Starting account balance in USD.
        iterations:      Number of Monte Carlo paths to simulate (default 10,000).
        ruin_fraction:   Fraction of initial balance at or below which the account
                         is considered "ruined" (default 0.50 → 50% loss).

    Returns:
        Dict with keys:
            ``median_dd``            — median MaxDD across all paths (fraction).
            ``95th_percentile_dd``   — 95th-pctl MaxDD across all paths (fraction).
            ``risk_of_ruin_pct``     — % of paths that hit the ruin threshold.
    """
    if not trades:
        logger.warning("run_trade_reshuffle: no trades supplied — returning error result.")
        return {"error": "No trades to simulate."}

    pnls = np.array([t["pnl"] for t in trades], dtype=np.float64)
    if len(pnls) == 0 or np.all(pnls == 0):
        logger.warning("run_trade_reshuffle: all PnLs are zero — results will be trivial.")

    ruin_threshold = initial_balance * (1.0 - ruin_fraction)
    max_dds        = np.zeros(iterations, dtype=np.float64)
    ruined_count   = 0

    rng = np.random.default_rng()   # seedless — true randomness per call

    for i in range(iterations):
        shuffled   = rng.permutation(pnls)
        equity     = initial_balance + np.cumsum(shuffled)

        if np.any(equity <= ruin_threshold):
            ruined_count += 1

        running_max     = np.maximum.accumulate(equity)
        # Avoid division by zero if equity is always zero (pathological input)
        safe_max        = np.where(running_max > 0, running_max, 1.0)
        drawdowns       = (running_max - equity) / safe_max
        max_dds[i]      = float(np.max(drawdowns))

    median_dd      = float(np.median(max_dds))
    dd_95th        = float(np.percentile(max_dds, 95))
    risk_of_ruin   = (ruined_count / iterations) * 100.0

    logger.info(
        "Monte Carlo (%d iterations): Median DD: %.1f%% | 95th Pctl DD: %.1f%% | "
        "Risk of Ruin (≥%.0f%% loss): %.2f%%",
        iterations, median_dd * 100, dd_95th * 100, ruin_fraction * 100, risk_of_ruin,
    )

    return {
        "median_dd":           median_dd,
        "95th_percentile_dd":  dd_95th,
        "risk_of_ruin_pct":    risk_of_ruin,
    }
