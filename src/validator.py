"""Model Validation Gatekeeper (consolidated onto the notebook engine).

Validation now runs the SAME grid-search-plateau engine used by --mode optimize
and by live trading (via ``src.strategy_backtest``) over a recent window of the
exec-timeframe panel, then classifies the result.  Because the model that is
validated is the exact live bundle that is traded, this Sharpe is directly
comparable to live behaviour.

The legacy processor -> HMM -> XGB -> vectorized_backtest path was retired in
the consolidation (see CONSOLIDATION.md).  ``obs_cov`` / ``trans_cov`` and the
MR-leak / regime-coverage / fold-instability gates are retained only for API
compatibility; with the notebook engine they are no-ops.
"""

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import setup_logger
from src.strategy_backtest import backtest_tf, load_live_bundle, LIVE_MODEL_PKL

logger = setup_logger(__name__)

SYNC_DATA_PATH        = Path("data/processed/mt5_sync_data.csv")
SHARPE_PASS_THRESHOLD = 0.8
SHARPE_WARN_THRESHOLD = 0.5
# H1 fires rarely; relax per-TF thresholds to avoid false fails on small samples.
TF_SHARPE_PASS: dict = {"H1": 0.25, "M15": 0.50, "M5": 0.70}
TF_SHARPE_WARN: dict = {"H1": 0.05, "M15": 0.20, "M5": 0.40}
MIN_TRADES_WARNING_BY_TF: dict = {"M5": 30, "M15": 15, "H1": 10}

# Recent-window sizes (in bars) per timeframe when ``period`` is not supplied.
_DEFAULT_WINDOW_BARS = {"M5": 20000, "M15": 12000, "H1": 6000}
# Approximate bars-per-calendar-day per timeframe for parsing ``period``.
_BARS_PER_DAY = {"M5": 288, "M15": 96, "H1": 24}


def _window_bars(tf: str, period) -> int:
    """Resolve a recent-window length in bars from a ``period`` string.

    Accepts '3m', '6m', '1y', '90d', '4w'.  Falls back to the per-TF default.
    """
    tf = tf.upper()
    default = _DEFAULT_WINDOW_BARS.get(tf, 6000)
    if not period:
        return default
    try:
        unit = str(period)[-1].lower()
        qty = float(str(period)[:-1])
        per_day = _BARS_PER_DAY.get(tf, 24)
        if unit == "d":
            days = qty
        elif unit == "w":
            days = qty * 7
        elif unit == "m":
            days = qty * 30
        elif unit == "y":
            days = qty * 365
        else:
            return default
        return max(500, int(days * per_day))
    except Exception:
        return default


def run_validation(
    sync_data_path: Path = SYNC_DATA_PATH,
    tf: str = "H1",
    broker: str = "headway_cent",
    account_size: float = 15.0,
    obs_cov: float = None,     # retired (Kalman); accepted for API compatibility
    trans_cov: float = None,   # retired (Kalman); accepted for API compatibility
    period=None,
) -> dict:
    """Validate the live bundle against a recent window of the exec panel.

    Returns a dict compatible with the historical validator surface (sharpe,
    n_trades, win_rate, max_dd, status, message, ...).

    Raises FileNotFoundError if the live bundle is missing.
    """
    tf = tf.upper()
    try:
        bundle = load_live_bundle()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Live model bundle not found at {LIVE_MODEL_PKL}. "
            "Run  python main.py --mode optimize  first."
        )

    tail = _window_bars(tf, period)
    logger.info("Validating [%s] on the most recent ~%d bars (broker=%s).", tf, tail, broker)

    # backtest_tf builds features on the full engine panel, then slices the
    # feature frame to the recent window (so rolling features stay correct).
    metrics = backtest_tf(tf, bundle=bundle, tail=tail)

    sharpe        = float(metrics.get("sharpe", 0.0))
    n_trades      = int(metrics.get("trade_count", 0))
    win_rate      = float(metrics.get("win_rate", 0.0))
    max_dd_pct    = float(metrics.get("max_drawdown", 0.0))   # PERCENT from engine
    max_dd_frac   = max_dd_pct / 100.0                        # fraction for the DD gate
    pf            = float(metrics.get("profit_factor", 1.0))
    total_return  = float(metrics.get("net_return_pct", 0.0)) # PERCENT
    score         = float(metrics.get("stability_adjusted_sharpe", sharpe))
    recovery      = float(metrics.get("calmar", 0.0))

    min_trades_warn = MIN_TRADES_WARNING_BY_TF.get(tf, 15)
    if n_trades < min_trades_warn:
        logger.warning(
            "Only %d trades in the validation window; Sharpe estimate may be "
            "unreliable. Consider a longer --period.", n_trades,
        )

    _pass_thr = TF_SHARPE_PASS.get(tf, SHARPE_PASS_THRESHOLD)
    _warn_thr = TF_SHARPE_WARN.get(tf, SHARPE_WARN_THRESHOLD)
    if sharpe >= _pass_thr:
        status  = "pass"
        message = (
            f"Recent-window Sharpe {sharpe:.3f} >= {_pass_thr} [{tf}]. "
            "Model is stable; safe to go live."
        )
    elif sharpe >= _warn_thr:
        status  = "warn"
        message = (
            f"Recent-window Sharpe {sharpe:.3f} is borderline "
            f"({_warn_thr}-{_pass_thr}, {tf}). Proceed with reduced size or wait."
        )
    else:
        status  = "fail"
        message = (
            f"Recent-window Sharpe {sharpe:.3f} < {_warn_thr} [{tf}]. "
            "Drift detected; DO NOT go live. Re-run --mode optimize."
        )

    logger.info(
        "Validation [%s]: status=%s  score=%.2f  sharpe=%.3f  pf=%.2f  trades=%d  wr=%.1f%%  dd=%.1f%%",
        tf, status, score, sharpe, pf, n_trades, win_rate * 100, max_dd_pct,
    )
    if status in ("warn", "fail"):
        logger.warning("VALIDATION %s: %s", status.upper(), message)

    return {
        "sharpe":                 sharpe,
        "n_trades":               n_trades,
        "win_rate":               win_rate,
        "floating_max_drawdown":  max_dd_frac,
        "max_drawdown":           max_dd_frac,
        "max_dd":                 max_dd_frac,
        "mr_trades":              0,
        "mr_leak_count":          0,
        "regime_coverage":        1.0,
        "fold_sharpe_std":        0.0,
        "profit_factor":          pf,
        "expected_payoff":        0.0,
        "recovery_factor":        recovery,
        "avg_efficiency":         0.0,
        "cost_efficiency":        0.0,
        "total_return":           total_return,
        "score":                  score,
        "status":                 status,
        "message":                message,
    }


# Minimum per-TF trade counts for a strategy to be deployable.
_MIN_TRADES_PER_TF = {"H1": 30, "M15": 60, "M5": 120}


def validate_strategy(result: dict, tf: str, metrics: dict = None) -> dict:
    """Deployment gate: hard-fail or warn based on backtest result quality.

    Priority: FAIL dd_cap_violated (DD>20%) -> FAIL mr_leak (now always 0) ->
    FAIL min_trades -> WARN fold_instability (now 0) -> WARN regime_instability
    (now 1.0, never fires).
    """
    _m = dict(result)
    if metrics:
        _m.update(metrics)

    tf_up = tf.upper()
    min_trades = _MIN_TRADES_PER_TF.get(tf_up, 30)

    details: dict = {}
    warnings: list = []

    floating_dd = float(
        _m.get("floating_max_drawdown", _m.get("max_drawdown", _m.get("max_dd", 0.0)))
    )
    details["floating_dd"] = floating_dd
    if floating_dd > 0.20:
        return {"status": "fail", "reason": "dd_cap_violated",
                "details": {**details, "threshold": 0.20}}

    mr_leak = int(_m.get("mr_leak_count", _m.get("mr_trades", 0)))
    details["mr_leak_count"] = mr_leak
    if mr_leak > 0:
        return {"status": "fail", "reason": "mr_leak",
                "details": {**details, "mr_trades_found": mr_leak}}

    n_trades = int(_m.get("n_trades", 0))
    details["n_trades"] = n_trades
    if n_trades < min_trades:
        return {"status": "fail", "reason": "min_trades",
                "details": {**details, "required": min_trades}}

    fold_sharpe_std = float(_m.get("fold_sharpe_std", _m.get("std_sharpe", 0.0)))
    details["fold_sharpe_std"] = fold_sharpe_std
    if fold_sharpe_std > 1.5:
        warnings.append("fold_instability")

    regime_coverage = float(_m.get("regime_coverage", 1.0))
    details["regime_coverage"] = regime_coverage
    if regime_coverage < 0.20:
        warnings.append("regime_instability")

    if warnings:
        return {"status": "warn", "reason": warnings[0],
                "details": {**details, "all_warnings": warnings}}

    return {"status": "pass", "reason": "", "details": details}


def check_model_age(tf: str = "H1", broker: str = "headway_cent") -> float:
    """Return the age of the live model bundle in days (float('inf') if absent)."""
    path = Path(LIVE_MODEL_PKL)
    if not path.exists():
        return float("inf")
    age_sec = _time.time() - path.stat().st_mtime
    return age_sec / 86_400
