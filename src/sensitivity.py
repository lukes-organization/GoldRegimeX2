"""
Sensitivity analysis for Z-Score cutoffs.

Tests Bull/Bear thresholds across a range of values on already-trained models
and reports trade count, win rate, Sharpe, max DD, and profit factor for each.

Does NOT re-train the model — probabilities are computed once from the saved
ensemble, then vectorized_backtest is called once per Z value with a config
override injected into SignalEvaluator via the evaluator_config parameter.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester import vectorized_backtest
from src.logger import setup_logger

logger = setup_logger(__name__)


def run_sensitivity(
    tf: str,
    broker: str,
    balance: float,
    df_aligned: pd.DataFrame,
    probabilities: np.ndarray,
    states_aligned: np.ndarray,
    split_idx: int,
    regime_stats: dict,
    z_range: np.ndarray | None = None,
    output_dir: str = "reports",
    use_tiered: bool = False,
) -> pd.DataFrame:
    """Test a range of Bull/Bear Z-Score cutoffs and return a results DataFrame.

    The MR (Chop) cutoffs are left at their TF-specific defaults throughout —
    only Z_CUTOFF_BULL and Z_CUTOFF_BEAR are swept.

    Args:
        tf:            Timeframe string (e.g. "H1", "M5").
        broker:        Broker key (e.g. "headway_cent").
        balance:       Account size in USD.
        df_aligned:    Feature DataFrame returned by prepare_features (aligned).
        probabilities: XGB probability array aligned with df_aligned.
        states_aligned: HMM state array aligned with df_aligned.
        split_idx:     IS/OOS split index.
        regime_stats:  Per-state probability stats from compute_regime_stats.
        z_range:       Array of Bull Z cutoffs to test.  Defaults to
                       ``np.arange(1.5, 3.1, 0.25)``.
        output_dir:    Directory to write CSV/JSON results.

    Returns:
        DataFrame with one row per Z cutoff.
    """
    if z_range is None:
        z_range = np.arange(1.5, 3.1, 0.25)  # 1.5 .. 3.0 in 0.25 steps

    # Phase 3 (updated): Z overrides are now threaded through evaluator_config
    # → vectorized_backtest → _run_bar_loop → SignalEngine.should_enter via
    # z_cutoff_bull / z_cutoff_bear, so each Z value genuinely changes entry
    # decisions.  The logger.warning below is intentionally removed.

    # Determine reference Z using current SignalEngine MIN_TREND_ZSCORE.
    # signal_evaluator was removed in the SignalEngine refactor; Z-score
    # thresholds now live in signal_engine.MIN_TREND_ZSCORE.
    from src.signal_engine import MIN_TREND_ZSCORE
    current_z = float(MIN_TREND_ZSCORE.get(tf.upper(), 1.0))

    logger.info(
        "Sensitivity analysis [%s/%s] balance=$%.0f  mode=%s",
        tf, broker, balance, "TIERED" if use_tiered else "STANDARD",
    )
    logger.info("  Current Z cutoff: ±%.2f", current_z)
    logger.info("  Testing range: %s", [round(float(z), 2) for z in z_range])
    logger.info("  Bars in dataset: %d  (split_idx=%d)", len(df_aligned), split_idx)

    rows = []
    for z in z_range:
        z = float(round(z, 2))
        cfg_override = {"Z_CUTOFF_BULL": z, "Z_CUTOFF_BEAR": -z}
        logger.info("Sensitivity override active [%s]: bull=%.2f bear=%.2f", tf, z, -z)

        result = vectorized_backtest(
            df_aligned, probabilities, states_aligned,
            split_idx=split_idx,
            account_size=balance,
            broker=broker,
            tf=tf,
            regime_stats=regime_stats,
            evaluator_config=cfg_override,
            use_tiered=use_tiered,
        )

        # Prefer OOS metrics when available, fall back to full-period
        oos_trades = result.get("oos_n_trades", result.get("n_trades", 0))
        oos_wr     = result.get("oos_win_rate",  result.get("win_rate", 0.0))
        oos_sharpe = result.get("oos_sharpe_ratio", result.get("sharpe_ratio", 0.0))
        oos_dd     = result.get("oos_max_drawdown",  result.get("max_drawdown", 0.0))
        oos_pf     = result.get("oos_profit_factor",  result.get("profit_factor", 1.0))
        oos_rf     = result.get("oos_recovery_factor", result.get("recovery_factor", 0.0))
        oos_ret    = result.get("oos_total_return",   result.get("total_return", 0.0))

        rows.append({
            "z_cutoff":        z,
            "oos_trades":      oos_trades,
            "oos_win_rate":    round(oos_wr * 100, 1),
            "oos_sharpe":      round(oos_sharpe, 3),
            "oos_max_dd_pct":  round(oos_dd * 100, 2),
            "oos_pf":          round(oos_pf, 2),
            "oos_recovery":    round(oos_rf, 2),
            "oos_return_pct":  round(oos_ret * 100, 2),
            "is_current":      abs(z - current_z) < 0.01,
            "is_best":         False,
        })

        logger.info(
            "  Z=%.2f | trades=%d | sharpe=%.3f | dd=%.1f%% | wr=%.1f%% | pf=%.2f",
            z, oos_trades, oos_sharpe, oos_dd * 100, oos_wr * 100, oos_pf,
        )

    df_res = pd.DataFrame(rows)

    # Mark best by Sharpe (most intuitive primary metric for the user)
    if len(df_res) > 0 and df_res["oos_sharpe"].max() > 0:
        df_res.loc[df_res["oos_sharpe"].idxmax(), "is_best"] = True

    _print_table(df_res, tf, broker, use_tiered=use_tiered)
    _save_results(df_res, tf, broker, output_dir, use_tiered=use_tiered)
    return df_res


# ── Internal helpers ──────────────────────────────────────────────────────────

def _print_table(df_res: pd.DataFrame, tf: str, broker: str, use_tiered: bool = False) -> None:
    mode_label = " [TIERED]" if use_tiered else ""
    rule = "=" * 84
    print(f"\n{rule}")
    print(f"  Z-Score Sensitivity Analysis — {tf} / {broker}{mode_label}")
    print(rule)
    print(f"{'Z Cut':>8} {'Trades':>8} {'Win%':>7} {'Sharpe':>8} "
          f"{'MaxDD%':>8} {'PF':>6} {'RF':>7} {'Ret%':>8}  Note")
    print("-" * 84)
    for _, row in df_res.iterrows():
        note = ""
        if row["is_current"] and row["is_best"]:
            note = "★ BEST (CURRENT)"
        elif row["is_best"]:
            note = "★ BEST"
        elif row["is_current"]:
            note = "<-- current"

        print(
            f"{row['z_cutoff']:>8.2f} {row['oos_trades']:>8d} {row['oos_win_rate']:>6.1f}% "
            f"{row['oos_sharpe']:>8.3f} {row['oos_max_dd_pct']:>7.1f}% "
            f"{row['oos_pf']:>6.2f} {row['oos_recovery']:>7.2f} "
            f"{row['oos_return_pct']:>7.1f}%  {note}"
        )
    print(f"{rule}\n")


def _save_results(df_res: pd.DataFrame, tf: str, broker: str, output_dir: str, use_tiered: bool = False) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_tiered" if use_tiered else ""

    csv_path = out / f"sensitivity_{tf}_{broker}{suffix}.csv"
    df_res.to_csv(csv_path, index=False)
    logger.info("Sensitivity CSV: %s", csv_path)

    best_row = df_res[df_res["is_best"]]
    curr_row = df_res[df_res["is_current"]]
    summary = {
        "tf":          tf,
        "broker":      broker,
        "mode":        "tiered" if use_tiered else "standard",
        "current_z":   float(curr_row["z_cutoff"].values[0]) if not curr_row.empty else None,
        "best_z":      float(best_row["z_cutoff"].values[0]) if not best_row.empty else None,
        "best_sharpe": float(best_row["oos_sharpe"].values[0]) if not best_row.empty else None,
        "results":     df_res.to_dict(orient="records"),
    }
    json_path = out / f"sensitivity_{tf}_{broker}{suffix}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info("Sensitivity JSON: %s", json_path)
