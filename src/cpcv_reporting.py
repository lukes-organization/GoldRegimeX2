from __future__ import annotations

import json
from pathlib import Path


def build_cpcv_report(
    tf: str,
    broker: str,
    optimization_summary: dict,
    cpcv_result: dict,
    objective_breakdown: dict,
    final_score: float,
    penalties: dict,
) -> dict:
    tf_up = tf.upper()

    return {
        "schema_version": "1.0",
        "timeframe": tf_up,
        "broker": broker,
        "optimization_summary": {
            "study_name": optimization_summary.get("study_name"),
            "best_trial": optimization_summary.get("best_trial"),
            "completed_trials": optimization_summary.get("completed_trials"),
            "search_space": optimization_summary.get("search_space", []),
        },
        "cpcv_aggregate_stats": {
            "cpcv_score": float(cpcv_result.get("cpcv_score", 0.0)),
            "n_valid_paths": int(cpcv_result.get("n_valid_paths", 0)),
            "median_sharpe": float(cpcv_result.get("median_sharpe", 0.0)),
            "std_sharpe": float(cpcv_result.get("std_sharpe", 0.0)),
            "median_trades": int(cpcv_result.get("median_trades", 0)),
            "median_win_rate": float(cpcv_result.get("median_win_rate", 0.0)),
            "median_drawdown": float(cpcv_result.get("median_drawdown", 0.0)),
            "median_return": float(cpcv_result.get("median_return", 0.0)),
            "median_pf": float(cpcv_result.get("median_pf", 1.0)),
            "median_expectancy": float(cpcv_result.get("median_expectancy", 0.0)),
            "median_calmar": float(cpcv_result.get("median_calmar", 0.0)),
        },
        "per_path_fold_stats": {
            "path_scores": list(cpcv_result.get("path_scores", [])),
            "path_sharpes": list(cpcv_result.get("path_sharpes", [])),
            "path_trades": list(cpcv_result.get("path_trades", [])),
            "path_winrates": list(cpcv_result.get("path_winrates", [])),
            "path_drawdowns": list(cpcv_result.get("path_drawdowns", [])),
            "path_returns": list(cpcv_result.get("path_returns", [])),
            "path_profit_factors": list(cpcv_result.get("path_pfs", [])),
            "path_expectancies": list(cpcv_result.get("path_expectancies", [])),
        },
        "regime_occupancy": dict(cpcv_result.get("regime_occupancy", {})),
        "trade_distribution": dict(cpcv_result.get("trade_distribution", {})),
        "lifecycle_telemetry": {
            "total_mr_leaks": int(cpcv_result.get("total_mr_leaks", 0)),
            "total_activation_events": int(cpcv_result.get("total_activation_events", 0)),
            "total_partial_close_events": int(cpcv_result.get("total_partial_close_events", 0)),
            "total_regime_forced_closes": int(cpcv_result.get("total_regime_forced_closes", 0)),
            "total_trail_updates": int(cpcv_result.get("total_trail_updates", 0)),
            "avg_activation_pnl_usd": float(cpcv_result.get("avg_activation_pnl_usd", 0.0)),
        },
        "objective_contribution_breakdown": {
            "component_values": dict(objective_breakdown.get("components", {})),
            "component_weights": dict(objective_breakdown.get("weights", {})),
            "base_score": float(objective_breakdown.get("base_score", 0.0)),
        },
        "final_score_and_penalties": {
            "final_score": float(final_score),
            "penalties": {k: float(v) for k, v in penalties.items()},
            "total_penalty": float(sum(float(v) for v in penalties.values())),
        },
    }


def write_cpcv_report(
    report: dict,
    tf: str,
    output_dir: str | Path = "reports",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cpcv_{tf.upper()}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def write_legacy_cpcv_report(
    report: dict,
    tf: str,
    broker: str,
    output_dir: str | Path = "reports",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cpcv_{tf.lower()}_{broker}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
