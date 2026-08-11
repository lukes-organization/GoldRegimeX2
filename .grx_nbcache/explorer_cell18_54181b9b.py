# -----------------------------
# Dual timeframe validation (consolidated): M15 and M5 independently
# Phase 4:  Each TF uses its own threshold from pipeline
# Phase 8:  Uses pre-trained models from pipeline (no retraining)
# Phase 5:  Uses split_dataset for consistent splits
# Phase 7:  No globals() calls
# Phase 11: All data from pipeline container
# -----------------------------

def run_mode_v2(tf):
    "Run IS/OOS validation for a single timeframe using pipeline state."
    p = pipeline[tf]
    model = p.model
    thr = float(p.threshold)

    # Re-split from raw_all using centralised split_dataset
    train_df, oos_df, _split_time = split_dataset(p.raw_all, HOLDOUT_FRAC)

    # IS evaluation with pre-trained model
    trades_is, met_is = evaluate_ml_model(model, train_df, tf, thr)
    score_is = score_metrics(met_is)

    # OOS evaluation with SAME pre-trained model
    trades_oos, met_oos = evaluate_ml_model(model, oos_df, tf, thr)
    score_oos = score_metrics(met_oos)

    rel_drop = float((score_is - score_oos) / (abs(score_is) + 1e-9))

    rows = [
        {
            "timeframe_mode": "%s_only" % tf,
            "scenario": "IS",
            "xgb_threshold": thr,
            "score": float(score_is),
            "win_rate": float(met_is.get("win_rate", 0)),
            "profit_factor": float(met_is.get("profit_factor", 0)),
            "max_drawdown_pct": float(met_is.get("max_drawdown_pct", 0)),
            "n_trades": int(met_is.get("trade_count", met_is.get("n_trades", 0))),
            "ending_balance_cents": float(met_is.get("ending_balance_cents", 0)),
            "net_return_pct": float(met_is.get("net_return_pct", 0)),
            "net_profit_cents": float(met_is.get("net_profit", 0)),
            "profit_per_trade_cents": float(met_is.get("profit_per_trade", 0)),
            "profit_per_trade_usd": float(met_is.get("profit_per_trade", 0)) / 100.0,
            "relative_score_drop_is_to_oos": rel_drop,
            "split_time": _split_time,
            "exec_rows": int(len(train_df)),
        },
        {
            "timeframe_mode": "%s_only" % tf,
            "scenario": "OOS",
            "xgb_threshold": thr,
            "score": float(score_oos),
            "win_rate": float(met_oos.get("win_rate", 0)),
            "profit_factor": float(met_oos.get("profit_factor", 0)),
            "max_drawdown_pct": float(met_oos.get("max_drawdown_pct", 0)),
            "n_trades": int(met_oos.get("trade_count", met_oos.get("n_trades", 0))),
            "ending_balance_cents": float(met_oos.get("ending_balance_cents", 0)),
            "net_return_pct": float(met_oos.get("net_return_pct", 0)),
            "net_profit_cents": float(met_oos.get("net_profit", 0)),
            "profit_per_trade_cents": float(met_oos.get("profit_per_trade", 0)),
            "profit_per_trade_usd": float(met_oos.get("profit_per_trade", 0)) / 100.0,
            "relative_score_drop_is_to_oos": rel_drop,
            "split_time": _split_time,
            "exec_rows": int(len(oos_df)),
        },
    ]
    return rows, trades_is, trades_oos


all_rows = []
trades_by_mode = {}

for tf in TIMEFRAMES:
    thr = pipeline[tf].threshold
    print("Running %s validation with threshold=%s" % (tf, thr))
    rows, t_is, t_oos = run_mode_v2(tf)
    all_rows.extend(rows)
    trades_by_mode["%s_only_IS" % tf] = t_is
    trades_by_mode["%s_only_OOS" % tf] = t_oos

dual_tf_summary = pd.DataFrame(all_rows)
dual_tf_summary["scenario_order"] = dual_tf_summary["scenario"].map({"IS": 0, "OOS": 1})
dual_tf_summary = dual_tf_summary.sort_values(["timeframe_mode", "scenario_order"]).drop(columns=["scenario_order"])

print("\nDual timeframe summary:")
display(dual_tf_summary)