# Reporting and leaderboards (use valid_results to exclude low-trade runs)

os.makedirs("reports", exist_ok=True)
leg_c_lot_rule = "A_if_A_hits_first_else_B"
# Required strategy summary (plus useful extras) -- filtered
summary_cols = [
    "timeframe",
    "strategy_name",
    "exit_model",
    "profit_factor",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "expectancy",
    "win_rate",
    "trade_count",
    "parameter_set",
    "profit_per_trade",
    "parameter_stability_score",
    "robust_score",
]
strategy_summary = valid_results[summary_cols].copy()
strategy_summary["leg_c_lot_rule"] = leg_c_lot_rule
strategy_summary.to_csv("reports/strategy_summary.csv", index=False)

# Full output for diagnostics (unfiltered)
all_results.to_csv("reports/strategy_full_results.csv", index=False)

# New required leaderboard format -- filtered
leaderboard = (
    valid_results.sort_values(
        ["profit_per_trade", "robust_score", "profit_factor", "sharpe"],
        ascending=False,
    )
    .copy()
)
leaderboard = leaderboard.assign(leg_c_lot_rule=leg_c_lot_rule)
leaderboard_view = leaderboard[
    [
        "timeframe",
        "strategy_name",
        "exit_model",
        "profit_factor",
        "sharpe",
        "sortino",
        "calmar",
        "expectancy",
        "profit_per_trade",
        "trade_count",
        "max_drawdown",
        "parameter_set",
        "parameter_stability_score",
        "robust_score",
    ]
].copy()

leaderboard_view = leaderboard_view.rename(
    columns={
        "timeframe": "Timeframe",
        "strategy_name": "Strategy",
        "exit_model": "Exit Model",
        "profit_factor": "PF",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "calmar": "Calmar",
        "expectancy": "Expectancy",
        "profit_per_trade": "Profit Per Trade",
        "trade_count": "Trade Count",
        "max_drawdown": "Max DD",
        "leg_c_lot_rule": "Leg C Lot Rule",
    },
)

leaderboard_view.to_csv("reports/strategy_leaderboard.csv", index=False)

# Focus comparison requested in objective -- filtered
focus = valid_results[
    (valid_results["strategy_name"].isin(["trend_pullback", "volatility_expansion"]))
    & (
        valid_results["exit_model"].isin(
            ["partial_tp_plus_mr", "partial_tp_mr_time_stop"]
        )
    )
].copy()

focus = focus.sort_values(
    ["timeframe", "profit_per_trade", "robust_score", "profit_factor", "sharpe"],
    ascending=[True, False, False, False, False],
).reset_index(drop=True)
focus["leg_c_lot_rule"] = leg_c_lot_rule
focus.to_csv("reports/strategy_focus_partialtp_mr.csv", index=False)

print("Saved: reports/strategy_summary.csv")
print("Saved: reports/strategy_full_results.csv")
print("Saved: reports/strategy_leaderboard.csv")
print("Saved: reports/strategy_focus_partialtp_mr.csv")

print("\nTop Leaderboard Rows")
display(leaderboard_view.head(30))

print("\nTop Focus Rows (partial TP + MR variants)")
display(
    focus[
        [
            "timeframe",
            "strategy_name",
            "exit_model",
            "profit_factor",
            "sharpe",
            "sortino",
            "calmar",
            "expectancy",
            "profit_per_trade",
            "trade_count",
            "max_drawdown",
            "parameter_set",
            "parameter_stability_score",
            "robust_score",
            "leg_c_lot_rule",
        ]
    ].head(30)
)