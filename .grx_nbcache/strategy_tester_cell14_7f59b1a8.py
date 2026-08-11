MAX_DD_PCT = 30.0  # adjust cap (percent)

cols = [
    "timeframe", "strategy_name", "exit_model",
    "trade_count", "net_profit", "profit_per_trade", "profit_per_trade_usd",
    "max_drawdown", "robust_score", "profit_factor", "sharpe"
]

results_df = globals().get("all_results", None)
if results_df is None:
    results_df = globals().get("valid_results", None)
if results_df is None:
    raise NameError("all_results and valid_results are not defined. Run the backtest cell first.")

filtered = results_df[
    results_df["max_drawdown"] <= MAX_DD_PCT
].copy()
filtered["profit_per_trade_usd"] = filtered["profit_per_trade"] / 100.0

display(
    filtered[cols]
    .sort_values(["timeframe", "robust_score"], ascending=[True, False])
    .reset_index(drop=True)
)

print("Remaining rows:", len(filtered))