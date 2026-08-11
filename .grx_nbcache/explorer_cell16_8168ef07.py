# -----------------------------
# Sanity checks
# -----------------------------

print("Fine unique n_paths:", sorted(fine_results["n_paths"].dropna().unique().tolist()))
print("Fine mean trades/100 (top 10):", fine_results["mean_trades_per_100"].head(10).round(3).tolist())
print("Fine turnover penalty (top 10):", fine_results["turnover_penalty"].head(10).round(3).tolist())

print("\nTop 10 with raw vs penalized:")
display(
    fine_results[
        [
            "xgb_threshold",
            "mean_sharpe_raw",
            "turnover_penalty",
            "mean_sharpe",
            "stability_adjusted_sharpe",
            "mean_trades_per_100",
        ]
    ].head(10)
)