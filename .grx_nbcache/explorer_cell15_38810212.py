# -----------------------------
# Fine pass
# -----------------------------
fine_results_dict = {}

for tf in ["M15", "M5"]:
    print(f"\n--- FINE PASS ({tf}) ---")
    active_train_df = m5_train if tf == "M5" else m15_train
    tf_coarse = coarse_results[coarse_results["timeframe"] == tf]

    refined_grid = build_refined_grid_from_top(tf_coarse, top_k=3, step=0.02)

    res = run_grid_parallel(
        exec_df=active_train_df,
        grid=refined_grid,
        n_blocks=4,
        k_val_blocks=1,
        embargo_bars=12,
        exec_tf=tf,
        n_jobs=N_JOBS
    )

    fine_results_dict[tf] = res
    print(f"\n{tf} Fine Results Top 3:")
    display(res.sort_values(["stability_adjusted_sharpe", "mean_sharpe"], ascending=False).head(3))

# Combine all results
fine_results = pd.concat(fine_results_dict.values(), ignore_index=True)