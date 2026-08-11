# -----------------------------
# Coarse pass
# -----------------------------
coarse_grid = build_coarse_grid()
coarse_results_dict = {}

for tf in ["M15", "M5"]:
    print(f"\n--- COARSE PASS ({tf}) ---")
    active_train_df = m5_train if tf == "M5" else m15_train

    res = run_grid_parallel(
        exec_df=active_train_df,
        grid=coarse_grid,
        n_blocks=4,
        k_val_blocks=1,
        embargo_bars=12,
        exec_tf=tf,
        n_jobs=N_JOBS
    )

    coarse_results_dict[tf] = res
    print(f"\n{tf} Coarse Results Top 3:")
    display(res.sort_values(["stability_adjusted_sharpe", "mean_sharpe"], ascending=False).head(3))

# Combine all results
coarse_results = pd.concat(coarse_results_dict.values(), ignore_index=True)