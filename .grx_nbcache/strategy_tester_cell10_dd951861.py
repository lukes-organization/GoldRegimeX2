# Concurrent execution across timeframe + strategy + exit model

group_tasks = [
    (tf, strategy_name, exit_model)
    for tf in TIMEFRAMES
    for strategy_name in STRATEGIES.keys()
    for exit_model in EXIT_MODELS
]

print("Total groups:", len(group_tasks))
for t in group_tasks:
    print("Task:", t)

def _run_task(task):
    tf, strategy_name, exit_model = task
    return run_group(tf, strategy_name, exit_model)

if JOBLIB_OK:
    try:
        nj = min(max(1, N_JOBS), len(group_tasks))
        print(f"Running parallel with joblib loky backend... n_jobs={nj}")
        group_results = Parallel(n_jobs=nj, backend="loky", verbose=10, batch_size=1)(
            delayed(_run_task)(task) for task in group_tasks
        )
    except Exception as e:
        print(f"[warn] loky failed: {type(e).__name__}: {e}")
        try:
            nj = min(max(1, N_JOBS), len(group_tasks))
            print(f"Falling back to threading... n_jobs={nj}")
            group_results = Parallel(n_jobs=nj, backend="threading", verbose=10, batch_size=1)(
                delayed(_run_task)(task) for task in group_tasks
            )
        except Exception as e2:
            print(f"[warn] threading failed: {type(e2).__name__}: {e2}")
            print("Falling back to sequential execution.")
            group_results = [_run_task(task) for task in group_tasks]
else:
    print("joblib unavailable. Running sequentially.")
    group_results = [_run_task(task) for task in group_tasks]

all_results = pd.concat(group_results, axis=0, ignore_index=True)

# Apply statistical significance filter before sorting
valid_results = all_results[all_results["trade_count"] >= 200]

best_rows = (
    valid_results.sort_values(
        ["robust_score", "profit_per_trade", "profit_factor", "sharpe"],
        ascending=False,
    )
    .groupby(["timeframe", "strategy_name", "exit_model"], as_index=False)
    .head(1)
    .reset_index(drop=True)
)

print("All results shape:", all_results.shape)
print("Best rows shape:", best_rows.shape)
display(best_rows.head(20))