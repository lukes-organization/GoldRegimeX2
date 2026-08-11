# ---------------------------------------------------------
# INGESTION LAYER: Import validated parameters from Strategy Tester
# FIX (2026-07-07): if no rows survive the strict DD cap for a timeframe,
# fall back to the row with the lowest max_drawdown available in the CSV
# (with a clear WARNING) instead of raising, so Explorer can proceed.
# The strict-fail path is preserved but only fires when BOTH the strict
# filter AND the lowest-DD fallback yield nothing, which only happens if
# the handoff CSV is truly empty for the timeframe.
# ---------------------------------------------------------
import json

def load_optimized_strategies(filepath="reports/strategy_winners_for_explorer.csv", max_dd=25.0):
    try:
        winners_df = pd.read_csv(filepath)
        print(f"Loaded {len(winners_df)} candidate strategies from bridge.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Bridge file {filepath} not found. Run Strategy_Tester.ipynb first.")

    best_params_by_tf = {}

    for tf in ["M15", "M5"]:
        tf_df = winners_df[winners_df["timeframe"] == tf].copy()
        if tf_df.empty:
            print(f"WARNING: {tf}: bridge CSV has zero rows for this timeframe.")
            best_params_by_tf[tf] = None
            continue

        strict = tf_df[tf_df["max_drawdown"] <= max_dd].copy()
        if not strict.empty:
            top_row = strict.sort_values("robust_score", ascending=False).iloc[0]
            source = "strict"
        else:
            # Fallback: lowest-DD row available for this timeframe.
            top_row = tf_df.sort_values("max_drawdown", ascending=True).iloc[0]
            source = "fallback_lowest_dd"
            print(
                f"WARNING: {tf}: no strategy in bridge CSV survives the {max_dd}% Max DD cap. "
                f"Falling back to lowest-DD available -> DD={top_row['max_drawdown']:.2f}%. "
                f"Re-run Strategy Tester so the handoff includes DD-survivors, "
                f"or lower the DD cap."
            )

        params_dict = json.loads(top_row["parameter_set"])
        best_params_by_tf[tf] = {
            "strategy_name": top_row["strategy_name"],
            "exit_model": top_row["exit_model"],
            "parameters": params_dict,
            "expected_pf": top_row["profit_factor"],
            "expected_dd": top_row["max_drawdown"],
            "selection_source": source,
        }
        tag = " (fallback)" if source != "strict" else ""
        print(
            f"Acquired {tf} Baseline{tag} -> "
            f"PF: {top_row['profit_factor']:.2f} | DD: {top_row['max_drawdown']:.2f}%"
        )

    return best_params_by_tf

ML_TARGET_PARAMS = load_optimized_strategies(max_dd=30.0)

if ML_TARGET_PARAMS.get(EXEC_TF) is None:
    raise RuntimeError(
        f"No {EXEC_TF} rows at all in reports/strategy_winners_for_explorer.csv. "
        f"Re-run Strategy Tester."
    )