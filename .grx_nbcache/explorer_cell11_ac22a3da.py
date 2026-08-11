# -----------------------------
# CPCV combo evaluator for M15 Trend + M5 Execution strategy
# -----------------------------
import math
import time
import warnings

def _empty_combo_result(xgb_threshold, exec_tf):
    return {
        "timeframe": exec_tf,
        "xgb_threshold": float(xgb_threshold),
        "mean_sharpe": 0.0,
        "mean_sharpe_raw": 0.0,
        "variance_sharpe": 0.0,
        "stability_adjusted_sharpe": 0.0,
        "turnover_penalty": 0.0,
        "mean_trades_per_100": 0.0,
        "n_paths": 0,
        "median_trades": 0,
    }

# Timeframe-scoped cache (Phase 7: no global mutable state)
_ML_FOLD_CACHE = {tf: {} for tf in TIMEFRAMES}

def _get_or_compute_ml_folds(exec_df, n_blocks, k_val_blocks, embargo_bars, exec_tf):
    tf_cache = _ML_FOLD_CACHE.get(exec_tf, {})

    cache_key = f"{exec_tf}_{id(exec_df)}_{n_blocks}_{k_val_blocks}"
    if cache_key in tf_cache:
        return tf_cache[cache_key]

    print(f"\n[!] Pre-computing Features and ML Models for {exec_tf}...")

    t_start = time.time()
    d_exec = exec_df.sort_index().copy()

    base_params = dict(ML_TARGET_PARAMS[exec_tf]["parameters"])
    base_params["exit_model"] = ML_TARGET_PARAMS[exec_tf]["exit_model"]

    feat = build_features(d_exec, exec_tf)
    event_end_pos = feat["event_end_pos"].to_numpy(dtype=np.int32)
    hmm_cols = hmm_feature_columns(feat)

    splitter = CPCVPurgedEmbargo(
        n_blocks=int(n_blocks),
        k_val_blocks=int(k_val_blocks),
        embargo_bars=int(embargo_bars),
    )

    folds_data = []
    splits = list(splitter.split(len(feat), event_end_pos))

    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        if len(train_idx) < 200 or len(val_idx) < 250:
            continue

        print(f" -> Training ML Fold {fold_idx}/{len(splits)}...")
        train_features = feat.iloc[train_idx]
        val_features = feat.iloc[val_idx]
        train_labels = train_features["tb_label"]

        model = HMMXGBComposite(random_state=RANDOM_STATE)
        model.fit(train_features, train_labels, hmm_features=hmm_cols)
        ml_probs = model.predict_proba_raw(val_features)

        folds_data.append((val_features, ml_probs))

    print(f"Pre-computation for {exec_tf} complete in {(time.time()-t_start)/60:.1f} minutes.\n")

    cache_data = {
        "folds_data": folds_data,
        "base_params": base_params
    }
    _ML_FOLD_CACHE[exec_tf][cache_key] = cache_data
    return cache_data

def evaluate_combo_cpcv(
    exec_df: pd.DataFrame,
    xgb_threshold: float,
    n_blocks: int,
    k_val_blocks: int,
    embargo_bars: int,
    exec_tf: str
) -> dict:
    warnings.filterwarnings("ignore")

    if not isinstance(ML_TARGET_PARAMS, dict) or ML_TARGET_PARAMS.get(exec_tf) is None:
        return _empty_combo_result(xgb_threshold, exec_tf)

    cache_data = _get_or_compute_ml_folds(exec_df, n_blocks, k_val_blocks, embargo_bars, exec_tf)
    folds_data = cache_data["folds_data"]
    base_params = cache_data["base_params"]

    path_scores = []
    path_trades = []
    path_t100 = []

    for val_features, ml_probs in folds_data:
        _, met = run_ml_filtered_backtest(
            timeframe=exec_tf,
            df=val_features,
            ml_probs=ml_probs,
            base_params=base_params,
            xgb_threshold=float(xgb_threshold),
        )

        path_scores.append(float(met.get("sharpe", 0.0)))
        n_trades = int(met.get("trade_count", 0))
        path_trades.append(n_trades)
        path_t100.append(float((n_trades / max(len(val_features), 1)) * 100.0))

    if len(path_scores) == 0:
        return _empty_combo_result(xgb_threshold, exec_tf)

    arr = np.array(path_scores, dtype=float)
    mean_raw = float(np.mean(arr))
    var_score = float(np.var(arr))
    mean_t100 = float(np.mean(np.array(path_t100, dtype=float)))

    limit = float(4.0)
    lam = float(3.0)
    excess = max(0.0, mean_t100 - limit)
    turnover_penalty = float(lam * excess)

    mean_score = float(mean_raw - turnover_penalty)
    stab = float(mean_score / (math.sqrt(max(var_score, 1e-12)) + 1e-6))

    return {
        "timeframe": exec_tf,
        "xgb_threshold": float(xgb_threshold),
        "mean_sharpe": float(mean_score),
        "mean_sharpe_raw": float(mean_raw),
        "variance_sharpe": float(var_score),
        "stability_adjusted_sharpe": float(stab),
        "turnover_penalty": float(turnover_penalty),
        "mean_trades_per_100": float(mean_t100),
        "n_paths": int(len(path_scores)),
        "median_trades": int(np.median(np.array(path_trades, dtype=int))),
    }

def run_grid_parallel(
    exec_df: pd.DataFrame,
    grid: list[tuple[float]],
    n_blocks: int,
    k_val_blocks: int,
    embargo_bars: int,
    exec_tf: str,
    n_jobs: int = 1,
) -> pd.DataFrame:
    t0 = time.time()
    _get_or_compute_ml_folds(exec_df, n_blocks, k_val_blocks, embargo_bars, exec_tf)

    out_local = []
    for i, g in enumerate(grid, 1):
        xgb_threshold = float(g[0])
        res = evaluate_combo_cpcv(exec_df, xgb_threshold, n_blocks, k_val_blocks, embargo_bars, exec_tf)
        out_local.append(res)

    df_out = pd.DataFrame(out_local)
    elapsed = time.time() - t0
    print(f"Grid sweep for {exec_tf} complete in {elapsed:.2f} seconds")
    return df_out