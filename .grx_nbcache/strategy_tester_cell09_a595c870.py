# Vectorized Plateau Stability Engine & Group Experiment Runner

import zlib

def _is_numeric_grid(values: list) -> bool:
    return all(isinstance(v, (int, float, np.integer, np.floating)) for v in values)

def build_step_map(param_grid: dict) -> dict:
    step_map = {}
    for key, vals in param_grid.items():
        uniq = sorted(set(vals), key=lambda x: str(x))
        if _is_numeric_grid(uniq) and len(uniq) > 1:
            diffs = np.diff(np.array(uniq, dtype=float))
            diffs = diffs[diffs > 0]
            step_map[key] = float(np.min(diffs)) if len(diffs) > 0 else 1.0
        else:
            step_map[key] = 1.0 if _is_numeric_grid(uniq) else None
    return step_map

def add_parameter_stability_score(
    results_df: pd.DataFrame,
    param_cols: list[str],
    step_map: dict,
    perf_col: str = "profit_per_trade",
    block_size: int = 128,  # lower to reduce memory pressure
) -> pd.DataFrame:
    if results_df.empty:
        out = results_df.copy()
        out["parameter_stability_score"] = np.nan
        return out

    out = results_df.copy()
    M = len(out)

    numeric_cols = [c for c in param_cols if step_map.get(c, None) is not None]
    cat_cols = [c for c in param_cols if step_map.get(c, None) is None]

    num_mat = out[numeric_cols].astype(float).to_numpy() if numeric_cols else np.empty((M, 0), dtype=float)
    step_sizes = np.array([float(step_map[c]) for c in numeric_cols], dtype=float) if numeric_cols else np.empty((0,), dtype=float)
    cat_mat = np.column_stack([pd.factorize(out[c].astype(str))[0] for c in cat_cols]).astype(np.int32) if cat_cols else np.empty((M, 0), dtype=np.int32)

    perf = out[perf_col].astype(float).to_numpy()
    perf2 = perf * perf
    scores = np.empty(M, dtype=float)

    for start in range(0, M, block_size):
        end = min(M, start + block_size)

        if num_mat.shape[1] > 0:
            diffs = np.abs(num_mat[start:end, None, :] - num_mat[None, :, :])
            num_mask = np.all(diffs <= step_sizes, axis=2)
        else:
            num_mask = np.ones((end - start, M), dtype=bool)

        if cat_mat.shape[1] > 0:
            cat_mask = np.all(cat_mat[start:end, None, :] == cat_mat[None, :, :], axis=2)
        else:
            cat_mask = np.ones((end - start, M), dtype=bool)

        mask = num_mask & cat_mask
        counts = mask.sum(axis=1).astype(np.float64)
        sums = mask @ perf
        sums2 = mask @ perf2

        means = np.where(counts > 0, sums / counts, -1e9)
        vars_ = np.maximum(np.where(counts > 0, sums2 / counts - means * means, 0.0), 0.0)
        scores[start:end] = means - np.sqrt(vars_)

    out["parameter_stability_score"] = scores
    return out

def _cap_entry_combos(
    entry_combos: list[dict],
    timeframe: str,
    strategy_name: str,
    exit_model: str,
) -> tuple[list[dict], int, bool]:
    enable = bool(globals().get("ENABLE_ENTRY_CAP", False))
    cap_map = globals().get("ENTRY_CAP_BY_TF", {})
    cap_val = cap_map.get(timeframe, None) if isinstance(cap_map, dict) else None

    original_n = len(entry_combos)
    if (not enable) or (cap_val is None):
        return entry_combos, original_n, False

    cap_n = int(cap_val)
    if cap_n <= 0 or original_n <= cap_n:
        return entry_combos, original_n, False

    seed_base = int(globals().get("ENTRY_CAP_SEED_BASE", 42))
    seed_key = f"{seed_base}|{timeframe}|{strategy_name}|{exit_model}"
    seed = zlib.crc32(seed_key.encode("utf-8")) & 0xFFFFFFFF

    rng = np.random.default_rng(seed)
    picked_idx = np.sort(rng.choice(original_n, size=cap_n, replace=False))
    capped = [entry_combos[i] for i in picked_idx.tolist()]
    return capped, original_n, True

def get_exit_grid_for_mode(timeframe: str, exit_model: str):
    ts_grid = TIME_STOP_GRID_BY_TF[timeframe]
    tr_grid = TRAIL_MULT_GRID

    if exit_model == "fixed_tp":
        cfg = [{"leg_a_atr_target": p} for p in LEG_A_ATR_TARGET_GRID]
        grid_map = {"leg_a_atr_target": LEG_A_ATR_TARGET_GRID}
    elif exit_model == "mr_exit":
        cfg = [{"leg_a_atr_target": None}]
        grid_map = {}
    elif exit_model == "fixed_tp_plus_mr":
        cfg = [{"leg_a_atr_target": p} for p in LEG_A_ATR_TARGET_GRID]
        grid_map = {"leg_a_atr_target": LEG_A_ATR_TARGET_GRID}
    elif exit_model == "partial_tp_plus_mr":
        cfg = [{"leg_a_atr_target": p} for p in LEG_A_ATR_TARGET_GRID]
        grid_map = {"leg_a_atr_target": LEG_A_ATR_TARGET_GRID}
    elif exit_model == "partial_tp_mr_time_stop":
        cfg = [
            {
                "leg_a_atr_target": p,
                "time_stop_minutes": t,
                "trail_mult": m,
            }
            for p, t, m in itertools.product(
                LEG_A_ATR_TARGET_GRID,
                ts_grid,
                tr_grid,
            )
        ]
        grid_map = {
            "leg_a_atr_target": LEG_A_ATR_TARGET_GRID,
            "time_stop_minutes": ts_grid,
            "trail_mult": tr_grid,
        }
    else:
        raise ValueError(f"Unsupported exit model: {exit_model}")

    return cfg, grid_map

def run_group(timeframe: str, strategy_name: str, exit_model: str) -> pd.DataFrame:
    df = FEATURES_BY_TF[timeframe]
    strategy = STRATEGIES[strategy_name]

    # Per-TF grid override: rebind the per-strategy param_grid and module-level exit grids
    # for M5 only, so M15 runs are byte-identical to before this change.
    _restore = False
    _orig_grid = dict(strategy.param_grid)
    _orig_legA = list(LEG_A_ATR_TARGET_GRID)
    _orig_entry = list(ENTRY_ATR_TARGET_GRID)
    _orig_atrT = list(ATR_TARGET_GRID)
    if timeframe == "M5":
        if strategy_name == "trend_pullback":
            strategy.param_grid = {**_orig_grid,
                "adx_threshold":    M5_ADX_GRID,
                "pullback_rsi":     M5_PULLBACK_RSI_GRID,
                "confirmation_bars":M5_CONFIRMATION_GRID,
                "atr_stop":         M5_ATR_STOP_GRID,
                "atr_target":       M5_ENTRY_TARGET_GRID,
                "session_filter":   _orig_grid.get("session_filter", SESSION_FILTER_VALUES),
            }
        elif strategy_name == "volatility_expansion":
            strategy.param_grid = {**_orig_grid,
                "atr_stop":         M5_ATR_STOP_GRID,
                "atr_target":       M5_ENTRY_TARGET_GRID,
                "session_filter":   _orig_grid.get("session_filter", SESSION_FILTER_VALUES),
            }
        LEG_A_ATR_TARGET_GRID[:] = M5_LEG_A_TARGET_GRID
        ENTRY_ATR_TARGET_GRID[:] = M5_ENTRY_TARGET_GRID
        ATR_TARGET_GRID[:]       = M5_LEG_A_TARGET_GRID
        _restore = True

    t0 = time.time()
    try:
        entry_combos_full = list(strategy.iter_param_dicts())
        entry_combos, full_n, was_capped = _cap_entry_combos(
            entry_combos_full, timeframe, strategy_name, exit_model
        )

        exit_cfgs, exit_grid_map = get_exit_grid_for_mode(timeframe, exit_model)

        rows = []
        total = len(entry_combos) * len(exit_cfgs)
        done = 0

        if was_capped:
            print(
                f"[{timeframe}][{strategy_name}][{exit_model}] entry combos capped: "
                f"{len(entry_combos)}/{full_n}"
            )
        else:
            print(
                f"[{timeframe}][{strategy_name}][{exit_model}] entry combos used: "
                f"{len(entry_combos)}/{full_n}"
            )

        for i, entry_params in enumerate(entry_combos, 1):
            signals = generate_routed_signals(df, entry_params, strategy_name)

            for exit_cfg in exit_cfgs:
                _, metrics = run_backtest(
                    timeframe=timeframe,
                    df=df,
                    signals=signals,
                    entry_params=entry_params,
                    exit_model=exit_model,
                    exit_params=exit_cfg,
                )

                combined_params = {
                    **entry_params,
                    **{k: v for k, v in exit_cfg.items() if v is not None},
                }

                row = {
                    "timeframe": timeframe,
                    "strategy_name": strategy_name,
                    "exit_model": exit_model,
                    "parameter_set": json.dumps(combined_params, sort_keys=True),
                    **metrics,
                }
                for k, v in combined_params.items():
                    row[k] = v
                rows.append(row)

                done += 1

            if i % 50 == 0 or i == len(entry_combos):
                print(f"[{timeframe}][{strategy_name}][{exit_model}] {done}/{total}")

        res = pd.DataFrame(rows)

        param_cols = list(strategy.param_cols) + list(exit_grid_map.keys())
        full_grid_map = {**strategy.param_grid, **exit_grid_map}
        step_map = build_step_map(full_grid_map)

        res = add_parameter_stability_score(
            res, param_cols=param_cols, step_map=step_map, perf_col="profit_per_trade"
        )

        res["robust_score"] = (
            0.45 * res["parameter_stability_score"].astype(float)
            + 0.35 * res["profit_per_trade"].astype(float)
            + 0.20 * res["profit_factor"].astype(float)
        )

        res = res.sort_values(
            ["robust_score", "profit_per_trade", "profit_factor", "sharpe"],
            ascending=False,
        ).reset_index(drop=True)

        return res
    finally:
        if _restore:
            strategy.param_grid = _orig_grid
            LEG_A_ATR_TARGET_GRID[:] = _orig_legA
            ENTRY_ATR_TARGET_GRID[:] = _orig_entry
            ATR_TARGET_GRID[:]       = _orig_atrT
        elapsed = time.time() - t0
        print(f"[{timeframe}][{strategy_name}][{exit_model}] done in {elapsed/60:.2f} minutes")