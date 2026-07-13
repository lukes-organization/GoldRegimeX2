# Strategy Tester Audit

* File: `pipeline_verification_bundle\Strategy_Tester_fixed.ipynb`
* Total cells: 26

## Stage → Cell map (execution order proxy)

* **Candidate Generation**: 5, 7, 9, 15, 16, 18, 23, 24
* **Feature Engineering**: 3, 15
* **Triple Barrier Method**: 14, 15, 17, 23
* **Session Filtering**: 1, 3, 5, 7, 20, 21, 22, 23, 24
* **HMM**: 3, 5, 6, 10, 11, 14, 15, 17, 23
* **XGBoost**: 14, 15, 23
* **Probability threshold**: _not found_
* **Risk Manager**: 14, 15
* **Execution**: 6, 7, 11, 12, 14, 15, 16, 18

## Per-cell summary

| Cell | Type | Stages | Defined | First line |
| ---: | ---- | ------ | ------- | ---------- |
| 0 | markdown |  |  | `# Strategy Tester (No TBM / CPCV / HMM / XGBoost)` |
| 1 | code | Session Filtering | N_JOBS, QUICK_MODE, RESEARCH_YEARS, M5_PATH, M15_PATH, TIMEFRAMES… | `# Imports + Config` |
| 2 | code |  | ENABLE_ENTRY_CAP, ENTRY_CAP_SEED_BASE | `# Optional exploratory speed cap (applies to QUICK_MODE True/False)` |
| 3 | code | Feature Engineering, Session Filtering, HMM | resolve_path, _normalize_ohlc, read_xau_raw, load_recent_years, enforce_recent_window, ema… | `# Data loading, strict 5-year reduction, indicators, and rule-based regimes` |
| 4 | code |  |  | `# Legacy cell disabled intentionally.` |
| 5 | code | Candidate Generation, Session Filtering, HMM | BaseStrategy, session_col_from_value, TrendPullbackStrategy, VolatilityExpansionStrategy, STRATEGIES, generate_routed_signals | `# Cell 5: Strategy Definitions & Regime-Based Signal Router (With Macro Filter)` |
| 6 | code | HMM, Execution | compute_metrics, _run_backtest_numba, _safe_float, run_backtest | `# Cell 6: Numba Backtest Engine with Early Eject, Pyramiding (Scale-In), and Asymmetric Guard` |
| 7 | code | Candidate Generation, Session Filtering, Execution | _is_numeric_grid, build_step_map, add_parameter_stability_score, _cap_entry_combos, get_exit_grid_for_mode, run_group | `# Vectorized Plateau Stability Engine & Group Experiment Runner` |
| 8 | code |  | group_tasks, _run_task, all_results, valid_results, best_rows | `# Concurrent execution across timeframe + strategy + exit model` |
| 9 | code | Candidate Generation | leg_c_lot_rule, summary_cols, strategy_summary, leaderboard, leaderboard, leaderboard_view… | `# Reporting and leaderboards (use valid_results to exclude low-trade runs)` |
| 10 | code | HMM | leg_c_lot_rule, HANDOFF_MAX_DD_PCT, handoff_cols, top_n, top_by_profit, dd_survivors… | `# Export top robust candidates for downstream wiring into GoldRegimeX_Explorer` |
| 11 | code | HMM, Execution | generate_regime_report | `# Regime Performance Attribution Analytics Report` |
| 12 | code | Execution | MAX_DD_PCT, cols, results_df, filtered | `MAX_DD_PCT = 25.0  # adjust cap (percent)` |
| 13 | markdown |  |  | `## Traceability Layer (Added)` |
| 14 | code | Triple Barrier Method, HMM, XGBoost, Risk Manager, Execution | CandidateTrade, make_candidate_id, parameter_set_id, RejectionReason | `# ============================================================` |
| 15 | code | Candidate Generation, Feature Engineering, Triple Barrier Method, HMM, XGBoost, Risk Manager, Execution | PipelineProfiler, PIPELINE_STAGES, profiler | `# ============================================================` |
| 16 | code | Candidate Generation, Execution | compute_entry_stop_target, build_position_active_mask, build_candidate_portfolio | `# ============================================================` |
| 17 | code | Triple Barrier Method, HMM | compute_ml_readiness_score | `# ============================================================` |
| 18 | code | Candidate Generation, Execution | candidate_portfolios, readiness_rows, flat_candidate_rows, candidate_trades_export, ml_readiness_df | `# ============================================================` |
| 19 | markdown |  |  | `## Pipeline Verification & Certification (Added)` |
| 20 | code | Session Filtering | VERIFY_PIPELINE | `# ============================================================` |
| 21 | code | Session Filtering |  | `# ============================================================` |
| 22 | code | Session Filtering |  | `# ============================================================` |
| 23 | code | Candidate Generation, Triple Barrier Method, Session Filtering, HMM, XGBoost |  | `# ============================================================` |
| 24 | code | Candidate Generation, Session Filtering |  | `# ============================================================` |
| 25 | code |  |  | `# ============================================================` |

## Inputs / Outputs / Shared / Exported (heuristic)

### Imports (module inputs)

* `dataclasses.asdict`
* `dataclasses.dataclass`
* `enum.Enum`
* `hashlib`
* `itertools`
* `json`
* `math`
* `numba.njit`
* `numpy`
* `os`
* `pandas`
* `pathlib.Path`
* `sys`
* `time`
* `zlib`

### Top-level definitions (exported objects / shared variables)

* `ADX_GRID`
* `ATR_EXPANSION_GRID`
* `ATR_STOP_GRID`
* `ATR_TARGET_GRID`
* `BREAKOUT_BUFFER_GRID`
* `BREAKOUT_LOOKBACK_GRID`
* `BaseStrategy`
* `COMMISSION_CENTS_PER_TRADE`
* `CONFIRMATION_GRID`
* `CandidateTrade`
* `ENABLE_ENTRY_CAP`
* `ENTRY_ATR_TARGET_GRID`
* `ENTRY_CAP_SEED_BASE`
* `EXIT_MODELS`
* `FEATURES_BY_TF`
* `HANDOFF_MAX_DD_PCT`
* `INITIAL_BALANCE_CENTS`
* `LEG_A_ATR_TARGET_GRID`
* `LEG_C_ATR_STOP`
* `LEG_C_ATR_TARGET`
* `M15_PATH`
* `M5_ADX_GRID`
* `M5_ATR_STOP_GRID`
* `M5_CONFIRMATION_GRID`
* `M5_ENTRY_TARGET_GRID`
* `M5_LEG_A_TARGET_GRID`
* `M5_PATH`
* `M5_PULLBACK_RSI_GRID`
* `MAX_DD_PCT`
* `N_JOBS`
* `PIPELINE_STAGES`
* `PIP_SIZE_PRICE`
* `PIP_VALUE_CENTS_PER_1LOT`
* `POSITION_A`
* `POSITION_B`
* `PULLBACK_RSI_GRID`
* `PipelineProfiler`
* `QUICK_MODE`
* `RESEARCH_YEARS`
* `RejectionReason`
* `SESSION_FILTER_VALUES`
* `SLIPPAGE_PIPS`
* `SPREAD_CAP_POINTS`
* `STRATEGIES`
* `TIMEFRAMES`
* `TIME_STOP_GRID_BY_TF`
* `TRAIL_MULT_GRID`
* `TrendPullbackStrategy`
* `VERIFY_PIPELINE`
* `VolatilityExpansionStrategy`
* `_cap_entry_combos`
* `_is_numeric_grid`
* `_normalize_ohlc`
* `_run_backtest_numba`
* `_run_task`
* `_safe_float`
* `add_parameter_stability_score`
* `add_session_features`
* `adx`
* `all_results`
* `atr`
* `best_rows`
* `build_candidate_portfolio`
* `build_features`
* `build_position_active_mask`
* `build_step_map`
* `candidate_portfolios`
* `candidate_trades_export`
* `cols`
* `compute_entry_stop_target`
* `compute_metrics`
* `compute_ml_readiness_score`
* `dd_survivors`
* `ema`
* `enforce_recent_window`
* `filtered`
* `flat_candidate_rows`
* `focus`
* `generate_regime_report`
* `generate_routed_signals`
* `get_exit_grid_for_mode`
* `group_tasks`
* `handoff`
* `handoff_cols`
* `leaderboard`
* `leaderboard_view`
* `leg_c_lot_rule`
* `load_recent_years`
* `m15_raw`
* `m15_raw_full`
* `m5_raw`
* `m5_raw_full`
* `make_candidate_id`
* `ml_readiness_df`
* `parameter_set_id`
* `per_tf_k`
* `per_tf_survivors`
* `profiler`
* `read_xau_raw`
* `readiness_rows`
* `resolve_path`
* `results_df`
* `rsi`
* `run_backtest`
* `run_group`
* `session_col_from_value`
* `strategy_summary`
* `summary_cols`
* `top_by_profit`
* `top_n`
* `true_range`
* `valid_results`
