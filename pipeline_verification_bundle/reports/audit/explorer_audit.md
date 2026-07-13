# Explorer Audit

* File: `pipeline_verification_bundle\GoldRegimeX_Explorer_fixed.ipynb`
* Total cells: 58

## Stage → Cell map (execution order proxy)

* **Candidate Generation**: 8, 19, 26, 27, 28, 29, 30, 33, 34, 35, 40, 41, 42, 43, 44, 55, 56
* **Feature Engineering**: 5, 9, 15, 19, 20, 27, 29, 33, 39, 43, 49, 51, 53
* **Triple Barrier Method**: 5, 20, 28, 32, 33, 39, 40, 43, 46, 49, 55
* **Session Filtering**: 5, 6, 8, 19, 20, 21, 22, 27, 28, 30, 46, 47, 48, 52, 55, 56
* **HMM**: 1, 2, 5, 7, 8, 11, 15, 17, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 32, 33, 35, 37, 39, 40, 42, 43, 44, 46, 50, 55
* **XGBoost**: 1, 7, 24, 28, 29, 32, 33, 38, 39, 40, 43, 44, 46, 50, 51
* **Probability threshold**: 17, 24, 29, 36, 42, 43
* **Risk Manager**: 1, 19, 22, 28, 32, 33, 43
* **Execution**: 1, 8, 9, 15, 19, 21, 22, 24, 27, 28, 29, 30, 33, 35, 41, 42, 43, 53, 55

## Per-cell summary

| Cell | Type | Stages | Defined | First line |
| ---: | ---- | ------ | ------- | ---------- |
| 0 | markdown |  |  | `# GoldRegime X - Iteration 2: Fast Sensitivity Plateau Lab (M5/M15)` |
| 1 | code | HMM, XGBoost, Risk Manager, Execution | _here, _project_root, _project_root_str, _prev_py_path | `import os` |
| 2 | code | HMM | RANDOM_STATE, EXEC_TF, TREND_TF, INITIAL_BALANCE_CENTS, SPREAD_CAP_POINTS, STOP_LOSS_PIPS… | `# -----------------------------` |
| 3 | code |  | load_optimized_strategies, ML_TARGET_PARAMS | `# ---------------------------------------------------------` |
| 4 | code |  | _normalize_ohlcv, read_xau_raw, read_mt4_csv, read_master_close, load_panel | `# -----------------------------` |
| 5 | code | Feature Engineering, Triple Barrier Method, Session Filtering, HMM | ema, true_range, atr, rsi, adx, synth_vix_zscore… | `# ---------------------------------------------------------` |
| 6 | code | Session Filtering | CPCVPurgedEmbargo | `# -----------------------------` |
| 7 | code | HMM, XGBoost | HMMXGBComposite | `# -----------------------------` |
| 8 | code | Candidate Generation, Session Filtering, HMM, Execution | POSITION_A, POSITION_B, LEG_C_ATR_STOP, LEG_C_ATR_TARGET, PIP_SIZE_PRICE, PIP_VALUE_CENTS_PER_1LOT… | `# -----------------------------` |
| 9 | code | Feature Engineering, Execution | _empty_combo_result, _ML_FOLD_CACHE, _get_or_compute_ml_folds, evaluate_combo_cpcv, run_grid_parallel | `# -----------------------------` |
| 10 | code |  | build_coarse_grid, build_refined_grid_from_top, plot_plateau_heatmaps, select_plateau_center | `# -----------------------------` |
| 11 | code | HMM | m5_is, m5_oos, split_time, m15_all_sorted, m5_train, m5_oos… | `# -----------------------------` |
| 12 | code |  | coarse_grid, coarse_results_dict, coarse_results | `# -----------------------------` |
| 13 | code |  | fine_results_dict, fine_results | `# -----------------------------` |
| 14 | code |  |  | `# -----------------------------` |
| 15 | code | Feature Engineering, HMM, Execution | _safe_float, _safe_int, _normalize_metrics, train_ml_model, evaluate_ml_model, _select_center_with_fallback… | `# -----------------------------` |
| 16 | code |  | run_mode_v2, all_rows, trades_by_mode, dual_tf_summary, dual_tf_summary | `# -----------------------------` |
| 17 | code | HMM, Probability threshold |  | `# -----------------------------` |
| 18 | code | HMM | ProductionRiskCircuitBreaker | `# -----------------------------` |
| 19 | code | Candidate Generation, Feature Engineering, Session Filtering, HMM, Risk Manager, Execution | diagnostics, candidate_reports, PipelineFunnel, FeatureLossAudit, RejectionBreakdown | `# ============================================================` |
| 20 | code | Feature Engineering, Triple Barrier Method, Session Filtering, HMM | _count_after_dropna, build_features_with_trace | `# ============================================================` |
| 21 | code | Session Filtering, HMM, Execution | generate_signals_diagnostic, diagnostic_signal_trace | `# ============================================================` |
| 22 | code | Session Filtering, HMM, Risk Manager, Execution | diagnostic_backtest_trace | `# ============================================================` |
| 23 | code | HMM | hmm_diagnostics | `# ============================================================` |
| 24 | code | XGBoost, Probability threshold, Execution | probability_distribution, threshold_sensitivity | `# ============================================================` |
| 25 | code |  | calibration_metrics | `# ============================================================` |
| 26 | code | Candidate Generation | compare_strategy_tester_signals | `# ============================================================` |
| 27 | code | Candidate Generation, Feature Engineering, Session Filtering, HMM, Execution | all_diagnostics | `# ============================================================` |
| 28 | code | Candidate Generation, Triple Barrier Method, Session Filtering, HMM, XGBoost, Risk Manager, Execution | print_funnel, print_labels, print_hmm, print_prob_dist, print_thresh_sens, print_feature_audit… | `# ============================================================` |
| 29 | code | Candidate Generation, Feature Engineering, HMM, XGBoost, Probability threshold, Execution | build_comparison_table, comp, reference_tf, target_tf, ref, tgt… | `# ============================================================` |
| 30 | code | Candidate Generation, Session Filtering, HMM, Execution | comp | `# ============================================================` |
| 31 | markdown |  |  | `## Traceability Layer (Added)` |
| 32 | code | Triple Barrier Method, HMM, XGBoost, Risk Manager | CandidateTrade, make_candidate_id, parameter_set_id, RejectionReason | `# ============================================================` |
| 33 | code | Candidate Generation, Feature Engineering, Triple Barrier Method, HMM, XGBoost, Risk Manager, Execution | PipelineProfiler, PIPELINE_STAGES, profiler | `# ============================================================` |
| 34 | code | Candidate Generation | attach_candidate_ids, assert_candidate_id_preserved | `# ============================================================` |
| 35 | code | Candidate Generation, HMM, Execution | run_ml_filtered_backtest_traced | `# ============================================================` |
| 36 | code | Probability threshold | probability_diagnostics_full, probability_drift_report | `# ============================================================` |
| 37 | code | HMM | hmm_diagnostics_v2 | `# ============================================================` |
| 38 | code | XGBoost | calibration_metrics_v2 | `# ============================================================` |
| 39 | code | Feature Engineering, Triple Barrier Method, HMM, XGBoost | get_candidate_lifecycle | `# ============================================================` |
| 40 | code | Candidate Generation, Triple Barrier Method, HMM, XGBoost | build_st_explorer_consistency_table | `# ============================================================` |
| 41 | code | Candidate Generation, Execution | build_timeframe_report | `# ============================================================` |
| 42 | code | Candidate Generation, HMM, Probability threshold, Execution | RootCauseAnalyzer | `# ============================================================` |
| 43 | code | Candidate Generation, Feature Engineering, Triple Barrier Method, HMM, XGBoost, Probability threshold, Risk Manager, Execution | all_diag_v2, all_candidates_by_tf | `# ============================================================` |
| 44 | code | Candidate Generation, HMM, XGBoost | drift_df, consistency_table, timeframe_report, rca, thresholds_by_tf, root_cause_df | `# ============================================================` |
| 45 | markdown |  |  | `## Pipeline Verification & Certification (Added)` |
| 46 | code | Triple Barrier Method, Session Filtering, HMM, XGBoost | VERIFY_PIPELINE | `# ============================================================` |
| 47 | code | Session Filtering |  | `# ============================================================` |
| 48 | code | Session Filtering |  | `# ============================================================` |
| 49 | code | Feature Engineering, Triple Barrier Method |  | `# ============================================================` |
| 50 | code | HMM, XGBoost |  | `# ============================================================` |
| 51 | code | Feature Engineering, XGBoost |  | `# ============================================================` |
| 52 | code | Session Filtering |  | `# ============================================================` |
| 53 | code | Feature Engineering, Execution |  | `# ============================================================` |
| 54 | code |  |  | `# ============================================================` |
| 55 | code | Candidate Generation, Triple Barrier Method, Session Filtering, HMM, Execution |  | `# ============================================================` |
| 56 | code | Candidate Generation, Session Filtering |  | `# ============================================================` |
| 57 | code |  |  | `# ============================================================` |

## Inputs / Outputs / Shared / Exported (heuristic)

### Imports (module inputs)

* `dataclasses.asdict`
* `dataclasses.dataclass`
* `dataclasses.field`
* `enum.Enum`
* `hashlib`
* `itertools`
* `json`
* `math`
* `matplotlib.pyplot`
* `numba.njit`
* `numpy`
* `os`
* `pandas`
* `pathlib.Path`
* `scipy.stats.entropy`
* `scipy.stats.kurtosis`
* `scipy.stats.skew`
* `seaborn`
* `sklearn.metrics.confusion_matrix`
* `sklearn.metrics.f1_score`
* `sklearn.metrics.precision_score`
* `sklearn.metrics.recall_score`
* `sys`
* `time`
* `typing.Any`
* `typing.Dict`
* `typing.Iterator`
* `typing.List`
* `typing.Tuple`
* `uuid`
* `warnings`

### Top-level definitions (exported objects / shared variables)

* `BALANCE_SCALE_THRESHOLD_CENTS`
* `BARS_PER_DAY`
* `BARS_PER_YEAR`
* `COARSE_CPCV_K_VAL`
* `COARSE_CPCV_N_BLOCKS`
* `COMMISSION_CENTS_PER_TRADE`
* `CPCVPurgedEmbargo`
* `CandidateTrade`
* `EMBARGO_HOURS`
* `EXEC_TF`
* `FINE_CPCV_K_VAL`
* `FINE_CPCV_N_BLOCKS`
* `FeatureLossAudit`
* `HMMXGBComposite`
* `HMM_FEATURES`
* `HOLDOUT_FRAC`
* `INITIAL_BALANCE_CENTS`
* `LABEL_COLS`
* `LEG_C_ATR_STOP`
* `LEG_C_ATR_TARGET`
* `LOT_CYCLE_SMALL`
* `MAX_DATA_YEARS`
* `MAX_POSITIONS_PER_CYCLE`
* `ML_TARGET_PARAMS`
* `N_JOBS`
* `PIPELINE_STAGES`
* `PIP_SIZE_PRICE`
* `PIP_VALUE_CENTS_PER_1LOT`
* `POSITION_A`
* `POSITION_B`
* `PipelineFunnel`
* `PipelineProfiler`
* `ProductionRiskCircuitBreaker`
* `RANDOM_STATE`
* `RR_MULT`
* `RejectionBreakdown`
* `RejectionReason`
* `RootCauseAnalyzer`
* `SLIPPAGE_PIPS`
* `SPREAD_CAP_POINTS`
* `STOP_LOSS_PIPS`
* `TF_TO_XAG_MASTER`
* `TF_TO_XAU_RAW`
* `TF_TO_XTI_MASTER`
* `TIMEFRAMES`
* `TREND_TF`
* `TimeframePipeline`
* `TrendPullbackStrategy`
* `VERIFY_PIPELINE`
* `_ML_FOLD_CACHE`
* `_count_after_dropna`
* `_empty_combo_result`
* `_get_or_compute_ml_folds`
* `_here`
* `_normalize_metrics`
* `_normalize_ohlcv`
* `_prev_py_path`
* `_project_root`
* `_project_root_str`
* `_run_backtest_numba`
* `_safe_float`
* `_safe_int`
* `_select_center_with_fallback`
* `_triple_barrier_numba`
* `add_session_features`
* `adx`
* `all_candidates_by_tf`
* `all_diag_v2`
* `all_diagnostics`
* `all_rows`
* `assert_candidate_id_preserved`
* `atr`
* `attach_candidate_ids`
* `build_coarse_grid`
* `build_comparison_table`
* `build_features`
* `build_features_with_trace`
* `build_refined_grid_from_top`
* `build_st_explorer_consistency_table`
* `build_timeframe_report`
* `calibration_metrics`
* `calibration_metrics_v2`
* `candidate_reports`
* `coarse_grid`
* `coarse_results`
* `coarse_results_dict`
* `comp`
* `compare_strategy_tester_signals`
* `compute_metrics`
* `consistency_table`
* `diagnostic_backtest_trace`
* `diagnostic_signal_trace`
* `diagnostics`
* `drift_df`
* `dual_tf_summary`
* `ema`
* `evaluate_combo_cpcv`
* `evaluate_ml_model`
* `fine_results`
* `fine_results_dict`
* `generate_signals_diagnostic`
* `get_candidate_lifecycle`
* `get_hmm_feature_list`
* `hmm_diagnostics`
* `hmm_diagnostics_v2`
* `hmm_feature_columns`
* `hmm_occ`
* `load_optimized_strategies`
* `load_panel`
* `m15_all`
* `m15_all_sorted`
* `m15_loss`
* `m15_oos`
* `m15_train`
* `m5_all`
* `m5_is`
* `m5_loss`
* `m5_oos`
* `m5_train`
* `make_candidate_id`
* `neg`
* `neu`
* `parameter_set_id`
* `pct_lost`
* `pipeline`
* `plot_plateau_heatmaps`
* `pos`
* `print_calibration`
* `print_feature_audit`
* `print_funnel`
* `print_hmm`
* `print_labels`
* `print_prob_dist`
* `print_rejections`
* `print_st_comparison`
* `print_thresh_sens`
* `prob_mean`
* `probability_diagnostics_full`
* `probability_distribution`
* `probability_drift_report`
* `profiler`
* `rca`
* `read_master_close`
* `read_mt4_csv`
* `read_xau_raw`
* `ref`
* `reference_tf`
* `root_cause_df`
* `rsi`
* `run_grid_parallel`
* `run_ml_filtered_backtest`
* `run_ml_filtered_backtest_traced`
* `run_mode_v2`
* `score_metrics`
* `select_plateau_center`
* `session_col_from_value`
* `split_dataset`
* `split_time`
* `st_cmp`
* `synth_vix_zscore`
* `target_tf`
* `tgt`
* `thr`
* `threshold_sensitivity`
* `thresholds_by_tf`
* `timeframe_report`
* `total`
* `trades_by_mode`
* `train_ml_model`
* `triple_barrier`
* `true_range`
* `validation_rows`
* `validation_summary`
