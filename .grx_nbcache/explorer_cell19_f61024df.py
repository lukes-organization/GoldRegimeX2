# -----------------------------
# Phase 9: Diagnostic Reports per Timeframe
# Each timeframe reports independently -- no cross-TF coupling.
# -----------------------------

for tf in TIMEFRAMES:
    p = pipeline[tf]
    met_is = p.metrics_is or {}
    met_oos = p.metrics_oos or {}
    plateau = p.plateau or {}

    print("=" * 60)
    print("TIMEFRAME: %s" % tf)
    print("=" * 60)
    if p.train_df is not None:
        print("  Training Samples:        %d" % len(p.train_df))
    else:
        print("  Training Samples:        N/A")
    if p.oos_df is not None:
        print("  OOS Samples:             %d" % len(p.oos_df))
    else:
        print("  OOS Samples:             N/A")
    if p.split_time is not None:
        print("  Split Time:               %s" % p.split_time)
    else:
        print("  Split Time:               N/A")
    if p.train_feat is not None:
        print("  Feature Count (HMM):      %d" % len(hmm_feature_columns(p.train_feat)))
    else:
        print("  Feature Count (HMM):      N/A")
    if p.model is not None:
        print("  HMM States:              %d" % p.model.n_components)
    else:
        print("  HMM States:              N/A")
    if p.threshold is not None:
        print("  Probability Threshold:    %s" % p.threshold)
    else:
        print("  Probability Threshold:    N/A")
    print("  IS Trades:               %s" % met_is.get("n_trades", "N/A"))
    print("  OOS Trades:              %s" % met_oos.get("n_trades", "N/A"))
    if met_is.get('win_rate') is not None:
        print("  IS Win Rate:             %.2f%%" % (met_is.get("win_rate", 0) * 100))
    else:
        print("  IS Win Rate:             N/A")
    if met_oos.get('win_rate') is not None:
        print("  OOS Win Rate:            %.2f%%" % (met_oos.get("win_rate", 0) * 100))
    else:
        print("  OOS Win Rate:            N/A")
    if met_is.get('profit_factor') is not None:
        print("  IS Profit Factor:        %.2f" % met_is.get("profit_factor", 0))
    else:
        print("  IS Profit Factor:        N/A")
    if met_oos.get('profit_factor') is not None:
        print("  OOS Profit Factor:       %.2f" % met_oos.get("profit_factor", 0))
    else:
        print("  OOS Profit Factor:       N/A")
    if met_is.get('max_drawdown_pct') is not None:
        print("  IS Max Drawdown:         %.2f%%" % met_is.get("max_drawdown_pct", 0))
    else:
        print("  IS Max Drawdown:         N/A")
    if met_oos.get('max_drawdown_pct') is not None:
        print("  OOS Max Drawdown:        %.2f%%" % met_oos.get("max_drawdown_pct", 0))
    else:
        print("  OOS Max Drawdown:        N/A")
    print("  Plateau Width:           %s" % plateau.get("plateau_width_local", "N/A"))
    print("  Plateau Score:           %s" % plateau.get("selection_score", "N/A"))
    if met_is:
        print("  IS Score (Sharpe):       %.4f" % score_metrics(met_is))
    else:
        print("  IS Score (Sharpe):       N/A")
    if met_oos:
        print("  OOS Score (Sharpe):      %.4f" % score_metrics(met_oos))
    else:
        print("  OOS Score (Sharpe):      N/A")
    if met_is and met_oos:
        s_is = score_metrics(met_is)
        s_oos = score_metrics(met_oos)
        drop = (s_is - s_oos) / (abs(s_is) + 1e-9)
        print("  Relative Score Drop:      %.2f%%" % (drop * 100), "  <- WARNING: >30%%" if drop > 0.30 else "")
    print()

print("=" * 60)
print("DIAGNOSTICS COMPLETE -- Both timeframes reported independently.")
print("=" * 60)