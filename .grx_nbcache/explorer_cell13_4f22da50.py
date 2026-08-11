# -----------------------------
# Main run: load M5 + M15 data and split into pipeline containers
# Phase 5:  Uses split_dataset for consistent splits
# Phase 11: Data stored in pipeline container, not loose globals
# -----------------------------

for tf in TIMEFRAMES:
    pipeline[tf].raw_all = load_panel(tf).sort_index().copy()

# --- IS / OOS chronological split ---
# M5 is the reference: its split determines split_time.
# M15 aligns to the same split_time boundary so both timeframes
# share a consistent cut-off.

m5_is, m5_oos, split_time = split_dataset(pipeline["M5"].raw_all, HOLDOUT_FRAC)
pipeline["M5"].train_df = m5_is
pipeline["M5"].oos_df = m5_oos
pipeline["M5"].split_time = split_time

# Align M15 to the same split_time boundary
m15_all_sorted = pipeline["M15"].raw_all.sort_index()
pipeline["M15"].train_df = m15_all_sorted.loc[m15_all_sorted.index <= split_time].copy()
pipeline["M15"].oos_df = m15_all_sorted.loc[m15_all_sorted.index > split_time].copy()
pipeline["M15"].split_time = split_time

print("=" * 60)
print("IS / OOS DATA SPLIT SUMMARY")
print("=" * 60)
print("Holdout fraction:  %.0f%%  (OOS)" % (HOLDOUT_FRAC * 100))
print("Split timestamp:  ", split_time)
for tf in TIMEFRAMES:
    p = pipeline[tf]
    total = len(p.raw_all)
    print("\n  %s total:  %d" % (tf, total))
    print("  %s IS   :  %d   (%.1f%%)   %s -> %s" % (tf, len(p.train_df), len(p.train_df)/total*100, p.train_df.index[0].strftime('%Y-%m-%d'), p.train_df.index[-1].strftime('%Y-%m-%d')))
    print("  %s OOS  :  %d   (%.1f%%)   %s -> %s" % (tf, len(p.oos_df), len(p.oos_df)/total*100, p.oos_df.index[0].strftime('%Y-%m-%d'), p.oos_df.index[-1].strftime('%Y-%m-%d')))
print()
print("Note: OOS typically has fewer trades because (1) it covers")
print("      a shorter calendar span (%.0f%% of data) and (2) regime" % (HOLDOUT_FRAC * 100))
print("      mix may differ, affecting signal generation.")
print("=" * 60)

# Backward-compatible references for cells not yet fully refactored
m5_train = pipeline["M5"].train_df
m5_oos = pipeline["M5"].oos_df
m15_train = pipeline["M15"].train_df
m15_oos = pipeline["M15"].oos_df
m5_all = pipeline["M5"].raw_all
m15_all = pipeline["M15"].raw_all