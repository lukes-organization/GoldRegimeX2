# Optional exploratory speed cap (applies to QUICK_MODE True/False)
ENABLE_ENTRY_CAP = False

# Hard cap per (timeframe, strategy, exit_model) on ENTRY parameter combos.
# This is the biggest speed lever for QUICK_MODE=False.
if QUICK_MODE:
    ENTRY_CAP_BY_TF = {"M15": 80, "M5": 40}
else:
    ENTRY_CAP_BY_TF = {"M15": 200, "M5": 80}

# Seed for deterministic subset selection
ENTRY_CAP_SEED_BASE = 42

print("ENABLE_ENTRY_CAP:", ENABLE_ENTRY_CAP)
print("ENTRY_CAP_BY_TF:", ENTRY_CAP_BY_TF)
print("ENTRY_CAP_SEED_BASE:", ENTRY_CAP_SEED_BASE)