# Imports + Config
import os
import math
import json
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    JOBLIB_OK = True
except Exception:
    JOBLIB_OK = False

np.random.seed(42)

# -----------------------------
# Runtime controls
# -----------------------------
N_JOBS = 6  # lower to reduce memory pressure under loky
QUICK_MODE = False
RESEARCH_YEARS = 5 # enforced always, regardless of QUICK_MODE

# -----------------------------
# Data paths
# -----------------------------
M5_PATH = Path("data/raw/XAU_5m_data.csv")
M15_PATH = Path("data/raw/XAU_15m_data.csv")

# -----------------------------
# Core assumptions
# -----------------------------
TIMEFRAMES = ["M15", "M5"]
INITIAL_BALANCE_CENTS = 1500.0
PIP_SIZE_PRICE = 0.10
PIP_VALUE_CENTS_PER_1LOT = 100.0
SLIPPAGE_PIPS = 0.30
SPREAD_CAP_POINTS = 40.0
COMMISSION_CENTS_PER_TRADE = 0.0

# Split position support
POSITION_A = 0.02
POSITION_B = 0.02

# M5 Grids (Tighter targets, wider stops to survive noise and high friction)
M5_ADX_GRID = [20, 25]
M5_PULLBACK_RSI_GRID = [30, 35]
M5_CONFIRMATION_GRID = [1, 2]
M5_ATR_STOP_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]  # Extended upward to allow M5 breathing room
M5_LEG_A_TARGET_GRID = [0.8, 1.0]
M5_ENTRY_TARGET_GRID = [1.0, 1.5]

# Exit modes
EXIT_MODELS = [
    "fixed_tp",
    "mr_exit",
    "fixed_tp_plus_mr",
    "partial_tp_plus_mr",
    "partial_tp_mr_time_stop",
]

# ATR multiple grids
LEG_A_ATR_TARGET_GRID = [1.0, 1.5, 2.0]  # Leg A target (ATR multiple)
ENTRY_ATR_TARGET_GRID = [2.0, 2.5, 3.0]  # Leg B (fixed TP) ATR multiple
ATR_TARGET_GRID = ENTRY_ATR_TARGET_GRID  # keep existing name used by strategies

# Leg C constants (Scale-in removed from grid search to save compute)
LEG_C_ATR_TARGET = 0.5
LEG_C_ATR_STOP = 0.5

TIME_STOP_GRID_BY_TF = {
    "M15": [120, 180, 240],
    "M5": [30, 60, 90],
}
TRAIL_MULT_GRID = [1.5, 2.0, 2.5]

# Session filter options
SESSION_FILTER_VALUES = [None, "London", "NY", "London_NY"]

# -----------------------------
# Strategy A grids (Loosened heavily to restore M15 trade frequency)
# -----------------------------
# Lowered ADX so we don't need a massive macro trend to trigger
ADX_GRID = [15.0, 18.0, 20.0, 25.0] 
# Loosened RSI heavily. On a 15-minute chart, trends rarely retrace all the way to 25/30 RSI. 
# Allowing 35-45 RSI captures shallower, highly valid trend pullbacks.
PULLBACK_RSI_GRID = [25, 30, 35.0, 40.0, 45.0] 
CONFIRMATION_GRID = [1, 2]  # Get in faster, less lag
ATR_STOP_GRID = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]  # Loosened to allow more breathing room for M15 noise

# -----------------------------
# Strategy B grids (Loosened for Frequency)
# -----------------------------
ATR_EXPANSION_GRID = [1.1, 1.25, 1.4]  # 1.5+ is too rare; 1.1+ catches standard volatility cycles
BREAKOUT_LOOKBACK_GRID = [10, 20, 30]
BREAKOUT_BUFFER_GRID = [0.0, 0.25]

if QUICK_MODE:
    # Stronger quick cut to keep runtime practical
    ADX_GRID = ADX_GRID[:2]
    PULLBACK_RSI_GRID = PULLBACK_RSI_GRID[:2]
    CONFIRMATION_GRID = CONFIRMATION_GRID[:2]
    ATR_STOP_GRID = ATR_STOP_GRID[:1]
    LEG_A_ATR_TARGET_GRID = LEG_A_ATR_TARGET_GRID[:2]
    ENTRY_ATR_TARGET_GRID = ENTRY_ATR_TARGET_GRID[:2]
    ATR_TARGET_GRID = ENTRY_ATR_TARGET_GRID

    ATR_EXPANSION_GRID = ATR_EXPANSION_GRID[:3]
    BREAKOUT_LOOKBACK_GRID = BREAKOUT_LOOKBACK_GRID[:2]
    BREAKOUT_BUFFER_GRID = BREAKOUT_BUFFER_GRID[:2]

    SESSION_FILTER_VALUES = [None, "London_NY"]
    EXIT_MODELS = ["fixed_tp", "mr_exit", "fixed_tp_plus_mr"]

    TIME_STOP_GRID_BY_TF["M15"] = TIME_STOP_GRID_BY_TF["M15"][:2]
    TIME_STOP_GRID_BY_TF["M5"] = TIME_STOP_GRID_BY_TF["M5"][:2]
    TRAIL_MULT_GRID = TRAIL_MULT_GRID[:2]

print("JOBLIB_OK:", JOBLIB_OK)
print("N_JOBS:", N_JOBS)
print("QUICK_MODE:", QUICK_MODE)
print("RESEARCH_YEARS (enforced):", RESEARCH_YEARS)