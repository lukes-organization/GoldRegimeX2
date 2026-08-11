# -----------------------------
# Global config
# -----------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Strategy TF contract
EXEC_TF = "M5"
TREND_TF = "M15"

# Account and execution assumptions
INITIAL_BALANCE_CENTS = 1500.0
SPREAD_CAP_POINTS = 40.0          # 40 points max = 4.0 pips
STOP_LOSS_PIPS = 15.0
RR_MULT = 2.0
MAX_POSITIONS_PER_CYCLE = 2
LOT_CYCLE_SMALL = [0.02, 0.02]    # base per-leg lots for the ~15 USD account
# --- Balance-tiered position sizing (evaluated in the live-simulation what-if) ---
# Step each leg's lot UP once REALIZED profit from the 15 USD start reaches this many
# cents. 5000 cents = 50 USD profit => account balance ~6500 cents (65 USD).
PROFIT_SCALE_THRESHOLD_CENTS = 5000.0
LOT_CYCLE_SCALED_OPTIONS = [0.03, 0.04]  # larger per-leg lots to evaluate above the threshold
BALANCE_SCALE_THRESHOLD_CENTS = 5000.0  # legacy alias (profit threshold, cents)
# --- DEPLOYED live-simulation sizing policy (applied ONLY in the Live Trading
#     Simulation cell, NOT in the optimization/sensitivity grid above) ---
LIVE_ENABLE_LOT_SCALING = True   # live replay escalates lots once profit clears the threshold
LIVE_SCALED_LOT = 0.03           # per-leg lot used above the threshold in the live replay

# Runtime controls
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

# CPCV-like controls for strategy parameter search
COARSE_CPCV_N_BLOCKS = 4
COARSE_CPCV_K_VAL = 2
FINE_CPCV_N_BLOCKS = 4
FINE_CPCV_K_VAL = 2
EMBARGO_HOURS = 24

# --- EXPERIMENT: time-of-day session filter -------------------------------
# True  => neutralize the NY/London/Asian time-of-day gate. ALL other filters
#          stay ON (trend/ADX, RSI pullback, HMM regime gate, spread cap,
#          ML threshold). Does NOT touch the model, HMM, or features.
# False => restore normal session gating.
DISABLE_SESSION_FILTER = False

# Holdout split
HOLDOUT_FRAC = 0.20
MAX_DATA_YEARS = 5   # Set to int (e.g. 5 or 10) to use only the last N years of data; None = use all available

BARS_PER_DAY = {"M15": 96, "M5": 288}
BARS_PER_YEAR = {"M15": 252 * 96, "M5": 252 * 288}

# Source paths
TF_TO_XAU_RAW = {
    "M15": Path("data/raw/XAU_15m_data.csv"),
    "M5": Path("data/raw/XAU_5m_data.csv"),
}

# Cross-commodity master close files (tab-separated MetaTrader format)
TF_TO_XAG_MASTER = {
    "M15": Path("data/raw/XAGUSD_M15_201601040100_202605072245.csv"),
    "M5":  Path("data/raw/XAGUSD_M5_201601040105_202605072255.csv"),
}
TF_TO_XTI_MASTER = {
    "M15": Path("data/raw/XTIUSD_M15_201702102000_202605072345.csv"),
    "M5":  Path("data/raw/XTIUSD_M5_201702102000_202605072355.csv"),
}

# USDCHF processed master (intraday DXY proxy, ~0.85 DXY correlation).
# Comma-separated, ISO dates - produced by src/data_consolidator.py.
TF_TO_USDCHF_MASTER = {
    "M15": Path("data/processed/USDCHF_master_M15.csv"),
    "M5":  Path("data/processed/USDCHF_master_M5.csv"),
}

# -----------------------------
# Timeframes (Phase 4 / Phase 6)
# -----------------------------
TIMEFRAMES = ["M15", "M5"]

# -----------------------------
# HMM Feature Registry (Phase 3)
# Explicit, version-controlled feature list for the stationary HMM.
# Avoids silently consuming all numeric columns.
# -----------------------------
HMM_FEATURES = {
    "volatility": [
        "atr_20", "atr_normalized", "volatility_20",
        "synth_vix_zscore", "atr14", "atr100", "atr_expansion",
    ],
    "cross_commodity": [
        "log_return", "xag_log_return", "xti_log_return", "usdchf_log_return",
        "gold_silver_ratio_z", "gold_oil_ratio_z", "gold_chf_ratio_z",
    ],
    "oscillators": [
        "rsi5",
    ],
    "temporal": [
        "hour_sin", "hour_cos",
    ],
    "regime": [
        "regime_code",
    ],
}


def get_hmm_feature_list():
    "Flatten the HMM feature registry into a single list."
    features = []
    for group in HMM_FEATURES.values():
        features.extend(group)
    return features


# -----------------------------
# Centralized Train/OOS Split (Phase 5)
# Single consistent split mechanism used by all stages.
# -----------------------------
def split_dataset(df, holdout_frac):
    "Chronological train/OOS split. Returns (train_df, oos_df, split_time)."
    df = df.sort_index().copy()
    if len(df) < 1000:
        raise RuntimeError("Not enough rows for split. Got %d rows." % len(df))
    split_idx = int(len(df) * (1.0 - holdout_frac))
    split_idx = max(2000, min(split_idx, len(df) - 1))
    train = df.iloc[:split_idx].copy()
    oos = df.iloc[split_idx:].copy()
    split_time = train.index[-1]
    return train, oos, split_time


# -----------------------------
# Pipeline Container (Phase 6-7 / Phase 11)
# Each timeframe owns its own state. No shared mutable state.
# -----------------------------
class TimeframePipeline:
    "Owns all state for a single timeframe ML experiment."
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.raw_all = None           # Raw panel data
        self.train_df = None          # IS training data
        self.oos_df = None            # OOS holdout data
        self.split_time = None        # Split timestamp
        self.train_feat = None        # Engineered features (train)
        self.model = None             # Trained HMMXGBComposite
        self.threshold = None         # Optimised xgb_threshold
        self.plateau = None           # Plateau centre dict
        self.metrics_is = None        # IS validation metrics
        self.metrics_oos = None       # OOS validation metrics
        self.trades_is = None         # IS trades dataframe
        self.trades_oos = None        # OOS trades dataframe


# Initialise one pipeline per timeframe
pipeline = {tf: TimeframePipeline(tf) for tf in TIMEFRAMES}

print("Execution TF:", EXEC_TF)
print("Trend TF:", TREND_TF)
print("Initial balance (cents):", INITIAL_BALANCE_CENTS)
print("Spread cap (points):", SPREAD_CAP_POINTS)
print("CPCV paths coarse:", math.comb(COARSE_CPCV_N_BLOCKS, COARSE_CPCV_K_VAL))
print("CPCV paths fine:", math.comb(FINE_CPCV_N_BLOCKS, FINE_CPCV_K_VAL))
print("Max data years:", MAX_DATA_YEARS if MAX_DATA_YEARS else "All available")