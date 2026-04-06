import numpy as np
import pandas as pd
from pathlib import Path
from src.logger import setup_logger

logger = setup_logger(__name__)

# Per-timeframe configuration ─────────────────────────────────────────────────
# Kalman obs_cov controls smoothing: higher value = more smoothing (less trust
# in raw observations). M15 is ~4× noisier than H1; M5 uses a low obs_cov
# (0.05) to keep the filter very responsive for fast scalping regime detection.
# n_states_default: M5 uses 4 states to capture micro-noise; M15/H1 use 3.
# Single shared USDCHF master file — intraday proxy for DXY (correlates ~0.85
# with USD Index for XAUUSD signals and is always available on Headway).
# Run  python main.py --mode consolidate  to build this from MT5 exports.
USDCHF_MASTER_PATH = Path("data/processed/USDCHF_master.csv")

# Legacy alias kept for any callers that still reference the old name directly.
DXY_RAW_PATH = USDCHF_MASTER_PATH

TF_CONFIG = {
    "H1": {
        "raw_path":     Path("data/raw/XAU_1h_data.csv"),
        "processed_path": Path("data/processed/gold_h1_processed.parquet"),
        "obs_cov_default": 1.0,
        "trans_cov_default": 0.01,
        "n_states_default": 3,
    },
    "M15": {
        "raw_path":     Path("data/raw/XAU_15m_data.csv"),
        "processed_path": Path("data/processed/gold_m15_processed.parquet"),
        "obs_cov_default": 4.0,
        "trans_cov_default": 0.01,
        "n_states_default": 3,
    },
    "M5": {
        "raw_path":     Path("data/raw/XAU_5m_data.csv"),
        "processed_path": Path("data/processed/gold_m5_processed.parquet"),
        "obs_cov_default": 0.05,   # low value = responsive Kalman for scalping
        "trans_cov_default": 0.01,
        "n_states_default": 4,     # 4 states to capture micro-regime noise
    },
}

# Legacy aliases kept for callers that use the bare constants directly
RAW_PATH = TF_CONFIG["H1"]["raw_path"]
PROCESSED_PATH = TF_CONFIG["H1"]["processed_path"]


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            "Export XAUUSD data from MT5 as CSV (semicolon-delimited) and place it at that path."
        )
    df = pd.read_csv(
        path, sep=";", parse_dates=["Date"], date_format="%Y.%m.%d %H:%M"
    )
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def load_usdchf_data(path: Path) -> pd.DataFrame | None:
    """Load the USDCHF master CSV and return a ``usdchf_log_return`` column.

    Returns ``None`` if the file does not exist — callers treat this as
    "USDCHF feature not available for this training run".

    Run ``python main.py --mode consolidate`` to build the master from MT5
    exports before training.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)
    df["usdchf_log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    logger.info("Loaded USDCHF master: %d rows from %s", len(df), path)
    return df[["usdchf_log_return"]]


def map_usdchf_to_bars(df_index: pd.DatetimeIndex, usdchf_df: pd.DataFrame) -> pd.Series:
    """Align USDCHF log returns onto XAUUSD bar timestamps.

    Handles both daily USDCHF data (e.g. annual CSV exports) and intraday data
    (e.g. direct M5 MT5 export):
    - Daily: normalise each bar's timestamp to midnight and ffill by date.
    - Intraday: reindex directly onto bar timestamps with ffill.
    """
    series = usdchf_df["usdchf_log_return"].copy()
    # Strip timezone for tz-aware MT5 sync indices
    idx = df_index.tz_localize(None) if df_index.tz is not None else df_index

    # Detect daily vs intraday by checking whether all times are midnight
    is_daily = (series.index == series.index.normalize()).all()

    if is_daily:
        # Daily data — map each intraday bar onto its calendar date
        # Extend the series forward to cover any bars beyond the last date
        normalized = idx.normalize()
        max_date = normalized.max()
        if max_date > series.index.max():
            extension = pd.date_range(
                series.index.max() + pd.Timedelta(days=1), max_date, freq="D"
            )
            series = pd.concat([series, pd.Series(series.iloc[-1], index=extension)])
        series = series.ffill()
        return normalized.map(series)
    else:
        # Intraday data — forward-fill onto bar timestamps directly
        return series.reindex(idx, method="ffill")


# Legacy aliases — kept so old imports don't crash if any caller still uses them.
def load_dxy_data(path: Path) -> pd.DataFrame | None:              # noqa: D103
    return load_usdchf_data(path)


def map_dxy_to_bars(df_index: pd.DatetimeIndex, dxy_df: pd.DataFrame) -> pd.Series:  # noqa: D103
    # Adapt old dxy_log_return column name if present (backwards compat)
    if "dxy_log_return" in dxy_df.columns and "usdchf_log_return" not in dxy_df.columns:
        dxy_df = dxy_df.rename(columns={"dxy_log_return": "usdchf_log_return"})
    return map_usdchf_to_bars(df_index, dxy_df)


def filter_data(df: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    cutoff = df.index.max() - pd.DateOffset(years=years)
    df = df[df.index >= cutoff]
    before = len(df)
    df = df[df["Volume"] > 5]
    logger.info(
        "Filtered to last %d years: %d rows (%d low-volume bars removed)",
        years, len(df), before - len(df),
    )
    return df


def compute_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).rename("log_return")


def kalman_smooth(data: np.ndarray, obs_cov: float = 1.0, trans_cov: float = 0.01) -> np.ndarray:
    mu = 0.0
    P = 1.0
    smoothed = np.zeros_like(data, dtype=np.float64)
    for i in range(len(data)):
        P = P + trans_cov
        if np.isnan(data[i]):
            smoothed[i] = mu
        else:
            K = P / (P + obs_cov)
            mu = mu + K * (data[i] - mu)
            P = (1 - K) * P
            smoothed[i] = mu
    return smoothed


def compute_volatility(log_returns: pd.Series, window: int = 20) -> pd.Series:
    return log_returns.rolling(window=window).std().rename("volatility")


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).rename("rsi")


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return (atr / df["Close"]).rename("atr_normalized")


def process_pipeline(
    obs_cov: float = None,
    trans_cov: float = None,
    save: bool = True,
    tf: str = "H1",
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline for the given timeframe.

    Args:
        obs_cov: Kalman observation covariance. Defaults to TF-specific value
                 (1.0 for H1, 4.0 for M15).
        trans_cov: Kalman transition covariance. Defaults to 0.01.
        save: Persist processed parquet to disk.
        tf: Timeframe — ``"H1"`` or ``"M15"``.
    """
    tf = tf.upper()
    if tf not in TF_CONFIG:
        raise ValueError(f"Unknown timeframe '{tf}'. Choose from {list(TF_CONFIG)}")

    cfg = TF_CONFIG[tf]
    obs_cov = obs_cov if obs_cov is not None else cfg["obs_cov_default"]
    trans_cov = trans_cov if trans_cov is not None else cfg["trans_cov_default"]

    df = load_raw_data(cfg["raw_path"])
    df = filter_data(df)

    df["log_return"] = compute_log_returns(df["Close"])
    df["kalman_return"] = kalman_smooth(df["log_return"].values, obs_cov, trans_cov)
    df["volatility"] = compute_volatility(df["log_return"])
    df["rsi"] = compute_rsi(df["Close"])
    df["rsi_slope"] = df["rsi"].diff()
    df["atr_normalized"] = compute_atr(df)

    # ── Optional USDCHF cross-asset feature ─────────────────────────────────
    # USDCHF is an intraday DXY proxy — correlates ~0.85 with USD Index and
    # is always available on Headway as a standard Forex pair.
    # Build the master with:  python main.py --mode consolidate
    usdchf_df = load_usdchf_data(USDCHF_MASTER_PATH)
    if usdchf_df is not None:
        df["usdchf_log_return"] = map_usdchf_to_bars(df.index, usdchf_df)
        n_usdchf = df["usdchf_log_return"].notna().sum()
        logger.info("USDCHF merged: %d non-null usdchf_log_return values.", n_usdchf)
    else:
        logger.info(
            "USDCHF master not found at %s — pipeline running with 4 base features. "
            "Run  python main.py --mode consolidate  to enable the 5th feature.",
            USDCHF_MASTER_PATH,
        )

    df.dropna(inplace=True)
    logger.info(
        "Pipeline [%s] complete: %d rows, columns: %s",
        tf, len(df), list(df.columns),
    )

    if save:
        out_path = cfg["processed_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        logger.info("Saved processed data to %s", out_path)

    return df
