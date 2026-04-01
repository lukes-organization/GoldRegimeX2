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
# Single DXY file — DXY is a daily index; one complete file is shared across all TFs.
# Export from MT5 History Center (XAUUSD → H1, then switch to DXY/USDX) and save here.
DXY_RAW_PATH = Path("data/raw/DXY_data.csv")

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


def load_dxy_data(path: Path) -> pd.DataFrame | None:
    """Load a DXY CSV and return a single-column DataFrame with ``dxy_log_return``.

    Returns ``None`` if the file does not exist — callers treat this as
    "DXY feature not available for this training run".

    Expected format: same semicolon-delimited MT5 export as XAUUSD, with a
    ``Close`` column.  File names follow the pattern ``DXY_{tf}_data.csv``
    (e.g. ``data/raw/DXY_1h_data.csv``).
    """
    if not path.exists():
        return None
    # DXY_data.csv is saved as YYYY-MM-DD (daily). Use infer to be format-agnostic.
    df = pd.read_csv(path, sep=";", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df["dxy_log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    logger.info("Loaded DXY data: %d rows from %s", len(df), path)
    return df[["dxy_log_return"]]


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

    # ── Optional DXY cross-asset feature ────────────────────────────────────
    # DXY is a daily index shared across all TFs — one file covers H1/M15/M5.
    dxy_path = DXY_RAW_PATH
    if dxy_path:
        dxy_df = load_dxy_data(dxy_path)
        if dxy_df is not None:
            # DXY is daily — normalize each intraday bar's timestamp to midnight
            # so it matches the DXY date index before mapping.
            df["dxy_log_return"] = df.index.normalize().map(dxy_df["dxy_log_return"])
            n_dxy = df["dxy_log_return"].notna().sum()
            logger.info("DXY merged: %d non-null dxy_log_return values.", n_dxy)
        else:
            logger.info(
                "DXY file not found at %s — pipeline running without cross-asset feature. "
                "Export DXY from MT5 History Center to enable it.",
                dxy_path,
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
