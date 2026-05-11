import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from src.logger import setup_logger

logger = setup_logger(__name__)

MODELS_DIR = Path("models")


def get_gmm_paths(tf: str, broker: str = "headway_cent") -> tuple[Path, Path]:
    """Return (gmm_path, scaler_path) for the given TF/broker pair."""
    return (
        MODELS_DIR / f"gmm_{tf.upper()}_{broker}.pkl",
        MODELS_DIR / f"scaler_{tf.upper()}_{broker}.pkl",
    )


def save_gmm_model(gmm, scaler, tf: str, broker: str = "headway_cent") -> None:
    """Persist the fitted GaussianMixture and StandardScaler to models/."""
    gmm_path, scaler_path = get_gmm_paths(tf, broker)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(gmm_path, "wb") as f:
        pickle.dump(gmm, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("GMM saved: %s | Scaler: %s", gmm_path.name, scaler_path.name)


def load_gmm_model(tf: str, broker: str = "headway_cent"):
    """Load the fitted GaussianMixture + StandardScaler.

    Raises FileNotFoundError if files are missing — run --mode train first.
    """
    gmm_path, scaler_path = get_gmm_paths(tf, broker)
    if not gmm_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"GMM/Scaler models not found for [{tf.upper()}/{broker}]. "
            f"Run  python main.py --mode train --tf {tf.upper()} --broker {broker}  first."
        )
    with open(gmm_path, "rb") as f:
        gmm = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("GMM + Scaler loaded [%s/%s]", tf.upper(), broker)
    return gmm, scaler


# XGBoost continuous feature columns that require StandardScaler normalization.
# Discrete columns (hmm_state, gmm_vol_cluster) are intentionally excluded.
# All external asset log-return columns are included; prepare_features filters
# dynamically to only those present in the DataFrame via get_continuous_cols().
CONTINUOUS_FEATURE_COLS = [
    "rsi_slope", "atr_normalized", "prev_log_return",
    "usdchf_log_return", "xagusd_log_return", "xtiusd_log_return",
    "us500_log_return", "usdjpy_log_return", "synth_vix_zscore",
]


def get_continuous_cols(df: pd.DataFrame) -> list[str]:
    """Return the subset of CONTINUOUS_FEATURE_COLS actually present in df.

    Filters to columns that exist AND are >50% non-null so the scaler is
    never fitted on a mostly-missing series.
    """
    return [
        c for c in CONTINUOUS_FEATURE_COLS
        if c in df.columns and df[c].notna().mean() > 0.5
    ]


def get_feature_scaler_path(tf: str, broker: str = "headway_cent") -> Path:
    """Return the XGBoost feature-scaler path for the given TF/broker pair."""
    return MODELS_DIR / f"feature_scaler_{tf.upper()}_{broker}.pkl"


def save_feature_scaler(scaler, tf: str, broker: str = "headway_cent") -> None:
    """Persist the fitted XGBoost feature StandardScaler to models/."""
    path = get_feature_scaler_path(tf, broker)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Feature scaler saved: %s", path.name)


def load_feature_scaler(tf: str, broker: str = "headway_cent"):
    """Load the fitted XGBoost feature StandardScaler.

    Raises FileNotFoundError if the file is missing — run --mode train first.
    """
    path = get_feature_scaler_path(tf, broker)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature scaler not found for [{tf.upper()}/{broker}]. "
            f"Run  python main.py --mode train --tf {tf.upper()} --broker {broker}  first."
        )
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Feature scaler loaded [%s/%s]", tf.upper(), broker)
    return scaler


# Per-timeframe configuration ─────────────────────────────────────────────────
# Kalman obs_cov controls smoothing: higher value = more smoothing (less trust
# in raw observations). M15 is ~4× noisier than H1; M5 uses a low obs_cov
# (0.05) to keep the filter very responsive for fast scalping regime detection.
# n_states_default: M5 uses 4 states to capture micro-noise; M15/H1 use 3.
# Per-TF USDCHF master files — each uses bars matching the trading timeframe
# so the cross-asset feature is aligned during merge.
# Run  python main.py --mode consolidate  to build all three from MT5 exports.
USDCHF_MASTER_PATH     = Path("data/processed/USDCHF_master.csv")      # H1
USDCHF_MASTER_PATH_M15 = Path("data/processed/USDCHF_master_M15.csv")  # M15
USDCHF_MASTER_PATH_M5  = Path("data/processed/USDCHF_master_M5.csv")   # M5

_USDCHF_PATH_BY_TF: dict[str, Path] = {
    "H1":  USDCHF_MASTER_PATH,
    "M15": USDCHF_MASTER_PATH_M15,
    "M5":  USDCHF_MASTER_PATH_M5,
}

# Legacy alias kept for any callers that still reference the old name directly.
DXY_RAW_PATH = USDCHF_MASTER_PATH

# ── New external asset master paths ──────────────────────────────────────────
# Build these with:  python main.py --mode consolidate

_XAGUSD_PATH_BY_TF: dict[str, Path] = {
    "H1":  Path("data/processed/XAGUSD_master.csv"),
    "M15": Path("data/processed/XAGUSD_master_M15.csv"),
    "M5":  Path("data/processed/XAGUSD_master_M5.csv"),
}

_XTIUSD_PATH_BY_TF: dict[str, Path] = {
    "H1":  Path("data/processed/XTIUSD_master.csv"),
    "M15": Path("data/processed/XTIUSD_master_M15.csv"),
    "M5":  Path("data/processed/XTIUSD_master_M5.csv"),
}

_US500_PATH_BY_TF: dict[str, Path] = {
    "H1":  Path("data/processed/US500_master.csv"),
    "M15": Path("data/processed/US500_master_M15.csv"),
    "M5":  Path("data/processed/US500_master_M5.csv"),
}

_USDJPY_PATH_BY_TF: dict[str, Path] = {
    "H1":  Path("data/processed/USDJPY_master.csv"),
    "M15": Path("data/processed/USDJPY_master_M15.csv"),
    "M5":  Path("data/processed/USDJPY_master_M5.csv"),
}

# Lookup table: asset_key → per-TF path dict
# asset_key matches the log-return column name prefix (e.g. "usdchf" → "usdchf_log_return")
_EXTERNAL_ASSET_PATHS: dict[str, dict[str, Path]] = {
    "usdchf": _USDCHF_PATH_BY_TF,
    "xagusd": _XAGUSD_PATH_BY_TF,
    "xtiusd": _XTIUSD_PATH_BY_TF,
    "us500":  _US500_PATH_BY_TF,
    "usdjpy": _USDJPY_PATH_BY_TF,
}

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


def load_asset_data(path: Path, col_name: str) -> pd.DataFrame | None:
    """Load any asset master CSV and compute its log-return column.

    Generic version of the old ``load_usdchf_data``.  Returns a single-column
    DataFrame with column *col_name* (e.g. ``"xagusd_log_return"``), or
    ``None`` if the file does not exist — callers treat that as "feature not
    available, degrade gracefully".

    Args:
        path:     Path to the processed master CSV (OHLCV, DatetimeIndex).
        col_name: Name to assign to the computed log-return series.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    if "Close" not in df.columns:
        logger.warning("Master file %s has no 'Close' column — skipping.", path.name)
        return None
    df[col_name] = np.log(df["Close"] / df["Close"].shift(1))
    logger.info("Loaded %s master: %d rows from %s", col_name, len(df), path.name)
    return df[[col_name]]


def map_asset_to_bars(
    df_index: pd.DatetimeIndex,
    asset_df: pd.DataFrame,
    col_name: str,
    return_raw: bool = False,
) -> "pd.Series | tuple[pd.Series, pd.Series]":
    """Align any asset log-return series onto XAUUSD bar timestamps.

    Generic version of the old ``map_usdchf_to_bars``.  Handles both daily
    and intraday source data via forward-fill (daily) or reindex+ffill
    (intraday).

    Args:
        df_index:   DatetimeIndex of XAUUSD bars to align onto.
        asset_df:   Single-column DataFrame from ``load_asset_data``.
        col_name:   Column name in *asset_df* (e.g. ``"xagusd_log_return"``).
        return_raw: When True, return ``(ffilled_series, raw_before_ffill)``.
                    The raw series has NaN wherever the asset had no bar at that
                    timestamp — use it for staleness counting.  Default False
                    returns only the ffilled series (backwards-compatible).

    Returns:
        Single ffilled Series (return_raw=False) or
        tuple (ffilled_series, raw_series) (return_raw=True).
    """
    series = asset_df[col_name].copy()
    idx = df_index.tz_localize(None) if df_index.tz is not None else df_index

    is_daily = (series.index == series.index.normalize()).all()

    if is_daily:
        normalized = idx.normalize()
        max_date = normalized.max()
        if max_date > series.index.max():
            extension = pd.date_range(
                series.index.max() + pd.Timedelta(days=1), max_date, freq="D"
            )
            series = pd.concat([series, pd.Series(series.iloc[-1], index=extension)])
        series = series.ffill()
        result = pd.Series(normalized.map(series).values, index=df_index, name=col_name)
        if return_raw:
            return result, result  # daily: no intrabar gaps, staleness always 0
        return result
    else:
        _series = series.copy()
        try:
            _series.index = _series.index.as_unit("ns")
            _idx = idx.as_unit("ns")
        except AttributeError:
            _idx = idx
        raw     = _series.reindex(_idx)          # NaN where asset has no bar
        ffilled = raw.ffill()
        raw_out   = pd.Series(raw.values,    index=df_index, name=col_name)
        ffill_out = pd.Series(ffilled.values, index=df_index, name=col_name)
        if return_raw:
            return ffill_out, raw_out
        return ffill_out


def _compute_staleness(raw_series: pd.Series, col_name: str) -> pd.Series:
    """Count consecutive bars since the last non-NaN observation.

    Call this BEFORE forward-filling *raw_series*.  The staleness counter
    gives XGBoost a data-quality signal: 0 = live observation, N = N bars
    of forward-fill (stale).

    Returns a pd.Series of non-negative ints with name ``{col_name}_staleness``.
    """
    counter = 0
    values = []
    for v in raw_series:
        if pd.notna(v):
            counter = 0
        else:
            counter += 1
        values.append(counter)
    return pd.Series(values, index=raw_series.index, name=f"{col_name}_staleness")


def compute_synth_vix(df: pd.DataFrame, period: int = 22) -> pd.Series:
    """Williams VIX Fix — synthetic implied-volatility proxy from OHLC.

    No external data required.  The VIX Fix approximates implied volatility
    by measuring how far the recent low is from the rolling highest close:
        vix_fix = (highest_close - Low) / highest_close × 100

    The z-score normalises across market regimes so XGBoost sees a
    stationary signal regardless of absolute price level.

    Returns a pd.Series named ``"synth_vix_zscore"``.
    """
    highest_close = df["Close"].rolling(period).max()
    vix_fix = (highest_close - df["Low"]) / highest_close * 100
    mean = vix_fix.rolling(20).mean()
    std  = vix_fix.rolling(20).std().replace(0, np.nan)
    return ((vix_fix - mean) / std).rename("synth_vix_zscore")


def load_usdchf_data(path: Path) -> pd.DataFrame | None:
    """Load the USDCHF master CSV and return a ``usdchf_log_return`` column.

    Returns ``None`` if the file does not exist — callers treat this as
    "USDCHF feature not available for this training run".

    Run ``python main.py --mode consolidate`` to build the master from MT5
    exports before training.
    """
    return load_asset_data(path, "usdchf_log_return")


def map_usdchf_to_bars(df_index: pd.DatetimeIndex, usdchf_df: pd.DataFrame) -> pd.Series:
    """Align USDCHF log returns onto XAUUSD bar timestamps.

    Thin wrapper around the generic ``map_asset_to_bars`` — kept for backwards
    compatibility with callers that reference it directly.
    """
    return map_asset_to_bars(df_index, usdchf_df, "usdchf_log_return")


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


def compute_gmm_vol_cluster(
    volatility: np.ndarray,
    n_components: int = 3,
    fitted_gmm=None,
    fitted_scaler=None,
    return_models: bool = False,
):
    """Cluster bars into Volatility Buckets (Low=0, Med=1, High=2) via GMM.

    Training mode  (fitted_gmm=None):
        Fit a StandardScaler then a GaussianMixture on non-NaN values.
        Remap component labels so 0=lowest-volatility, n-1=highest.
        Pass return_models=True to get (labels, gmm, scaler) back.

    Inference mode (fitted_gmm + fitted_scaler provided):
        Apply saved scaler.transform() then gmm.predict() — strictly no
        re-fitting.  This prevents cluster drift between training and live data.

    Falls back to quantile buckets (training mode only) when GMM collapses.
    """
    valid_mask = ~np.isnan(volatility)
    labels_out = np.zeros(len(volatility), dtype=np.int8)
    vol_valid  = volatility[valid_mask].reshape(-1, 1)

    if fitted_gmm is not None and fitted_scaler is not None:
        # ── Inference: use saved scaler + GMM, no fitting ──────────────────
        X = fitted_scaler.transform(vol_valid)
        raw_labels = fitted_gmm.predict(X)
        order = np.argsort(fitted_gmm.means_.ravel())
        remap = np.empty(fitted_gmm.n_components, dtype=int)
        for rank, orig in enumerate(order):
            remap[orig] = rank
        labels_valid = remap[raw_labels].astype(np.int8)
        labels_out[valid_mask] = labels_valid
        return labels_out

    # ── Training: fit scaler + GMM ──────────────────────────────────────────
    scaler = StandardScaler()
    X      = scaler.fit_transform(vol_valid)
    gmm    = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
    raw_labels = gmm.fit_predict(X)

    order = np.argsort(gmm.means_.ravel())
    remap = np.empty(n_components, dtype=int)
    for rank, orig in enumerate(order):
        remap[orig] = rank
    labels_valid = remap[raw_labels].astype(np.int8)

    if len(np.unique(labels_valid)) < n_components:
        logger.warning(
            "GMM collapsed to %d clusters (expected %d) — using quantile fallback.",
            len(np.unique(labels_valid)), n_components,
        )
        boundaries = np.percentile(vol_valid.ravel(),
                                   [100 / n_components * i for i in range(1, n_components)])
        labels_valid = np.digitize(vol_valid.ravel(), boundaries).astype(np.int8)

    labels_out[valid_mask] = labels_valid
    if return_models:
        return labels_out, gmm, scaler
    return labels_out


def process_pipeline(
    obs_cov: float = None,
    trans_cov: float = None,
    save: bool = True,
    tf: str = "H1",
    save_models: bool = False,
    broker: str = "headway_cent",
    lstm_model=None,
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
    if save_models:
        labels, _gmm, _scaler = compute_gmm_vol_cluster(
            df["volatility"].values, return_models=True
        )
        df["gmm_vol_cluster"] = labels
        save_gmm_model(_gmm, _scaler, tf=tf, broker=broker)
    else:
        df["gmm_vol_cluster"] = compute_gmm_vol_cluster(df["volatility"].values)

    # ── External cross-asset features ────────────────────────────────────────
    # Each asset uses the TF-matched master file so bar frequencies stay in sync.
    # Assets whose master file is absent degrade gracefully (column simply absent).
    # Build all masters with:  python main.py --mode consolidate
    for asset_key, path_by_tf in _EXTERNAL_ASSET_PATHS.items():
        col_name  = f"{asset_key}_log_return"
        path      = path_by_tf.get(tf, path_by_tf.get("H1"))
        asset_df  = load_asset_data(path, col_name)
        if asset_df is not None:
            ffilled_series, raw_series = map_asset_to_bars(df.index, asset_df, col_name, return_raw=True)
            df[f"{asset_key}_staleness"] = _compute_staleness(raw_series, col_name)
            df[col_name] = ffilled_series
            n_valid = df[col_name].notna().sum()
            logger.info(
                "%s [%s] merged: %d non-null %s values.",
                asset_key.upper(), tf, n_valid, col_name,
            )
        else:
            logger.debug(
                "%s master not found at %s — pipeline running without %s feature. "
                "Run  python main.py --mode consolidate  to enable it.",
                asset_key.upper(), path, col_name,
            )

    # Drop only rows where core features are NaN; preserve rows where external
    # assets have no history yet (e.g. XTIUSD starts Feb 2017, not Jan 2016).
    _core_cols = ["log_return", "kalman_return", "volatility", "rsi", "rsi_slope",
                  "atr_normalized", "gmm_vol_cluster"]
    df.dropna(subset=[c for c in _core_cols if c in df.columns], inplace=True)

    # Backfill external asset columns so short-history assets don't cause row loss.
    # bfill() propagates the first available value backward to fill pre-history rows.
    _ext_cols = [c for c in df.columns
                 if c.endswith("_log_return") or c.endswith("_staleness")]
    if _ext_cols:
        df[_ext_cols] = df[_ext_cols].bfill()

    # ── Synthetic VIX (Williams VIX Fix) ────────────────────────────────────
    df["synth_vix_zscore"] = compute_synth_vix(df)

    # ── Optional LSTM context features (lstm_ctx_0..3) ───────────────────────
    # Only added when a trained LSTMContextModel is supplied.  If lstm_model is
    # None the columns are absent and XGBoost trains without them (graceful
    # degradation).  Run --mode train_lstm to produce the model file first.
    if lstm_model is not None:
        from src.engine_lstm import add_lstm_context
        add_lstm_context(df, lstm_model)

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


def load_data_with_hmm_labels(tf: str, broker: str = "headway_cent") -> pd.DataFrame:
    """Load processed parquet and attach HMM state labels.

    Loads the saved HMM model for the given TF/broker and adds a
    ``hmm_state`` column so the LSTM regime classifier has targets to
    train on.

    Raises:
        FileNotFoundError: if the processed parquet or HMM model are missing.
    """
    tf  = tf.upper()
    cfg = TF_CONFIG.get(tf)
    if cfg is None:
        raise ValueError(f"Unknown timeframe '{tf}'")

    parquet_path = cfg["processed_path"]
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed parquet not found at {parquet_path}. "
            f"Run  python main.py --mode process --tf {tf}  first."
        )

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded processed data: %d rows [%s]", len(df), tf)

    if "hmm_state" in df.columns:
        logger.info("hmm_state already present in parquet — skipping HMM predict.")
        return df

    from src.engine_hmm import load_model as _load_hmm, predict_states, get_model_path, fit_hmm

    hmm_path = get_model_path(tf, broker)
    if hmm_path.exists():
        hmm_model = _load_hmm(hmm_path)
        logger.info("HMM model loaded from %s.", hmm_path)
    else:
        logger.warning(
            "HMM model not found at %s — fitting a fresh HMM with default params "
            "(%s n_states=%d) for LSTM label generation.  "
            "Run --mode train --tf %s --broker %s for optimised labels.",
            hmm_path, tf, TF_CONFIG[tf]["n_states_default"], tf, broker,
        )
        n_states = TF_CONFIG[tf]["n_states_default"]
        hmm_model, _, _ = fit_hmm(df, n_states=n_states)

    df["hmm_state"] = predict_states(hmm_model, df)
    logger.info("HMM states added: %s", dict(
        pd.Series(df["hmm_state"]).value_counts().sort_index()
    ))
    return df
