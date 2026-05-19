"""MT5 Data Sync — downloads recent OHLCV bars from the MetaTrader5 terminal.

The MT5 terminal must already be running and logged into the desired account
before calling any function here.  The MetaTrader5 Python package is imported
lazily so this module can be imported on machines that do not have MT5 installed
(e.g. a CI environment that only runs the backtest pipeline).
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from src.logger import setup_logger

logger = setup_logger(__name__)

SYNC_OUTPUT_PATH   = Path("data/processed/mt5_sync_data.csv")
DEFAULT_SYMBOL     = "XAUUSD"
DXY_SYMBOL_ALIASES = ["DXY", "USDX", "DOLLAR", "DXYUSD"]   # broker-specific names for USD Index

# Lazy MT5 timeframe map — populated on first call to _get_tf_map()
_MT5_TF_MAP: dict | None = None


def _get_tf_map() -> dict:
    global _MT5_TF_MAP
    if _MT5_TF_MAP is None:
        import MetaTrader5 as mt5
        _MT5_TF_MAP = {
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1":  mt5.TIMEFRAME_H1,
        }
    return _MT5_TF_MAP


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_period(period_str: str) -> int:
    """Convert a period string such as ``'3m'`` to a month count integer."""
    period_str = period_str.strip().lower()
    if period_str.endswith("m") and period_str[:-1].isdigit():
        return int(period_str[:-1])
    raise ValueError(
        f"Unrecognised period format: '{period_str}'. "
        "Expected a digit followed by 'm', e.g. '3m', '6m', '12m'."
    )


def connect_mt5(login: int = None, password: str = None, server: str = None) -> bool:
    """Initialise the MT5 package and optionally log in programmatically.

    If *login* is None the function relies on the account that is already
    active in the terminal.  Returns ``True`` on success.
    """
    import MetaTrader5 as mt5
    if not mt5.initialize():
        logger.error("MT5 initialize() failed: %s", mt5.last_error())
        return False
    if login is not None:
        if not mt5.login(login, password=password, server=server):
            logger.error("MT5 login(%d) failed: %s", login, mt5.last_error())
            mt5.shutdown()
            return False
    info = mt5.account_info()
    if info:
        logger.info(
            "MT5 connected — login=%d  server=%s  balance=%.2f %s",
            info.login, info.server, info.balance, info.currency,
        )
    return True


def disconnect_mt5() -> None:
    """Shut down the MT5 Python connection (safe to call when not connected)."""
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
        logger.debug("MT5 disconnected.")
    except Exception:
        pass


def fetch_bars(symbol: str, tf: str, months: int) -> pd.DataFrame:
    """Download completed OHLCV bars for *symbol* on *tf* going back *months*.

    The currently open (incomplete) bar is always excluded.

    Returns a DataFrame with columns ``Open, High, Low, Close, Volume`` and a
    UTC DatetimeIndex named ``Date`` — matching the convention expected by the
    standalone feature-engineering functions in ``processor.py``.
    """
    import MetaTrader5 as mt5
    from dateutil.relativedelta import relativedelta

    tf_map = _get_tf_map()
    tf_key = tf.upper()
    if tf_key not in tf_map:
        raise ValueError(f"Unknown timeframe '{tf}'. Supported: {list(tf_map)}")

    date_from = datetime.utcnow() - relativedelta(months=months)
    date_to   = datetime.utcnow()

    rates = mt5.copy_rates_range(symbol, tf_map[tf_key], date_from, date_to)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"MT5 returned no data for {symbol} {tf}: {mt5.last_error()}\n"
            "Ensure the symbol is in Market Watch and the terminal is connected."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = (
        df.rename(columns={
            "time":        "Date",
            "open":        "Open",
            "high":        "High",
            "low":         "Low",
            "close":       "Close",
            "tick_volume": "Volume",
        })
        [["Date", "Open", "High", "Low", "Close", "Volume"]]
    )
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.iloc[:-1]  # drop the currently open bar

    logger.info(
        "Fetched %d %s bars for %s: %s -> %s",
        len(df), tf_key, symbol, df.index.min(), df.index.max(),
    )
    return df


def _find_dxy_symbol(mt5) -> str | None:
    """Return the first DXY-equivalent symbol available on the connected broker.

    Tries each alias in ``DXY_SYMBOL_ALIASES`` in order.  If the symbol is not
    visible in Market Watch, attempts to add it first.  Returns ``None`` if no
    DXY equivalent can be found.
    """
    for alias in DXY_SYMBOL_ALIASES:
        info = mt5.symbol_info(alias)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(alias, True)
            return alias
    return None


def fetch_cross_asset_bars(
    tf: str = "H1",
    months: int = 3,
) -> pd.DataFrame:
    """Fetch XAUUSD and DXY bars, align on shared timestamps, return merged DataFrame.

    The returned DataFrame has all XAUUSD OHLCV columns plus ``dxy_close``.
    Bars where either symbol has no data are dropped (inner join).  If no DXY
    symbol can be found on the broker the function falls back to XAUUSD-only
    data with a warning.

    Args:
        tf:     Timeframe string — ``"H1"``, ``"M15"``, or ``"M5"``.
        months: Number of months of history to fetch.
    """
    import MetaTrader5 as mt5

    xau_df = fetch_bars(DEFAULT_SYMBOL, tf, months)

    dxy_sym = _find_dxy_symbol(mt5)
    if dxy_sym is None:
        logger.warning(
            "No DXY symbol found on broker (tried %s). "
            "Returning XAUUSD-only data — dxy_log_return will not be available.",
            DXY_SYMBOL_ALIASES,
        )
        return xau_df

    try:
        dxy_df = fetch_bars(dxy_sym, tf, months)
    except Exception as exc:
        logger.warning("DXY fetch failed (%s): %s — returning XAUUSD-only data.", dxy_sym, exc)
        return xau_df

    merged = xau_df.join(dxy_df[["Close"]].rename(columns={"Close": "dxy_close"}), how="inner")
    n_dropped = len(xau_df) - len(merged)
    if n_dropped:
        logger.info("Cross-asset align: dropped %d bars with missing DXY data.", n_dropped)

    logger.info(
        "Cross-asset merge complete: %d bars [%s + %s]",
        len(merged), DEFAULT_SYMBOL, dxy_sym,
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Primary entry point
# ─────────────────────────────────────────────────────────────────────────────

# Raw CSV path mapping — mirrors TF_CONFIG["raw_path"] in processor.py.
# Used by ensure_data_updated to know which file to append into.
_RAW_CSV_PATHS: dict[str, Path] = {
    "H1":  Path("data/raw/XAU_1h_data.csv"),
    "M15": Path("data/raw/XAU_15m_data.csv"),
    "M5":  Path("data/raw/XAU_5m_data.csv"),
}


def ensure_data_updated(tf: str, symbol: str = "XAUUSD") -> None:
    """Check if the local raw CSV for *tf* is stale; if so, fetch and append
    the missing bars from MT5.

    The function is intentionally non-fatal: if MT5 is unavailable (e.g. the
    terminal is not running) it logs a warning and returns so the rest of the
    pipeline can continue with existing data.

    The raw CSV format is semicolon-delimited with header:
        Date;Open;High;Low;Close;Volume
    where Date is ``%Y.%m.%d %H:%M`` (matching what ``load_raw_data`` expects).
    """
    from datetime import datetime, timezone

    tf_key = tf.upper()
    file_path = _RAW_CSV_PATHS.get(tf_key)
    if file_path is None:
        logger.warning("[SYNC] Unknown TF '%s' — skipping data update check.", tf)
        return
    if not file_path.exists():
        logger.warning("[SYNC] Raw CSV not found at %s — skipping.", file_path)
        return

    # Read last known timestamp
    try:
        df_existing = pd.read_csv(
            file_path, sep=";", parse_dates=["Date"], date_format="%Y.%m.%d %H:%M",
        )
        df_existing.set_index("Date", inplace=True)
        df_existing.sort_index(inplace=True)
        last_dt_naive = df_existing.index[-1].to_pydatetime()
    except Exception as exc:
        logger.warning("[SYNC] Could not read %s: %s — skipping.", file_path, exc)
        return

    last_dt_utc = last_dt_naive.replace(tzinfo=timezone.utc)
    utc_now = datetime.now(timezone.utc)

    # Skip if data is younger than one bar interval
    _tf_seconds = {"H1": 3600, "M15": 900, "M5": 300}
    lag_s = (utc_now - last_dt_utc).total_seconds()
    if lag_s < _tf_seconds.get(tf_key, 3600):
        logger.info(
            "[SYNC] %s data is fresh (lag=%.0f s). No update needed.", tf_key, lag_s,
        )
        return

    logger.info("[SYNC] %s data is stale by %.0f s — fetching from MT5.", tf_key, lag_s)

    # Connect and fetch
    if not connect_mt5():
        logger.warning("[SYNC] MT5 unavailable — proceeding with existing data.")
        return

    try:
        import MetaTrader5 as mt5
        tf_map = _get_tf_map()
        rates = mt5.copy_rates_range(symbol, tf_map[tf_key], last_dt_utc, utc_now)
    finally:
        disconnect_mt5()

    if rates is None or len(rates) == 0:
        logger.info("[SYNC] MT5 returned no new bars for %s %s.", symbol, tf_key)
        return

    df_new = pd.DataFrame(rates)
    df_new["time"] = pd.to_datetime(df_new["time"], unit="s", utc=True)
    df_new.set_index("time", inplace=True)
    df_new.index = df_new.index.tz_localize(None)  # strip tz — CSV has naive datetimes

    # Drop the currently open (incomplete) bar
    df_new = df_new.iloc[:-1]

    # Keep only rows strictly newer than the last known bar
    df_new = df_new[df_new.index > last_dt_naive]

    if len(df_new) == 0:
        logger.info("[SYNC] %s %s is already up to date.", symbol, tf_key)
        return

    # Rename to match CSV column convention
    df_new = df_new.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume",
    })
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_new.columns]
    df_new = df_new[cols]
    df_new.index.name = "Date"

    # Append — header=False so column names are not written again; sep matches original
    df_new.to_csv(file_path, mode="a", header=False, sep=";", date_format="%Y.%m.%d %H:%M")
    logger.info("[SYNC] Appended %d new %s bars to %s.", len(df_new), tf_key, file_path)


def sync_mt5_data(
    symbol: str = DEFAULT_SYMBOL,
    tf: str = "H1",
    period: str = "3m",
    output_path: Path = SYNC_OUTPUT_PATH,
) -> pd.DataFrame:
    """Connect to MT5, fetch recent bars, save a CSV, then disconnect.

    Raises ``ConnectionError`` when the MT5 terminal cannot be reached.
    """
    if not connect_mt5():
        raise ConnectionError(
            "Could not connect to MetaTrader5 terminal. "
            "Ensure MT5 is running and logged into your account."
        )
    try:
        months = parse_period(period)
        df = fetch_bars(symbol, tf.upper(), months)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info("Saved %d bars -> %s", len(df), output_path)
        return df
    finally:
        disconnect_mt5()
