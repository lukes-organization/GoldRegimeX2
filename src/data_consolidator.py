"""USDCHF data consolidator.

Merges all USDCHF raw CSV exports from ``data/raw/`` into a single sorted
master file at ``data/processed/USDCHF_master.csv``.  The master file is used
by the training pipeline as an intraday DXY proxy (USDCHF correlates ~0.85
with DXY for XAUUSD signals and is always available on Headway as a standard
Forex pair).

Usage:
    python main.py --mode consolidate
    # or directly:
    python -c "from src.data_consolidator import consolidate_usdchf; consolidate_usdchf()"
"""

import pandas as pd
from pathlib import Path
from src.logger import setup_logger

logger = setup_logger(__name__)

RAW_DIR     = Path("data/raw")
MASTER_PATH = Path("data/processed/USDCHF_master.csv")


def _read_usdchf_csv(f: Path) -> pd.DataFrame | None:
    """Read a single USDCHF CSV, auto-detecting delimiter and date format."""
    try:
        # Detect separator from the first line
        header = f.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        sep = ";" if ";" in header else ","

        # Try parsing as-is first; fall back to infer_datetime_format
        for fmt in ["%m/%d/%Y", "%Y.%m.%d %H:%M", "%Y-%m-%d", None]:
            try:
                kwargs = dict(sep=sep, parse_dates=["Date"])
                if fmt:
                    kwargs["date_format"] = fmt
                df = pd.read_csv(f, **kwargs)
                df.set_index("Date", inplace=True)
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    logger.info("Loaded %d rows from %s", len(df), f)
                    return df
            except Exception:
                continue
        logger.warning("Could not parse dates in %s — skipping.", f)
        return None
    except Exception as exc:
        logger.warning("Failed to read %s: %s — skipping.", f, exc)
        return None



def consolidate_usdchf(
    raw_dir:  Path = RAW_DIR,
    out_path: Path = MASTER_PATH,
) -> pd.DataFrame:
    """Consolidate all USDCHF CSV files from *raw_dir* into a single master.

    Scans for files matching ``*USDCHF*.csv`` (case-insensitive) in *raw_dir*.
    Supports both comma-delimited annual exports (``MM/DD/YYYY`` dates) and
    semicolon-delimited MT5 exports (``YYYY.MM.DD HH:MM`` dates).  The format
    is auto-detected per file so mixed sources work together.

    Duplicate timestamps are de-duplicated and the result is sorted
    chronologically before saving.

    Args:
        raw_dir:  Directory to scan for USDCHF CSV exports.
        out_path: Destination for the consolidated master CSV.

    Returns:
        The consolidated DataFrame (also persisted to *out_path*).
    """
    # Case-insensitive glob for Windows and Linux
    files = sorted(
        f for f in raw_dir.glob("*.csv")
        if "usdchf" in f.name.lower()
    )

    if not files:
        logger.warning(
            "No USDCHF CSV files found in %s. "
            "Export USDCHF data from MT5 History Center (any timeframe that "
            "matches your trading TF) and place in data/raw/ with 'USDCHF' "
            "in the filename (e.g. USDCHF_5m_data.csv).",
            raw_dir,
        )
        return pd.DataFrame()

    frames = []
    for f in files:
        df = _read_usdchf_csv(f)
        if df is not None:
            frames.append(df)

    if not frames:
        logger.error("All USDCHF source files failed to load — no master produced.")
        return pd.DataFrame()

    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.sort_index(inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path)
    logger.info(
        "USDCHF master saved: %d rows -> %s  (from %d source file(s))",
        len(merged), out_path, len(files),
    )
    return merged
