"""USDCHF data consolidator.

Merges USDCHF raw CSV exports from ``data/raw/`` into sorted master files at
``data/processed/``.  One master file is produced per trading timeframe so
each training pipeline uses a USDCHF series whose bar frequency matches the
XAUUSD bars it is merged onto:

    USDCHF_master.csv     — H1  (source: USDCHF_H1.csv)
    USDCHF_master_M15.csv — M15 (source: USDCHF_M15_*.csv)
    USDCHF_master_M5.csv  — M5  (source: USDCHF_M5_*.csv)

The master files are used by the training pipeline as an intraday DXY proxy
(USDCHF correlates ~0.85 with DXY for XAUUSD signals and is always available
on Headway as a standard Forex pair).

Usage:
    python main.py --mode consolidate
    # or directly:
    python -c "from src.data_consolidator import consolidate_usdchf; consolidate_usdchf()"
"""

import pandas as pd
from pathlib import Path
from src.logger import setup_logger

logger = setup_logger(__name__)

RAW_DIR      = Path("data/raw")
MASTER_PATH  = Path("data/processed/USDCHF_master.csv")       # H1
MASTER_PATH_M15 = Path("data/processed/USDCHF_master_M15.csv")
MASTER_PATH_M5  = Path("data/processed/USDCHF_master_M5.csv")


def _read_usdchf_csv(f: Path) -> pd.DataFrame | None:
    """Read a single USDCHF CSV, auto-detecting delimiter and header presence.

    Handles two formats:
    - Named-column exports (``Date;Open;High;Low;Close;Volume`` header) from MT5
      History Center or the annual XTUP files downloaded with a ``MM/DD/YYYY`` date
    - Headerless MT5 bar exports (``USDCHF_H1.csv`` etc.) where the first column
      is a raw datetime string with no header row at all
    """
    try:
        first_line = f.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        sep = ";" if ";" in first_line else ","

        # If the first character is a digit the file has no header row —
        # MT5 "Save as CSV" exports often omit the header entirely.
        if first_line[0].isdigit():
            df = pd.read_csv(
                f, sep=sep, header=None,
                names=["Date", "Open", "High", "Low", "Close", "Volume"],
                parse_dates=["Date"],
            )
            df.set_index("Date", inplace=True)
            if pd.api.types.is_datetime64_any_dtype(df.index):
                logger.info("Loaded %d rows (headerless) from %s", len(df), f)
                return df
            logger.warning("Could not parse dates in headerless %s — skipping.", f)
            return None

        # Normal file with a header row
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


def _consolidate_files(files: list[Path], out_path: Path, label: str) -> pd.DataFrame:
    """Merge a list of USDCHF CSV files into a single sorted master CSV."""
    frames = []
    for f in files:
        df = _read_usdchf_csv(f)
        if df is not None:
            frames.append(df)

    if not frames:
        logger.error("All %s source files failed to load — no master produced.", label)
        return pd.DataFrame()

    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.sort_index(inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path)
    logger.info(
        "USDCHF %s master saved: %d rows -> %s  (source: %s)",
        label, len(merged), out_path, ", ".join(f.name for f in files),
    )
    return merged


def consolidate_usdchf(
    raw_dir:  Path = RAW_DIR,
    out_path: Path = MASTER_PATH,
) -> pd.DataFrame:
    """Build the H1 USDCHF master CSV from ``USDCHF_H1.csv``.

    Priority:
    1. ``USDCHF_H1.csv`` in *raw_dir* — the preferred source.  It is the
       hourly MT5 export that matches the H1 trading timeframe exactly and
       covers 2010-present.  If this file exists, it is used exclusively and
       all annual XTUP files are ignored.
    2. Falls back to scanning for any ``*USDCHF*.csv`` file in *raw_dir* so
       that legacy setups without ``USDCHF_H1.csv`` continue to work.
    """
    h1_file = raw_dir / "USDCHF_H1.csv"
    if h1_file.exists():
        files = [h1_file]
        logger.info("Using USDCHF_H1.csv as the sole USDCHF H1 source.")
    else:
        files = sorted(
            f for f in raw_dir.glob("*.csv")
            if "usdchf" in f.name.lower()
            and "m5" not in f.name.lower()
            and "m15" not in f.name.lower()
        )

    if not files:
        logger.warning(
            "No USDCHF H1 CSV found in %s. "
            "Export USDCHF H1 data from MT5 History Center and save as "
            "data/raw/USDCHF_H1.csv, then re-run --mode consolidate.",
            raw_dir,
        )
        return pd.DataFrame()

    return _consolidate_files(files, out_path, "H1")


def consolidate_usdchf_m15(
    raw_dir:  Path = RAW_DIR,
    out_path: Path = MASTER_PATH_M15,
) -> pd.DataFrame:
    """Build the M15 USDCHF master CSV from ``USDCHF_M15_*.csv``.

    Scans *raw_dir* for any file matching ``USDCHF_M15_*.csv`` (e.g.
    ``USDCHF_M15_201601040000_202603310000.csv`` exported from MT5 History
    Center).  Multiple matching files are merged and de-duplicated.
    """
    files = sorted(raw_dir.glob("USDCHF_M15_*.csv"))
    if not files:
        # Broader fallback: any csv with M15 in the name
        files = sorted(
            f for f in raw_dir.glob("*.csv")
            if "usdchf" in f.name.lower() and "m15" in f.name.lower()
        )

    if not files:
        logger.warning(
            "No USDCHF M15 CSV found in %s. "
            "Export USDCHF M15 data from MT5 and save as "
            "data/raw/USDCHF_M15_<dates>.csv, then re-run --mode consolidate.",
            raw_dir,
        )
        return pd.DataFrame()

    return _consolidate_files(files, out_path, "M15")


def consolidate_usdchf_m5(
    raw_dir:  Path = RAW_DIR,
    out_path: Path = MASTER_PATH_M5,
) -> pd.DataFrame:
    """Build the M5 USDCHF master CSV from ``USDCHF_M5_*.csv``.

    Scans *raw_dir* for any file matching ``USDCHF_M5_*.csv`` (e.g.
    ``USDCHF_M5_201601040000_202603310000.csv`` exported from MT5 History
    Center).  Multiple matching files are merged and de-duplicated.
    """
    files = sorted(raw_dir.glob("USDCHF_M5_*.csv"))
    if not files:
        files = sorted(
            f for f in raw_dir.glob("*.csv")
            if "usdchf" in f.name.lower() and "m5" in f.name.lower()
        )

    if not files:
        logger.warning(
            "No USDCHF M5 CSV found in %s. "
            "Export USDCHF M5 data from MT5 and save as "
            "data/raw/USDCHF_M5_<dates>.csv, then re-run --mode consolidate.",
            raw_dir,
        )
        return pd.DataFrame()

    return _consolidate_files(files, out_path, "M5")
