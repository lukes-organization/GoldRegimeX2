# -----------------------------
# Data loading helpers
# -----------------------------

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    if "open" in cols:
        rename_map[cols["open"]] = "Open"
    if "high" in cols:
        rename_map[cols["high"]] = "High"
    if "low" in cols:
        rename_map[cols["low"]] = "Low"
    if "close" in cols:
        rename_map[cols["close"]] = "Close"
    if "volume" in cols:
        rename_map[cols["volume"]] = "Volume"

    out = df.rename(columns=rename_map)
    need = ["Open", "High", "Low", "Close"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    return out

def read_xau_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=";")
    if "Date" not in df.columns:
        raise ValueError(f"Date column missing in {path}")

    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = _normalize_ohlcv(df)
    return df

def read_mt4_csv(path: Path) -> pd.DataFrame:
    """Read MetaTrader-exported CSV (tab-separated, <UPPERCASE> headers, date+time cols)."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="	")
    # Strip angle brackets from column names: <DATE> → DATE, <CLOSE> → CLOSE etc.
    df.columns = [c.strip("<>") for c in df.columns]
    # Combine DATE + TIME into a datetime index
    df["DateTime"] = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["TIME"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()
    # Normalise to title-case for _normalize_ohlcv compatibility
    rename_map = {}
    for tag in ["OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "VOL", "SPREAD"]:
        if tag in df.columns:
            rename_map[tag] = tag.capitalize()
    # Special: TICKVOL → Volume (since MT4 TICKVOL is the useful volume field)
    if "Tickvol" in df.columns:
        rename_map["Tickvol"] = "Volume"
    df = df.rename(columns=rename_map)
    return df

def read_master_close(path: Path) -> pd.Series:
    """Read commodity series — auto-detects XAU semicolon format vs MT4 tab format."""
    if not path.exists():
        raise FileNotFoundError(path)

    # Peek at first line to detect format
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()

    if "	" in header or header.strip().startswith("<"):
        # MT4 tab-separated format (XAG/XTI)
        df = read_mt4_csv(path)
        col_name = "Close" if "Close" in df.columns else "close"
        return df[col_name].astype(float).sort_index()
    else:
        # Delimited-with-header: XAU semicolon, or processed master (comma, ISO dates).
        sep = ";" if ";" in header else ","
        df = pd.read_csv(path, sep=sep, index_col=0, parse_dates=True).sort_index()
        if "Close" in df.columns:
            s = df["Close"].astype(float)
        elif "close" in df.columns:
            s = df["close"].astype(float)
        else:
            raise ValueError(f"Close column missing in {path}")
        return s

def load_panel(tf: str) -> pd.DataFrame:
    tf = tf.upper()
    xau = read_xau_raw(TF_TO_XAU_RAW[tf])

    out = xau.copy()

    # Cross-commodity series (uses TF_TO_XAG_MASTER / TF_TO_XTI_MASTER from config).
    xag_map = TF_TO_XAG_MASTER
    xti_map = TF_TO_XTI_MASTER

    if isinstance(xag_map, dict) and tf in xag_map and Path(xag_map[tf]).exists():
        xag = read_master_close(Path(xag_map[tf])).rename("XAG_Close")
        out["XAG_Close"] = xag.reindex(out.index).ffill()
    else:
        # Fallback keeps old feature functions from breaking.
        out["XAG_Close"] = out["Close"].astype(float)

    if isinstance(xti_map, dict) and tf in xti_map and Path(xti_map[tf]).exists():
        xti = read_master_close(Path(xti_map[tf])).rename("XTI_Close")
        out["XTI_Close"] = xti.reindex(out.index).ffill()
    else:
        out["XTI_Close"] = out["Close"].astype(float)

    # USDCHF (intraday DXY proxy, ~0.85 DXY correlation) - processed master.
    usdchf_map = globals().get("TF_TO_USDCHF_MASTER")
    if isinstance(usdchf_map, dict) and tf in usdchf_map and Path(usdchf_map[tf]).exists():
        usdchf = read_master_close(Path(usdchf_map[tf])).rename("USDCHF_Close")
        out["USDCHF_Close"] = usdchf.reindex(out.index).ffill()
    else:
        out["USDCHF_Close"] = out["Close"].astype(float)

    # Drop rows where cross-commodity data is NaN
    # (but only if we actually loaded real external data — skip if fallback was used
    #  to avoid dropping the entire early history where XAG/XTI don't exist yet)
    has_real_xag = isinstance(xag_map, dict) and tf in xag_map and Path(xag_map[tf]).exists()
    has_real_xti = isinstance(xti_map, dict) and tf in xti_map and Path(xti_map[tf]).exists()
    if has_real_xag or has_real_xti:
        # Only drop if BOTH external series are NaN — keep rows where at least one exists
        # Forward-fill already applied above; remaining NaN means no data at all for that period
        drop_mask = out["XAG_Close"].isna() & out["XTI_Close"].isna()
        if drop_mask.any():
            print(f"  [{tf}] Dropping {drop_mask.sum()} rows with no XAG or XTI data (before their start dates)")
        out = out[~drop_mask].copy()
        # Fill any remaining single-commodity NaNs with XAU Close as fallback
        out["XAG_Close"] = out["XAG_Close"].fillna(out["Close"].astype(float))
        out["XTI_Close"] = out["XTI_Close"].fillna(out["Close"].astype(float))

    # Keep USDCHF_Close gap-free (leading NaN before its first bar -> XAU Close).
    out["USDCHF_Close"] = out["USDCHF_Close"].fillna(out["Close"].astype(float))

    # --- MAX_DATA_YEARS filter: truncate to last N years ---
    max_years = MAX_DATA_YEARS
    if max_years is not None and isinstance(max_years, (int, float)) and max_years > 0:
        cutoff = out.index.max() - pd.DateOffset(years=int(max_years))
        before_count = len(out)
        out = out.loc[out.index >= cutoff].copy()
        print(f"  [{tf}] MAX_DATA_YEARS={max_years}: {before_count} → {len(out)} rows (cutoff {cutoff:%Y-%m-%d})")

    return out