# ============================================================================
# Automatic MT5 Data Updater  (runs on EVERY fresh run of this notebook)
# ----------------------------------------------------------------------------
# Pulls the latest bars STRAIGHT FROM the running MT5 terminal and APPENDS them
# PERMANENTLY to the very same CSV files this notebook loads, written in each
# file's exact on-disk format so the loader cells parse them without loss.
#
# GAP-BASED refresh (not a fixed window): for each file and timeframe it reads
# the last timestamp already stored, then fetches only enough recent bars to
# bring that file up to *now* (today). Missing/empty files get an initial
# backfill instead. Every run is idempotent - bars are de-duplicated on
# timestamp (keep=last), so re-running never creates duplicates and always
# refreshes the latest (forming) bar.
#
# Assets & formats (only files THIS notebook is configured to read are touched):
#   * XAU (XAUUSD)  -> data/raw/XAU_{5m,15m}_data.csv
#                      semicolon; Date '%Y.%m.%d %H:%M'
#   * XAG (XAGUSD)  -> data/raw/XAGUSD_*.csv        (Explorer only)
#   * XTI (XTIUSD)  -> data/raw/XTIUSD_*.csv        (Explorer only)
#                      tab MT4/MT5 export; <DATE>'%Y.%m.%d' + <TIME>'%H:%M:%S'
#   * USDCHF        -> data/processed/USDCHF_master_{M15,M5}.csv  (Explorer only)
#                      comma; Date ISO '%Y-%m-%d %H:%M:%S' (data_consolidator format)
#
# Requires the MT5 terminal running & logged in. If MetaTrader5 is unavailable
# it prints a warning and continues on the existing CSVs so the pipeline never
# breaks. Toggle with AUTO_UPDATE_DATA below.
# ============================================================================

AUTO_UPDATE_DATA = True          # master switch; set False to skip the MT5 refresh
MT5_SYMBOLS = {                  # notebook asset key -> broker symbol in MT5
    "XAU": "XAUUSD",
    "XAG": "XAGUSD",
    "XTI": "XTIUSD",
    "USDCHF": "USDCHF",
}
# Gap-based sizing: pull only enough recent bars to catch each CSV up to *now*.
TF_MINUTES = {"M5": 5, "M15": 15, "H1": 60}
INITIAL_BACKFILL_BARS = {"M5": 4032, "M15": 1344, "H1": 336}  # used only if a file is missing/empty
FETCH_BUFFER_BARS = 5            # small overlap so the boundary bar is always refreshed
MAX_FETCH_BARS = 200000          # safety cap for very stale files

import pandas as pd
from pathlib import Path as _Path


def _resolve_target(path):
    """Resolve the CSV to the SAME location the notebook loaders read from.
    Checks cwd-relative first, then the parent dir (mirrors ST resolve_path so a
    notebook run from notebooks/ still appends to ../data/raw). If neither
    exists yet, returns the cwd-relative path for first-time creation."""
    path = _Path(path)
    if path.exists():
        return path
    alt = _Path.cwd().parent / path
    if alt.exists():
        return alt
    return path


def _read_last_ts(path, fmt):
    """Return the most recent timestamp already stored in the CSV (or None if the
    file is missing/empty/unparseable). Used to fetch ONLY the bars needed to
    catch the file up to now, per timeframe, instead of a fixed window. Parses
    each on-disk format with its exact date convention."""
    path = _Path(path)
    if not path.exists():
        return None
    try:
        if fmt == "xau_semicolon":
            df = pd.read_csv(path, sep=";", dtype=str)
            dcol = [c for c in df.columns if c.strip().lower() == "date"]
            if not dcol:
                return None
            ts = pd.to_datetime(df[dcol[0]], format="%Y.%m.%d %H:%M", errors="coerce")
        elif fmt == "mt4_tab":
            df = pd.read_csv(path, sep="\t", dtype=str)
            tags = {c.strip("<>").upper(): c for c in df.columns}
            if "DATE" in tags and "TIME" in tags:
                ts = pd.to_datetime(df[tags["DATE"]].astype(str) + " " + df[tags["TIME"]].astype(str),
                                    format="%Y.%m.%d %H:%M:%S", errors="coerce")
            elif "DATE" in tags:
                ts = pd.to_datetime(df[tags["DATE"]], format="%Y.%m.%d", errors="coerce")
            else:
                return None
        else:  # master_comma (data/processed/*_master_*.csv)
            df = pd.read_csv(path, dtype=str)
            if df.shape[1] == 0:
                return None
            ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        ts = ts.dropna()
        if len(ts) == 0:
            return None
        return ts.max()
    except Exception as exc:
        print("   [warn] could not read last timestamp from %s: %s" % (path.name, exc))
        return None


def _bars_needed(tf, last_dt, now, tf_minutes=None, buffer=5, cap=200000, initial=None):
    """How many recent bars to pull so the file is refreshed up to `now`.
    Sized from the gap between the CSV's last bar and now (never a fixed 2-week
    window). If the file is missing/empty, fall back to an initial backfill."""
    tfu = str(tf).upper()
    tf_minutes = tf_minutes or {"M5": 5, "M15": 15, "H1": 60}
    initial = initial or {"M5": 4032, "M15": 1344, "H1": 336}
    if last_dt is None:
        return int(initial.get(tfu, 1344))
    minutes = tf_minutes.get(tfu, 15)
    gap_min = (now - last_dt).total_seconds() / 60.0
    if gap_min <= 0:
        return int(buffer)
    n = int(gap_min // minutes) + int(buffer)
    return max(1, min(n, int(cap)))


def _fetch_mt5(mt5, symbol, tf, n_bars):
    """Pull the most recent n_bars for symbol/tf straight from MT5. Returns a
    DataFrame with a datetime 'dt' column plus raw OHLCV/spread, or None.
    Position-based (copy_rates_from_pos) so it is timezone-safe."""
    tf_const = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
    }.get(tf.upper())
    if tf_const is None:
        print("   [warn] unsupported timeframe %s" % tf)
        return None
    # Make sure the symbol is in Market Watch before requesting rates.
    try:
        if not mt5.symbol_select(symbol, True):
            print("   [warn] could not select %s in Market Watch: %s" % (symbol, mt5.last_error()))
    except Exception as exc:
        print("   [warn] symbol_select(%s) failed: %s" % (symbol, exc))
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, int(n_bars))
    if rates is None or len(rates) == 0:
        print("   [warn] MT5 returned no bars for %s %s: %s" % (symbol, tf, mt5.last_error()))
        return None
    r = pd.DataFrame(rates)
    r["dt"] = pd.to_datetime(r["time"], unit="s")
    return r


def _append_xau_semicolon(path, r):
    """Append MT5 bars to a semicolon XAU file, preserving its exact schema and
    the '%Y.%m.%d %H:%M' Date format the loaders parse. De-dupes on timestamp
    (keep=last so re-fetched bars refresh) and writes back in place."""
    path = _resolve_target(path)
    new = pd.DataFrame({
        "Date": r["dt"].dt.strftime("%Y.%m.%d %H:%M"),
        "Open": r["open"], "High": r["high"], "Low": r["low"], "Close": r["close"],
        "Volume": r["tick_volume"],
    })
    if "spread" in r.columns:
        new["Spread"] = r["spread"]

    if path.exists():
        existing = pd.read_csv(path, sep=";", dtype=str)
        cols = list(existing.columns)
        if "Date" not in [c for c in cols]:
            raise ValueError("Unexpected XAU schema (no Date column): %s" % cols)
        aligned = {}
        for c in cols:
            cl = c.strip().lower()
            if cl == "date":
                aligned[c] = new["Date"].astype(str)
            elif cl == "open":
                aligned[c] = new["Open"].astype(str)
            elif cl == "high":
                aligned[c] = new["High"].astype(str)
            elif cl == "low":
                aligned[c] = new["Low"].astype(str)
            elif cl == "close":
                aligned[c] = new["Close"].astype(str)
            elif cl in ("volume", "tickvol", "vol"):
                aligned[c] = new["Volume"].astype(str)
            elif cl == "spread":
                aligned[c] = (new["Spread"].astype(str) if "Spread" in new.columns
                              else pd.Series([""] * len(new)))
            else:
                aligned[c] = pd.Series([""] * len(new))
        new_aligned = pd.DataFrame(aligned)[cols]
        combined = pd.concat([existing, new_aligned], ignore_index=True)
    else:
        combined = new
        cols = list(new.columns)

    key = pd.to_datetime(combined["Date"], format="%Y.%m.%d %H:%M", errors="coerce")
    n_before = len(combined)
    combined = combined.assign(_k=key).dropna(subset=["_k"])
    combined = combined.drop_duplicates(subset=["_k"], keep="last").sort_values("_k")
    combined = combined.drop(columns=["_k"])[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, sep=";", index=False)
    return len(new), len(combined), n_before


def _append_mt4_tab(path, r):
    """Append MT5 bars to a tab-separated MT4/MT5-export file (<DATE>\t<TIME>...),
    preserving its exact bracketed headers and '%Y.%m.%d' / '%H:%M:%S' formats.
    The file must already exist (the notebook reads a specific export)."""
    path = _resolve_target(path)
    if not path.exists():
        print("   [warn] %s not found - skipping (expected a pre-existing MT4 export)." % path.name)
        return 0, 0, 0
    existing = pd.read_csv(path, sep="\t", dtype=str)
    cols = list(existing.columns)
    tags = [c.strip("<>").upper() for c in cols]
    if "DATE" not in tags:
        raise ValueError("Unexpected MT4 schema (no <DATE>): %s" % cols)
    has_real_volume = "real_volume" in r.columns
    build = {}
    for c in cols:
        t = c.strip("<>").upper()
        if t == "DATE":
            build[c] = r["dt"].dt.strftime("%Y.%m.%d")
        elif t == "TIME":
            build[c] = r["dt"].dt.strftime("%H:%M:%S")
        elif t == "OPEN":
            build[c] = r["open"].astype(str)
        elif t == "HIGH":
            build[c] = r["high"].astype(str)
        elif t == "LOW":
            build[c] = r["low"].astype(str)
        elif t == "CLOSE":
            build[c] = r["close"].astype(str)
        elif t == "TICKVOL":
            build[c] = r["tick_volume"].astype(str)
        elif t == "VOL":
            build[c] = (r["real_volume"].astype(str) if has_real_volume
                        else r["tick_volume"].astype(str))
        elif t == "SPREAD":
            build[c] = (r["spread"].astype(str) if "spread" in r.columns
                        else pd.Series([""] * len(r)))
        else:
            build[c] = pd.Series([""] * len(r))
    new = pd.DataFrame(build)[cols]
    combined = pd.concat([existing, new], ignore_index=True)

    dcol = cols[tags.index("DATE")]
    if "TIME" in tags:
        tcol = cols[tags.index("TIME")]
        key = pd.to_datetime(combined[dcol].astype(str) + " " + combined[tcol].astype(str),
                             format="%Y.%m.%d %H:%M:%S", errors="coerce")
    else:
        key = pd.to_datetime(combined[dcol], format="%Y.%m.%d", errors="coerce")
    combined = combined.assign(_k=key).dropna(subset=["_k"])
    combined = combined.drop_duplicates(subset=["_k"], keep="last").sort_values("_k")
    combined = combined.drop(columns=["_k"])[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, sep="\t", index=False)
    return len(new), len(combined), 0


def _append_master_comma(path, r):
    """Append MT5 bars to a data/processed/*_master_*.csv file in the EXACT
    format the consolidator writes: comma-separated, first column 'Date' in ISO
    '%Y-%m-%d %H:%M:%S', columns Date,Open,High,Low,Close,Volume. Used for the
    USDCHF masters. De-dupes on timestamp (keep=last) and writes back in place."""
    path = _resolve_target(path)
    new_vals = {
        "date": r["dt"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": r["open"].astype(str), "high": r["high"].astype(str),
        "low": r["low"].astype(str), "close": r["close"].astype(str),
        "volume": r["tick_volume"].astype(str),
    }
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        cols = list(existing.columns)
        if len(cols) == 0:
            raise ValueError("Empty master file: %s" % path)
        date_col = cols[0]
        aligned = {}
        for c in cols:
            cl = c.strip().lower()
            if c == date_col or cl in ("date", "datetime", "time"):
                aligned[c] = new_vals["date"]
            elif cl == "open":
                aligned[c] = new_vals["open"]
            elif cl == "high":
                aligned[c] = new_vals["high"]
            elif cl == "low":
                aligned[c] = new_vals["low"]
            elif cl == "close":
                aligned[c] = new_vals["close"]
            elif cl in ("volume", "tickvol", "vol"):
                aligned[c] = new_vals["volume"]
            else:
                aligned[c] = pd.Series([""] * len(r))
        new_df = pd.DataFrame(aligned)[cols]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        date_col = "Date"
        combined = pd.DataFrame({
            "Date": new_vals["date"], "Open": new_vals["open"], "High": new_vals["high"],
            "Low": new_vals["low"], "Close": new_vals["close"], "Volume": new_vals["volume"],
        })
    key = pd.to_datetime(combined[date_col], errors="coerce")
    combined = combined.assign(_k=key).dropna(subset=["_k"])
    combined = combined.drop_duplicates(subset=["_k"], keep="last").sort_values("_k")
    combined = combined.drop(columns=["_k"])[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, sep=",", index=False)
    return len(r), len(combined), 0


def _canonical_targets(g, default_symbols):
    """Build the (asset, tf, path, fmt) work list from whatever data paths THIS
    notebook defines, so we only ever touch files the notebook actually reads.
      * XAU     -> semicolon raw   (TF_TO_XAU_RAW, else M5_PATH/M15_PATH, else default)
      * XAG/XTI -> MT4 tab raw     (TF_TO_XAG_MASTER / TF_TO_XTI_MASTER if present)
      * USDCHF  -> processed comma (TF_TO_USDCHF_MASTER if present)
    """
    targets = []
    xau_map = {}
    if isinstance(g.get("TF_TO_XAU_RAW"), dict):
        xau_map = dict(g["TF_TO_XAU_RAW"])
    else:
        if "M5_PATH" in g:
            xau_map["M5"] = g["M5_PATH"]
        if "M15_PATH" in g:
            xau_map["M15"] = g["M15_PATH"]
    if not xau_map:
        xau_map = {"M5": _Path("data/raw/XAU_5m_data.csv"),
                   "M15": _Path("data/raw/XAU_15m_data.csv")}
    for tf, p in xau_map.items():
        targets.append(("XAU", tf, _Path(p), "xau_semicolon"))

    for asset, varname, fmt in [("XAG", "TF_TO_XAG_MASTER", "mt4_tab"),
                                ("XTI", "TF_TO_XTI_MASTER", "mt4_tab"),
                                ("USDCHF", "TF_TO_USDCHF_MASTER", "master_comma")]:
        m = g.get(varname)
        if isinstance(m, dict):
            for tf, p in m.items():
                targets.append((asset, tf, _Path(p), fmt))
    return targets

def _mt5_connect():
    """Initialise the MetaTrader5 bridge. Raises if the terminal is unavailable."""
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed: %s" % (mt5.last_error(),))
    return mt5

if not AUTO_UPDATE_DATA:
    print("[data-updater] AUTO_UPDATE_DATA=False - skipping MT5 refresh; using existing CSVs.")
else:
    try:
        _mt5 = _mt5_connect()
    except Exception as _exc:
        _mt5 = None
        print("[data-updater] MT5 unavailable (%s)." % _exc)
        print("               Using existing CSV files unchanged.")
    if _mt5 is not None:
        try:
            _now = pd.Timestamp.now()
            print("=" * 74)
            print("MT5 DATA UPDATE (gap-fill to latest)  -  %s" % _now.strftime("%Y-%m-%d %H:%M"))
            print("=" * 74)
            _targets = _canonical_targets(globals(), MT5_SYMBOLS)
            _touched = 0
            for _asset, _tf, _path, _fmt in _targets:
                _symbol = MT5_SYMBOLS.get(_asset, _asset)
                _resolved = _resolve_target(_path)
                _last = _read_last_ts(_resolved, _fmt)
                _n = _bars_needed(_tf, _last, _now, TF_MINUTES, FETCH_BUFFER_BARS,
                                  MAX_FETCH_BARS, INITIAL_BACKFILL_BARS)
                _r = _fetch_mt5(_mt5, _symbol, _tf, _n)
                if _r is None:
                    continue
                if _fmt == "xau_semicolon":
                    _added, _total, _ = _append_xau_semicolon(_path, _r)
                elif _fmt == "master_comma":
                    _added, _total, _ = _append_master_comma(_path, _r)
                else:
                    _added, _total, _ = _append_mt4_tab(_path, _r)
                _touched += 1
                _from = _last.strftime("%Y-%m-%d %H:%M") if _last is not None else "(new file)"
                print("  [ok] %-6s %-3s  from %-16s  +%d bars -> %d rows  (%s)"
                      % (_symbol, _tf, _from, _added, _total, _resolved.name))
            if _touched == 0:
                print("  (no assets updated - check MT5 symbols / Market Watch)")
        finally:
            try:
                _mt5.shutdown()
            except Exception:
                pass
        print("Data refresh complete. The loader cells below now read the updated CSVs.")