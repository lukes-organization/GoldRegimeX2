"""Automatic weekly data updater — appends fresh MT5 bars to the raw CSV files.

Runs every Sunday during a live/demo session.  After a successful update,
``TCNMaintenanceScheduler`` in ``tcn_maintenance.py`` detects the refresh and
triggers automatic TCN fine-tuning.

The MT5 terminal must already be running and logged in before
``WeeklyDataUpdater.update_all_timeframes()`` is called (same requirement as the
live bridge).
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__)

# Raw CSV file paths per timeframe (semicolon-delimited, same format as MT5 export)
_TF_RAW_FILES = {
    "H1":  Path("data/raw/XAU_1h_data.csv"),
    "M15": Path("data/raw/XAU_m15_data.csv"),
    "M5":  Path("data/raw/XAU_5m_data.csv"),
}

# How many recent bars to fetch from MT5 each update cycle
_FETCH_BARS = {
    "H1":  336,   # ~2 weeks of H1 bars
    "M15": 1344,  # ~2 weeks of M15 bars
    "M5":  4032,  # ~2 weeks of M5 bars
}

_LAST_UPDATE_FILE = Path("data/raw/.last_auto_update")
_UPDATE_LOCK_FILE = Path("data/raw/.update_in_progress")


class WeeklyDataUpdater:
    """Pull fresh XAUUSD bars from MT5 and append them to the raw training CSVs.

    Schedule:
        - Runs automatically every Sunday when ``update_all_timeframes()``
          is called from the live/demo trading loop.
        - A lock file prevents concurrent runs.
        - A timestamp file prevents re-running within the same week.
    """

    def should_update(self) -> bool:
        """Return True if today is Sunday and we haven't updated this week."""
        today = datetime.now()

        if today.weekday() != 6:      # 6 = Sunday
            return False

        # Lock guard
        if _UPDATE_LOCK_FILE.exists():
            age_sec = (today - datetime.fromtimestamp(
                _UPDATE_LOCK_FILE.stat().st_mtime
            )).total_seconds()
            if age_sec < 3600:        # lock valid for 1 hour
                logger.debug("Data update: lock file active — skipping.")
                return False

        # Already updated this week?
        if _LAST_UPDATE_FILE.exists():
            try:
                last = datetime.strptime(
                    _LAST_UPDATE_FILE.read_text().strip(), "%Y-%m-%d"
                )
                if (today - last).days < 6:
                    logger.debug(
                        "Data update: last run %d days ago — skipping.",
                        (today - last).days,
                    )
                    return False
            except ValueError:
                pass  # Corrupted timestamp — proceed

        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _create_lock() -> None:
        _UPDATE_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        _UPDATE_LOCK_FILE.touch()

    @staticmethod
    def _release_lock() -> None:
        if _UPDATE_LOCK_FILE.exists():
            _UPDATE_LOCK_FILE.unlink()

    @staticmethod
    def _mark_updated() -> None:
        _LAST_UPDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_UPDATE_FILE.write_text(datetime.now().strftime("%Y-%m-%d"))

    def _fetch_bars(self, tf: str) -> pd.DataFrame | None:
        """Fetch recent bars from MT5 for *tf* and return as a semicolon-format df."""
        try:
            import MetaTrader5 as mt5
            from src.mt5_sync import _get_tf_map

            tf_map  = _get_tf_map()
            mt5_tf  = tf_map.get(tf.upper())
            if mt5_tf is None:
                logger.warning("Unknown timeframe '%s' for MT5 fetch.", tf)
                return None

            n_bars = _FETCH_BARS.get(tf.upper(), 336)
            rates  = mt5.copy_rates_from_pos("XAUUSD", mt5_tf, 0, n_bars)
            if rates is None or len(rates) == 0:
                logger.warning("MT5 returned no bars for %s: %s", tf, mt5.last_error())
                return None

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
            return df

        except Exception as exc:
            logger.error("MT5 fetch failed for %s: %s", tf, exc)
            return None

    def _update_single(self, tf: str) -> bool:
        """Fetch and append new bars for one timeframe.  Returns True on success."""
        raw_path = _TF_RAW_FILES.get(tf.upper())
        if raw_path is None:
            return False

        logger.info("  Updating %s → %s", tf, raw_path)
        new_df = self._fetch_bars(tf)
        if new_df is None or len(new_df) == 0:
            logger.warning("  No data received for %s.", tf)
            return False

        try:
            if raw_path.exists():
                existing = pd.read_csv(
                    raw_path, sep=";",
                    parse_dates=["Date"],
                )
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["Date"], keep="last")
                combined = combined.sort_values("Date").reset_index(drop=True)
            else:
                combined = new_df.sort_values("Date").reset_index(drop=True)

            raw_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(raw_path, sep=";", index=False)

            prev_len = len(
                pd.read_csv(raw_path, sep=";", usecols=["Date"])
            ) if raw_path.exists() else 0
            new_bars = len(combined) - prev_len
            logger.info(
                "  ✓ %s: %d total bars  (%+d new)",
                tf, len(combined), new_bars,
            )
            return True

        except Exception as exc:
            logger.error("  ✗ Failed to write %s CSV: %s", tf, exc)
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def update_all_timeframes(self) -> bool:
        """Pull latest bars for H1, M15, M5 and append to their raw CSVs.

        Returns True if at least one timeframe was updated successfully.
        Only executes on Sundays when MT5 is connected.
        """
        if not self.should_update():
            return False

        self._create_lock()
        try:
            logger.info("=" * 60)
            logger.info(
                "WEEKLY DATA UPDATE — %s",
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
            logger.info("=" * 60)

            successes = sum(
                self._update_single(tf) for tf in _TF_RAW_FILES
            )

            if successes > 0:
                self._mark_updated()
                logger.info(
                    "Weekly data update complete: %d/%d timeframes updated.",
                    successes, len(_TF_RAW_FILES),
                )
                return True

            logger.error("Weekly data update: no timeframes updated.")
            return False

        except Exception as exc:
            logger.error("Weekly data update failed: %s", exc)
            return False
        finally:
            self._release_lock()
