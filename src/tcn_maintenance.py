"""Automatic TCN model maintenance — staleness checks and background retraining.

``TCNMaintenanceScheduler.run_maintenance_cycle()`` is called from the live/demo
trading loop once per hour.  It:

    1. Reads ``tcn_metadata.json`` for each TF/broker to check model age.
    2. If a model is ≥ 7 days old (or missing), launches a fine-tune subprocess.
    3. Also runs when ``WeeklyDataUpdater`` marks a fresh data pull.

All retraining is fire-and-forget (non-blocking subprocess) so the live
session is never interrupted.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.logger import setup_logger
from src.notifier import send_telegram_msg

logger = setup_logger(__name__)

TCN_MAX_AGE_DAYS = 7        # retrain if model is this many days old
_LAST_CHECK_FILE  = Path("models/tcn/.last_maintenance_check")
_RETRAIN_LOCK     = Path("models/tcn/.retrain_in_progress")
_CHECK_INTERVAL   = 3600    # minimum seconds between maintenance cycles (1 hour)


class TCNMaintenanceScheduler:
    """Hourly TCN health monitor and auto-retrainer.

    Args:
        broker:  Broker key (e.g. 'headway_cent').
        balance: Account balance — passed to ``--mode train_tcn``.
        tfs:     Timeframes to monitor (default: H1, M15, M5).
    """

    def __init__(
        self,
        broker: str  = "headway_cent",
        balance: float = 15.0,
        tfs: List[str] | None = None,
    ):
        self.broker  = broker
        self.balance = balance
        self.tfs     = tfs or ["H1", "M15", "M5"]

    # ── Staleness checks ──────────────────────────────────────────────────────

    def check_models(self) -> List[str]:
        """Return a list of timeframes whose TCN models are stale or missing."""
        stale: List[str] = []

        for tf in self.tfs:
            meta_path = Path(f"models/tcn/{tf.upper()}_{self.broker}/tcn_metadata.json")

            if not meta_path.exists():
                logger.info("TCN model for [%s] not found — needs training.", tf)
                stale.append(tf)
                continue

            try:
                meta       = json.loads(meta_path.read_text())
                trained_at = meta.get("trained_at")
                if not trained_at:
                    logger.warning("TCN [%s] metadata has no trained_at.", tf)
                    stale.append(tf)
                    continue

                dt = datetime.fromisoformat(trained_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - dt).days

                if age_days >= TCN_MAX_AGE_DAYS:
                    logger.info(
                        "TCN [%s] is stale: %d days old (limit %d).",
                        tf, age_days, TCN_MAX_AGE_DAYS,
                    )
                    stale.append(tf)
                else:
                    logger.debug("TCN [%s] is fresh: %d days old.", tf, age_days)

            except Exception as exc:
                logger.error("TCN staleness check [%s] failed: %s", tf, exc)
                stale.append(tf)

        return stale

    def should_retrain_after_data_update(self) -> bool:
        """Return True if data was updated since the last maintenance run."""
        update_marker = Path("data/raw/.last_auto_update")
        if not update_marker.exists():
            return False

        update_time = datetime.fromtimestamp(update_marker.stat().st_mtime)
        hours_since  = (datetime.now() - update_time).total_seconds() / 3600
        if hours_since >= 24:
            return False

        # Data was updated in the last 24 hours — check if TCN was retrained after
        if not _LAST_CHECK_FILE.exists():
            return True

        last_check_time = datetime.fromtimestamp(_LAST_CHECK_FILE.stat().st_mtime)
        return last_check_time < update_time

    # ── Retraining ────────────────────────────────────────────────────────────

    def trigger_retraining(self, tf: str) -> bool:
        """Launch TCN fine-tune for *tf* as a non-blocking background process."""
        # TF-specific epoch counts — H1 benefits from more epochs due to less data
        tf_epochs = {"H1": 30, "M15": 20, "M5": 15}
        epochs = tf_epochs.get(tf.upper(), 20)

        cmd = [
            sys.executable, "main.py",
            "--mode",         "train_tcn",
            "--tf",           tf.upper(),
            "--broker",       self.broker,
            "--balance",      str(self.balance),
            "--epochs",       str(epochs),
            "--fine_tune",
            "--recent_years", "2",
            "--temperature",  "1.5",
        ]
        logger.info("Auto-retraining TCN [%s]: %s", tf, " ".join(cmd))
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            return True
        except Exception as exc:
            logger.error("Failed to launch TCN retraining for [%s]: %s", tf, exc)
            return False

    # ── Maintenance cycle ─────────────────────────────────────────────────────

    def should_run_check(self) -> bool:
        """True if at least ``_CHECK_INTERVAL`` seconds have passed since last check."""
        if not _LAST_CHECK_FILE.exists():
            return True
        elapsed = time.time() - _LAST_CHECK_FILE.stat().st_mtime
        return elapsed >= _CHECK_INTERVAL

    def run_maintenance_cycle(self) -> None:
        """Full maintenance pass: check freshness, trigger retraining if needed."""
        # Stale lock guard (2-hour window)
        if _RETRAIN_LOCK.exists():
            age_sec = time.time() - _RETRAIN_LOCK.stat().st_mtime
            if age_sec < 7200:
                logger.debug("TCN maintenance: retraining already in progress.")
                return
            _RETRAIN_LOCK.unlink()   # stale lock — clean up

        try:
            stale_tfs = self.check_models()

            # Also retrain everything if data was recently updated
            if self.should_retrain_after_data_update():
                logger.info("Data update detected — scheduling TCN retrain for all TFs.")
                for tf in self.tfs:
                    if tf not in stale_tfs:
                        stale_tfs.append(tf)

            if not stale_tfs:
                logger.debug("All TCN models are fresh.")
                _LAST_CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
                _LAST_CHECK_FILE.touch()
                return

            logger.info("TCN models need retraining: %s", ", ".join(stale_tfs))

            _RETRAIN_LOCK.parent.mkdir(parents=True, exist_ok=True)
            _RETRAIN_LOCK.touch()

            for tf in stale_tfs:
                self.trigger_retraining(tf)

            send_telegram_msg(
                f"<b>TCN Auto-Retrain Triggered</b>\n"
                f"Timeframes: <code>{', '.join(stale_tfs)}</code>\n"
                f"Fine-tuning in background…"
            )

            _LAST_CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LAST_CHECK_FILE.touch()
            # Remove lock after all subprocesses are launched (they run independently)
            _RETRAIN_LOCK.unlink()

        except Exception as exc:
            logger.error("TCN maintenance cycle failed: %s", exc)
            if _RETRAIN_LOCK.exists():
                _RETRAIN_LOCK.unlink()
