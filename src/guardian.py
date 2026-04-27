"""Multi-timeframe health monitor for Gold Regime X.

Periodically re-validates HMM+XGBoost signal quality on the most recently
synced MT5 data.  If rolling Sharpe for any timeframe drops below the alert
threshold a Telegram notification is fired immediately.

Also performs a daily TCN staleness check: if a model is older than
TCN_MAX_AGE_DAYS it is automatically retrained via a background subprocess.

Usage (via main.py):
    python main.py --mode guardian --tf M5,M15,H1 --period 3m --interval 3600
    (defaults: all three TFs, last 3 months of data, check every hour)

The check re-uses the same validator.run_validation() that gates live trading,
so the health score is directly comparable to sync_validate output.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

from src.logger import setup_logger
from src.notifier import send_telegram_msg

logger = setup_logger(__name__)

SHARPE_ALERT_THRESHOLD = 0.6   # Telegram alert fires below this
SHARPE_CRITICAL        = 0.4   # escalated "CRITICAL" label
TCN_CHECK_INTERVAL     = 86400  # Check TCN staleness once per day (seconds)


# ── TCN staleness helpers ──────────────────────────────────────────────────────

def _check_tcn_staleness(tfs: list, broker: str) -> Dict[str, Any]:
    """Return staleness info for each TF's TCN model without loading weights."""
    from src.tcn_maintenance import TCNMaintenanceScheduler

    scheduler = TCNMaintenanceScheduler(broker=broker, tfs=tfs)
    stale_tfs = scheduler.check_models()

    results: Dict[str, Any] = {}
    for tf in tfs:
        import json
        from datetime import datetime, timezone
        meta_path = Path(f"models/tcn/{tf.upper()}_{broker}/tcn_metadata.json")
        if tf in stale_tfs:
            if not meta_path.exists():
                results[tf] = {"status": "not_found", "action": "train", "trained_at": None}
            else:
                try:
                    meta = json.loads(meta_path.read_text())
                    trained_at = meta.get("trained_at")
                    if trained_at:
                        dt  = datetime.fromisoformat(trained_at)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        age = (datetime.now(timezone.utc) - dt).days
                        results[tf] = {
                            "status": "stale", "action": "retrain",
                            "trained_at": trained_at, "age_days": age,
                        }
                    else:
                        results[tf] = {"status": "no_date", "action": "retrain",
                                       "trained_at": None}
                except Exception as exc:
                    results[tf] = {"status": "error", "action": "manual",
                                   "error": str(exc)}
        else:
            try:
                meta = json.loads(meta_path.read_text())
                trained_at = meta.get("trained_at")
                if trained_at:
                    from datetime import datetime, timezone
                    dt  = datetime.fromisoformat(trained_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - dt).days
                    results[tf] = {
                        "status": "fresh", "action": "none",
                        "trained_at": trained_at, "age_days": age,
                    }
                else:
                    results[tf] = {"status": "fresh", "action": "none",
                                   "trained_at": None, "age_days": 0}
            except Exception:
                results[tf] = {"status": "fresh", "action": "none",
                               "trained_at": None, "age_days": 0}

    return results


def _auto_retrain_tcn(tf: str, broker: str) -> None:
    """Fire-and-forget: retrain TCN for *tf* in a background subprocess."""
    from src.tcn_maintenance import TCNMaintenanceScheduler
    scheduler = TCNMaintenanceScheduler(broker=broker)
    scheduler.trigger_retraining(tf)


# ── Main guardian loop ────────────────────────────────────────────────────────

def run_guardian(
    tfs: list,
    broker: str,
    account_size: float,
    period: str = "3m",
    interval_sec: int = 3600,
) -> None:
    """Monitor timeframe health continuously; alert via Telegram on degradation.

    Args:
        tfs:           List of timeframe strings, e.g. ["M5", "M15", "H1"].
        broker:        Broker key used for backtest cost simulation.
        account_size:  USD balance for lot-sizing in the backtest.
        period:        MT5 lookback period for synced data ('3m', '6m', etc.).
        interval_sec:  Seconds between health checks.  Default 3600 (1 hour).
    """
    from src.validator import run_validation

    logger.info(
        "Guardian started — TFs=%s  interval=%ds  alert_threshold=%.1f",
        tfs, interval_sec, SHARPE_ALERT_THRESHOLD,
    )
    send_telegram_msg(
        f"<b>Guardian online</b>\n"
        f"Monitoring: <b>{', '.join(tfs)}</b>\n"
        f"Check interval: every {interval_sec // 60} min"
    )

    cycle           = 0
    last_tcn_check  = 0.0   # epoch time of last TCN staleness pass

    while True:
        cycle += 1
        lines   = [f"<b>Guardian Health Check #{cycle}</b>"]
        alerts  = []

        # ── HMM/XGBoost health check (every cycle) ────────────────────────
        for tf in tfs:
            try:
                result = run_validation(
                    tf=tf,
                    broker=broker,
                    account_size=account_size,
                    period=period,
                )
                sharpe = result["sharpe"]
                status = result["status"]
                trades = result["n_trades"]

                if sharpe < SHARPE_CRITICAL:
                    icon = "CRITICAL"
                elif sharpe < SHARPE_ALERT_THRESHOLD:
                    icon = "WARN"
                elif status == "pass":
                    icon = "OK"
                else:
                    icon = "WARN"

                lines.append(
                    f"  [{tf}] Sharpe={sharpe:.3f}  "
                    f"Trades={trades}  [{icon}]"
                )
                logger.info(
                    "Guardian [%s]: sharpe=%.3f  status=%s  trades=%d",
                    tf, sharpe, status, trades,
                )

                if sharpe < SHARPE_ALERT_THRESHOLD:
                    level = "CRITICAL" if sharpe < SHARPE_CRITICAL else "WARNING"
                    alerts.append(
                        f"<b>GUARDIAN {level} — [{tf}]</b>\n"
                        f"Rolling Sharpe: <b>{sharpe:.3f}</b> "
                        f"(threshold: {SHARPE_ALERT_THRESHOLD})\n"
                        f"Action: python main.py --mode optimize --tf {tf}"
                    )
            except Exception as exc:
                msg = f"  [{tf}] Check error: {exc}"
                lines.append(msg)
                logger.error("Guardian [%s] check failed: %s", tf, exc)

        # ── TCN staleness check (once per day) ────────────────────────────
        now = time.time()
        if now - last_tcn_check >= TCN_CHECK_INTERVAL:
            logger.info("Running daily TCN staleness check…")
            staleness = _check_tcn_staleness(tfs, broker)
            for tf, info in staleness.items():
                status = info["status"]
                if status == "fresh":
                    logger.info(
                        "TCN [%s]: fresh (age=%s days, trained=%s)",
                        tf, info.get("age_days", "?"), info.get("trained_at"),
                    )
                elif status in ("stale", "no_date", "not_found"):
                    age_str = (
                        f"{info['age_days']} days old"
                        if "age_days" in info
                        else status
                    )
                    logger.warning(
                        "TCN [%s] is %s — triggering auto-retrain.", tf, age_str
                    )
                    send_telegram_msg(
                        f"TCN [{tf}] is {age_str} — auto-retrain triggered."
                    )
                    _auto_retrain_tcn(tf, broker)
                elif status == "error":
                    logger.error(
                        "TCN staleness check [%s] error: %s",
                        tf, info.get("error"),
                    )
            last_tcn_check = now

        # ── Send periodic digest and any alerts ───────────────────────────
        send_telegram_msg("\n".join(lines))
        for alert in alerts:
            send_telegram_msg(alert)

        logger.info(
            "Guardian cycle #%d complete — next in %ds.", cycle, interval_sec
        )
        time.sleep(interval_sec)
