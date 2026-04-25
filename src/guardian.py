"""Multi-timeframe health monitor for Gold Regime X.

Periodically re-validates HMM+XGBoost signal quality on the most recently
synced MT5 data.  If rolling Sharpe for any timeframe drops below the alert
threshold a Telegram notification is fired immediately.

Also performs a daily LSTM staleness check: if a model is older than
LSTM_MAX_AGE_DAYS it is automatically retrained via a background subprocess.

Usage (via main.py):
    python main.py --mode guardian --tf M5,M15,H1 --period 3m --interval 3600
    (defaults: all three TFs, last 3 months of data, check every hour)

The check re-uses the same validator.run_validation() that gates live trading,
so the health score is directly comparable to sync_validate output.
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

from src.logger import setup_logger
from src.notifier import send_telegram_msg

logger = setup_logger(__name__)

SHARPE_ALERT_THRESHOLD = 0.6   # Telegram alert fires below this
SHARPE_CRITICAL        = 0.4   # escalated "CRITICAL" label
LSTM_MAX_AGE_DAYS      = 7     # Retrain LSTM if older than this
LSTM_CHECK_INTERVAL    = 86400  # Check LSTM staleness once per day (seconds)


# ── LSTM staleness helpers ─────────────────────────────────────────────────────

def _check_lstm_staleness(
    tfs: list,
    broker: str,
) -> Dict[str, Any]:
    """Return staleness info for each TF's LSTM model without loading weights."""
    import json
    from datetime import datetime, timezone

    results: Dict[str, Any] = {}
    for tf in tfs:
        lstm_dir = Path(f"models/lstm/{tf.upper()}_{broker}")

        # Determine which metadata file to read (ensemble takes precedence)
        meta_path = lstm_dir / "ensemble_metadata.json"
        if not meta_path.exists():
            meta_path = lstm_dir / "lstm_metadata.json"

        if not meta_path.exists():
            results[tf] = {"status": "not_found", "action": "train", "trained_at": None}
            continue

        try:
            meta       = json.loads(meta_path.read_text())
            trained_at = meta.get("trained_at")
            if not trained_at:
                results[tf] = {"status": "no_date", "action": "retrain", "trained_at": None}
                continue

            dt  = datetime.fromisoformat(trained_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - dt).days
            stale = age >= LSTM_MAX_AGE_DAYS

            results[tf] = {
                "status":     "stale" if stale else "fresh",
                "action":     "retrain" if stale else "none",
                "trained_at": trained_at,
                "age_days":   age,
            }
        except Exception as exc:
            results[tf] = {"status": "error", "action": "manual", "error": str(exc)}

    return results


def _auto_retrain_lstm(tf: str, broker: str) -> None:
    """Fire-and-forget: retrain LSTM for *tf* in a background subprocess."""
    cmd = [
        sys.executable, "main.py",
        "--mode", "train_lstm",
        "--tf",     tf,
        "--broker", broker,
        "--epochs", "50",
        "--fine_tune",
        "--recent_years", "2",
    ]
    logger.info("Auto-retraining LSTM [%s]: %s", tf, " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Wait up to 60 min for fine-tune to finish
        stdout, _ = proc.communicate(timeout=3600)
        if proc.returncode == 0:
            logger.info("LSTM [%s] auto-retrain complete.", tf)
            send_telegram_msg(f"LSTM [{tf}] auto-retrain complete.")
        else:
            logger.error("LSTM [%s] auto-retrain failed:\n%s", tf, stdout[-500:])
            send_telegram_msg(
                f"LSTM [{tf}] auto-retrain FAILED.\n"
                f"Run manually: python main.py --mode train_lstm --tf {tf} --broker {broker}"
            )
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error("LSTM [%s] auto-retrain timed out.", tf)
        send_telegram_msg(f"LSTM [{tf}] auto-retrain TIMEOUT — killed.")
    except Exception as exc:
        logger.error("LSTM [%s] auto-retrain exception: %s", tf, exc)


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

    cycle            = 0
    last_lstm_check  = 0.0   # epoch time of last LSTM staleness pass

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

        # ── LSTM staleness check (once per day) ───────────────────────────
        now = time.time()
        if now - last_lstm_check >= LSTM_CHECK_INTERVAL:
            logger.info("Running daily LSTM staleness check…")
            staleness = _check_lstm_staleness(tfs, broker)
            for tf, info in staleness.items():
                status = info["status"]
                if status == "fresh":
                    logger.info(
                        "LSTM [%s]: fresh (age=%d days, trained=%s)",
                        tf, info.get("age_days", "?"), info["trained_at"],
                    )
                elif status in ("stale", "no_date", "not_found"):
                    age_str = (
                        f"{info['age_days']} days old"
                        if "age_days" in info
                        else status
                    )
                    logger.warning("LSTM [%s] is %s — triggering auto-retrain.", tf, age_str)
                    send_telegram_msg(
                        f"LSTM [{tf}] is {age_str} — auto-retrain triggered."
                    )
                    _auto_retrain_lstm(tf, broker)
                elif status == "error":
                    logger.error("LSTM staleness check [%s] error: %s", tf, info.get("error"))
            last_lstm_check = now

        # ── Send periodic digest and any alerts ───────────────────────────
        send_telegram_msg("\n".join(lines))
        for alert in alerts:
            send_telegram_msg(alert)

        logger.info(
            "Guardian cycle #%d complete — next in %ds.", cycle, interval_sec
        )
        time.sleep(interval_sec)
