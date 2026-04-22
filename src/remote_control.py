"""Telegram remote control panel for Gold Regime X.

Runs a long-polling Telegram bot that accepts keyboard button commands from
the authorised user and dispatches them as subprocesses so the listener
thread is never blocked.

Keyboard layout:
    ┌──────────────┬──────────────┬──────────────┐
    │  🚀 START M5 │ 🚀 START M15 │ 🚀 START H1  │
    ├──────────────┴──────┬───────┴──────────────┤
    │  🛑 STOP TRADING    │  📉 OPTIMIZE M5       │
    ├─────────────────────┴──────────────────────┤
    │              📊 BOT STATUS                  │
    └────────────────────────────────────────────┘

Required env vars (see .env.example):
    TELEGRAM_BOT_TOKEN   — BotFather token
    TELEGRAM_CHAT_ID     — Your chat ID (used for outbound heartbeat messages)
    ALLOWED_USER_ID      — Your numeric Telegram user ID (security gate)
    LIVE_BROKER          — Default broker  (default: headway_cent)
    LIVE_BALANCE         — Default balance (default: 15)

Usage:
    python main.py --mode listen
"""

import os
import subprocess
import time

import requests
from requests.exceptions import ReadTimeout

from src.logger import setup_logger
from src.notifier import get_credentials, send_telegram_msg

logger = setup_logger(__name__)

# ── Persistent reply keyboard ──────────────────────────────────────────────────
# resize_keyboard=True  → compact size on mobile
# one_time_keyboard=False → stays visible after every tap
_KEYBOARD = {
    "keyboard": [
        ["🚀 START M5",    "🚀 START M15",             "🚀 START H1"],
        ["🛑 STOP TRADING", "📉 OPTIMIZE M5 (0.50-0.55)", "📊 BOT STATUS"],
    ],
    "resize_keyboard":   True,
    "one_time_keyboard": False,
}

# Tracks subprocesses launched by this session so we can terminate them
_procs: dict[str, subprocess.Popen] = {}


def _api(token: str, method: str, _req_timeout=(10, 15), **params) -> dict:
    """Call a Telegram Bot API method; returns the parsed JSON response.

    ``_req_timeout`` is passed to requests as (connect_timeout, read_timeout)
    and is intentionally NOT forwarded to Telegram — hence the underscore prefix.
    Long-polling calls should supply a read timeout > the Telegram ``timeout``
    parameter to avoid spurious ReadTimeout exceptions.
    """
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/{method}",
            json=params,
            timeout=_req_timeout,
        )
        return r.json()
    except ReadTimeout:
        # Expected for getUpdates when no new messages arrive within the poll
        # window — Telegram returns an empty result list, which is normal.
        logger.debug("Telegram API (%s): long-poll returned empty (no new messages).", method)
        return {}
    except Exception as exc:
        logger.warning("Telegram API (%s) failed: %s", method, exc)
        return {}


def _reply(token: str, chat_id, text: str) -> None:
    """Send a message with the persistent control keyboard attached."""
    _api(
        token, "sendMessage",
        chat_id=chat_id,
        text=text,
        reply_markup=_KEYBOARD,
        parse_mode="HTML",
    )


def _proc_alive(key: str) -> bool:
    """Return True if the subprocess stored under *key* is still running."""
    proc = _procs.get(key)
    return proc is not None and proc.poll() is None


def _handle(token: str, chat_id, user_id: str, text: str) -> None:
    """Authorise and dispatch a single inbound command."""
    # /start and /help are shown to anyone — they only reveal the keyboard
    if text in ("/start", "/help"):
        _reply(
            token, chat_id,
            "<b>Gold Regime X — Command Center</b>\n\n"
            "Use the buttons below to manage the trading system remotely.\n\n"
            "<b>🚀 START M5</b>   — Launch the live M5 trading loop\n"
            "<b>🚀 START M15</b>  — Launch the live M15 trading loop\n"
            "<b>🚀 START H1</b>   — Launch the live H1 trading loop\n"
            "<b>🛑 STOP TRADING</b>  — Terminate all active trading loops\n"
            "<b>📉 OPTIMIZE M5</b>   — Resume/start M5 optimisation (0.50–0.55 range)\n"
            "<b>📊 BOT STATUS</b>    — Live P&amp;L, daily trade count, process health",
        )
        return

    allowed = os.getenv("ALLOWED_USER_ID", "")
    if user_id != allowed:
        _api(token, "sendMessage", chat_id=chat_id, text="Unauthorized.")
        return

    broker  = os.getenv("LIVE_BROKER",  "headway_cent")
    balance = os.getenv("LIVE_BALANCE", "15")

    def _start_tf(tf: str) -> None:
        """Launch a live trading loop for *tf* if one is not already running."""
        key = f"trading_{tf}"
        if _proc_alive(key):
            _reply(token, chat_id, f"{tf} trading is already running.")
            return
        _procs[key] = subprocess.Popen([
            "python", "main.py",
            "--mode", "live", "--yes",
            "--tf", tf, "--broker", broker, "--balance", balance,
        ])
        _reply(token, chat_id,
               f"<b>{tf} trading started</b>\nbroker={broker}  balance=${balance}")

    if text == "🚀 START M5":
        _start_tf("M5")

    elif text == "🚀 START M15":
        _start_tf("M15")

    elif text == "🚀 START H1":
        _start_tf("H1")

    elif text == "🛑 STOP TRADING":
        stopped = []
        for tf in ("M5", "M15", "H1"):
            key = f"trading_{tf}"
            if _proc_alive(key):
                _procs[key].terminate()
                stopped.append(tf)
        if stopped:
            _reply(token, chat_id, f"<b>Trading stopped.</b>  ({', '.join(stopped)})")
        else:
            _reply(token, chat_id, "No active trading process found.")

    elif text == "📉 OPTIMIZE M5 (0.50-0.55)":
        if _proc_alive("optimizer"):
            _reply(token, chat_id, "Optimisation is already running.")
            return
        _procs["optimizer"] = subprocess.Popen([
            "python", "main.py",
            "--mode", "optimize", "--tf", "M5",
            "--broker", broker, "--balance", balance, "--trials", "500",
        ])
        _reply(
            token, chat_id,
            "📉 <b>M5 Optimisation started</b>\n"
            "Signal thresholds: <b>Z-Score calibrated</b> (auto from IS regime stats)\n"
            "Target: <b>500 trials</b> (resumes from study.db if interrupted)\n\n"
            "You will receive Telegram updates at every 10% milestone.",
        )

    elif text == "📊 BOT STATUS":
        # ── Process health ───────────────────────────────────────────────────
        lines = ["<b>System Status</b>"]
        for tf in ("M5", "M15", "H1"):
            key   = f"trading_{tf}"
            alive = _proc_alive(key)
            icon  = "✅" if alive else "❌"
            state = "Running" if alive else "Stopped"
            lines.append(f"{tf} Loop:       {icon} {state}")
        optimizer_icon = "🔄" if _proc_alive("optimizer") else "💤"
        opt_state = "Running" if _proc_alive("optimizer") else "Idle"
        lines.append(f"Optimizer:    {optimizer_icon} {opt_state}")
        lines.append("—" * 28)
        proc_block = "\n".join(lines) + "\n"

        # ── P&L report from auditor ──────────────────────────────────────────
        # Session limit: 2/day for small accounts (≤$50), 3 otherwise
        session_limit = 2 if float(balance) <= 50 else 3
        try:
            from src.auditor import get_daily_report
            pnl_block = get_daily_report(broker=broker, session_limit=session_limit)
        except Exception as exc:
            pnl_block = f"P&amp;L unavailable: {exc}"

        _reply(token, chat_id, proc_block + pnl_block)

    else:
        _reply(token, chat_id, "Use the keyboard buttons below to control the bot.")


def run_listener() -> None:
    """Poll Telegram for updates and dispatch commands until KeyboardInterrupt.

    Uses the getUpdates long-polling method (timeout=30s) so the bot reacts
    within seconds without holding a persistent WebSocket connection.
    """
    token, _ = get_credentials()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set.  "
            "Copy .env.example to .env and fill in your credentials."
        )

    logger.info("Remote control listener started.  Waiting for commands...")
    send_telegram_msg("<b>Gold Regime X</b> remote control is <b>online</b>.")

    offset = 0
    while True:
        try:
            # timeout=25: Telegram holds the connection up to 25s for new msgs.
            # _req_timeout read must exceed that so requests doesn't give up first.
            data = _api(token, "getUpdates", offset=offset,
                        timeout=25, _req_timeout=(10, 35))
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg     = update.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                user_id = str(msg.get("from", {}).get("id", ""))
                text    = msg.get("text", "").strip()
                if chat_id and text:
                    logger.info(
                        "Inbound: user_id=%s  chat_id=%s  text=%r",
                        user_id, chat_id, text,
                    )
                    _handle(token, chat_id, user_id, text)

        except KeyboardInterrupt:
            logger.info("Remote control stopped.")
            send_telegram_msg("Gold Regime X remote control is <b>offline</b>.")
            break
        except Exception as exc:
            logger.error("Listener error: %s — retrying in 10s", exc)
            time.sleep(10)
