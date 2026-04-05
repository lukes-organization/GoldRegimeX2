"""Daily performance auditor for Gold Regime X.

Fetches closed deal history from MT5 for the last 24 hours and formats
a Telegram-ready HTML summary.  Floating P&L from any currently open positions
is included so you see the real-time risk alongside historical results.

Usage (via main.py):
    python main.py --mode audit --broker headway_cent

The report is also triggered automatically by:
  - The Telegram remote control "📊 BOT STATUS" button
  - The nightly scheduler inside --mode listen (fires at 23:55 UTC)
"""

from datetime import datetime, timedelta, timezone

from src.logger import setup_logger
from src.risk_manager import CENT_MULTIPLIER

logger = setup_logger(__name__)


def get_daily_report(
    broker: str = "headway_cent",
    days: int = 1,
    session_limit: int = None,
) -> str:
    """Fetch and format a performance summary for the last *days* days.

    Args:
        broker:        Broker name — used to normalise cent-account P&L.
        days:          Look-back window in days (default 1 = last 24 hours).
        session_limit: If provided, the trade count line shows "X/N used today"
                       so the user can see remaining capacity at a glance.

    Returns an HTML-formatted string suitable for Telegram (parse_mode='HTML').
    Never raises — returns an error string if MT5 is unavailable.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return "MetaTrader5 package not installed — cannot fetch report."

    to_date   = datetime.now()
    from_date = to_date - timedelta(days=days)

    history = mt5.history_deals_get(from_date, to_date)
    if history is None or len(history) == 0:
        period = "24 hours" if days == 1 else f"{days} days"
        return f"No trades closed in the last {period}."

    try:
        import pandas as pd
    except ImportError:
        return "pandas not installed — cannot format report."

    df = pd.DataFrame(list(history), columns=history[0]._asdict().keys())

    # DEAL_ENTRY_OUT = closed exit deals (excludes balance deposits, etc.)
    df = df[df["entry"] == mt5.DEAL_ENTRY_OUT]
    if df.empty:
        return "No exit deals recorded in the last 24 hours."

    # Normalise P&L: cent accounts report in cents, convert to USD
    divisor = float(CENT_MULTIPLIER) if broker == "headway_cent" else 1.0
    df["profit_usd"] = df["profit"] / divisor

    total_trades = len(df)
    wins         = int((df["profit"] > 0).sum())
    losses       = total_trades - wins
    win_rate     = (wins / total_trades * 100) if total_trades else 0.0
    total_pnl    = float(df["profit_usd"].sum())
    best_trade   = float(df["profit_usd"].max())
    worst_trade  = float(df["profit_usd"].min())

    # Floating P&L from any currently open positions
    positions = mt5.positions_get()
    floating_line = ""
    if positions:
        floating = sum(p.profit / divisor for p in positions)
        floating_line = f"Floating P&L:  <code>{floating:+.2f} USD</code>\n"

    icon_pnl = "+" if total_pnl >= 0 else ""
    period   = "24h" if days == 1 else f"{days}d"
    stamp    = datetime.now(timezone.utc).strftime("%H:%M UTC")

    # Trade count line — show session capacity if limit is known
    if session_limit:
        trades_str = f"<b>{total_trades}/{session_limit} used today</b>  (Wins: {wins} | Losses: {losses})"
    else:
        trades_str = f"<b>{total_trades}</b>  (Wins: {wins} | Losses: {losses})"

    report = (
        f"<b>Gold Regime X — Daily Report ({period})</b>\n"
        f"{'—'*28}\n"
        f"Total PnL:     <code>{icon_pnl}{total_pnl:.2f} USD</code>\n"
        f"{floating_line}"
        f"Trades:        {trades_str}\n"
        f"Win Rate:      <b>{win_rate:.1f}%</b>\n"
        f"Best Trade:    <code>+{best_trade:.2f} USD</code>\n"
        f"Worst Trade:   <code>{worst_trade:.2f} USD</code>\n"
        f"{'—'*28}\n"
        f"<i>Generated at {stamp}</i>"
    )
    logger.info(
        "Daily report: pnl=%.2f  trades=%d  win_rate=%.1f%%",
        total_pnl, total_trades, win_rate,
    )
    return report
