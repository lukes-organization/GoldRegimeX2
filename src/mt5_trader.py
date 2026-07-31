"""MT5 Live Execution Engine.

Provides account telemetry, feature engineering parity with the training
pipeline, margin validation, and live order placement through the
MetaTrader5 Python package.

Usage (via main.py):
    python main.py --mode demo --broker headway_cent --balance 15 --tf H1
    python main.py --mode live --broker headway_cent --balance 15 --tf H1

IMPORTANT: Remove the GoldRegimeX.mq5 EA from the XAUUSD chart before running
this script.  Both the EA and the Python bridge use MAGIC_NUMBER = 123456 and
their session counters are independent — running both simultaneously will
double-count trades.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.notifier import send_telegram_msg

from src.logger import setup_logger
# -- Notebook-engine live seam (legacy ML stack removed in consolidation) --
# Entries and exits now come solely from strategy_backtest.latest_signal(),
# a 1:1 mirror of run_ml_filtered_backtest. See CONSOLIDATION.md.

# Kept modules (still part of the consolidated system)
from src.risk_manager import AdaptiveRiskManager, BROKER_CONFIGS, CENT_MULTIPLIER, DailyEquityGate
from src.trade_lifecycle import config_for_tf, floating_pnl_usd

# Notebook-engine live signal seam (ready for the live-executor migration)
try:
    from src.strategy_backtest import latest_signal, load_live_bundle
except Exception:
    latest_signal = load_live_bundle = None


logger = setup_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOL               = "XAUUSD"
# Each timeframe gets its own magic number so H1/M15/M5 instances running
# simultaneously don't block or close each other's positions.
TF_MAGIC_MAP  = {"H1": 123456, "M15": 123457, "M5": 123458}
ALL_GRX_MAGICS = frozenset(TF_MAGIC_MAP.values())   # used for cross-TF global guard
MAGIC_NUMBER  = TF_MAGIC_MAP["H1"]   # backwards-compat alias (MQL5 EA default)
CHOP_STATE                   = 2        # HMM Chop state index
BULL_STATE                   = 0        # HMM Bull state index
BEAR_STATE                   = 1        # HMM Bear state index
DEFAULT_DEVIATION            = 20       # fallback deviation for check_margin / send_market_order
N_BARS_WARMUP                = 200      # bars fetched for Kalman/HMM warm-up
POLL_INTERVAL_SEC            = 5        # seconds between bar-change checks
HIGH_VOL_SELF_TRANS_THRESHOLD = 0.70    # self-transition prob below this -> elevated deviation
MIN_SPREAD_RATIO = {"headway_cent": 1.5, "standard": 3.0}  # TP1 vs spread floor

# SL = ATR × multiplier (per TF — M5 tighter to avoid noise-outs on scalps)
TF_ATR_MULTIPLIER  = {"M5": 1.5, "M15": 2.0, "H1": 2.0}

# ── ATR-linked Hybrid Trailing Stop ──────────────────────────────────────────
# ATR_TRAIL_CONFIG: per-TF activation thresholds and trail multipliers.
# activation_pnl: floating P&L that triggers Phase 1 (BE + 2×spread + partial).
# trail_mult:     Phase 2 ATR trail distance multiplier (unchanged from original).
# M5 scalp_target: close-at-profit target for between-bar scalp exit logic.
ATR_TRAIL_CONFIG: dict = {
    "H1":  {"activation_pnl": 1.50, "trail_mult": 2.5, "partial_close": True},
    "M15": {"activation_pnl": 1.50, "trail_mult": 1.5, "partial_close": True},
    "M5":  {"activation_pnl": 1.00, "trail_mult": 1.5, "partial_close": False,
            "scalp_target": 4.00, "recycle": True},
}
# Legacy aliases (kept so old references still resolve — use ATR_TRAIL_CONFIG for new code)
PROFIT_ACTIVATION_USD = 2.50          # superseded by ATR_TRAIL_CONFIG[tf]['activation_pnl']
ATR_TRAIL_MULTIPLIER  = {"M5": 1.5, "M15": 1.5, "H1": 2.5}   # superseded by ATR_TRAIL_CONFIG

# Minimum lot for partial close.  MT5 rejects close volumes below 0.01.
# When a position is already at 0.01 lots the partial close is skipped and the
# ATR trail runs on the full position instead.
MIN_LOT_GUARD = 0.01

# ── Per-timeframe signal thresholds and order parameters ──────────────────────
TF_PROB_THRESHOLD  = {"M5": 0.55, "M15": 0.55, "H1": 0.55}   # fallback — Optuna value used when available
TF_SHORT_THRESHOLD = {"M5": 0.45, "M15": 0.45, "H1": 0.45}   # fallback only
TF_DEFAULT_DEV     = {"M5": 30,   "M15": 20,   "H1": 20}
TF_HIGH_VOL_DEV    = {"M5": 50,   "M15": 50,   "H1": 50}

# ── State-aware multi-stage TP multipliers (relative to SL distance) ──────────
# Bull / Bear: TP1 quick partial, TP2 runner.
# Chop: tighter single TP — position 2 (runner) is skipped.
# Format: {regime: [tp1_mult, tp2_mult]}; single-element list = one TP, close all.
# M5 uses tighter mults (0.8 / 2.0) — TP1 locks in quick profit, TP2 is realistic
# for a scalp runner vs the original [1.0, 3.0] which rarely filled on M5.
TF_TP_CONFIG = {
    # M5 growth accounts (pos_per_trade=3) use all three TPs.
    # Small accounts (pos_per_trade=2) only use TP1+TP2 — TP3 entry is ignored.
    # TP3 (3.0x) only fills on genuine momentum sessions; trailing guard exits
    # position 3 gracefully when momentum fades before the target.
    "M5":  {"trending": [0.8, 1.5, 3.0], "chop": [0.5]},
    "M15": {"trending": [1.0, 2.0], "chop": [0.8]},   # partial at 1:1, runner at 2:1
    "H1":  {"trending": [1.0, 2.0], "chop": [1.0]},   # same ratio as M15 — 3.0x was rarely filled
}

# ── Regime-aware TP/SL multipliers (all values are direct ATR multiples) ──────
# sl_mult × ATR = stop-loss distance; tp1/tp2_mult × ATR = take-profit distances.
TP_SL_CONFIG: dict = {
    "H1":  {
        "trend": {"tp1_mult": 1.5, "tp2_mult": 3.0, "sl_mult": 2.0},
        "chop":  {"tp1_mult": 1.0, "tp2_mult": None, "sl_mult": 1.4},
    },
    "M15": {
        "trend": {"tp1_mult": 1.2, "tp2_mult": 2.5, "sl_mult": 2.0},
        "chop":  {"tp1_mult": 0.8, "tp2_mult": None, "sl_mult": 1.4},
    },
    "M5":  {
        "trend": {"tp1_mult": 0.8, "tp2_mult": 1.5, "sl_mult": 1.5},
        "chop":  {"tp1_mult": 0.5, "tp2_mult": None, "sl_mult": 1.05},
    },
}


# Lazy MT5 timeframe map

# ── External asset fallback cache ─────────────────────────────────────────────
# When a live MT5 bar fetch fails (e.g. symbol not subscribed), carry the last
# known log return from the master CSV.  Keyed by "{col_name}_{tf}".
_ASSET_FALLBACK_CACHE: dict[str, float] = {}

_MT5_TF_MAP: dict | None = None


def _get_tf_map() -> dict:
    global _MT5_TF_MAP
    if _MT5_TF_MAP is None:
        import MetaTrader5 as mt5
        _MT5_TF_MAP = {
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1":  mt5.TIMEFRAME_H1,
        }
    return _MT5_TF_MAP


def _normalise_balance(raw_balance: float, broker: str) -> float:
    """Convert MT5 raw balance to USD for AdaptiveRiskManager.

    On Headway Cent accounts the terminal displays balance in cents
    (e.g. $15 USD shows as 1500.00).  The --broker headway_cent flag triggers
    the divide-by-100 normalisation.
    """
    if broker == "headway_cent":
        return raw_balance / CENT_MULTIPLIER
    return raw_balance


# ─────────────────────────────────────────────────────────────────────────────
# Account queries
# ─────────────────────────────────────────────────────────────────────────────

def get_account_telemetry() -> dict:
    """Return a snapshot of the current MT5 account state.

    Raises ``RuntimeError`` if not connected to MT5.
    """
    import MetaTrader5 as mt5
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(
            f"mt5.account_info() returned None: {mt5.last_error()}. "
            "Is the terminal connected?"
        )
    telemetry = {
        "balance":      info.balance,
        "equity":       info.equity,
        "margin":       info.margin,
        "free_margin":  info.margin_free,
        "margin_level": info.margin_level,
        "currency":     info.currency,
        "login":        info.login,
        "server":       info.server,
        "company":      info.company,
        "trade_mode":   info.trade_mode,
    }
    logger.debug(
        "Telemetry: balance=%.2f  free_margin=%.2f  equity=%.2f",
        telemetry["balance"], telemetry["free_margin"], telemetry["equity"],
    )
    return telemetry


def display_account_info(trading_balance: float = None) -> None:
    """Print a formatted account dashboard to stdout."""
    import MetaTrader5 as mt5
    t = get_account_telemetry()
    is_demo = t["trade_mode"] == mt5.ACCOUNT_TRADE_MODE_DEMO
    print("=" * 50)
    print(f"  GOLD REGIME X — LIVE BRIDGE")
    print("=" * 50)
    print(f"  Broker:      {t['company']}")
    print(f"  Login:       {t['login']}")
    print(f"  Server:      {t['server']}")
    print(f"  Mode:        {'DEMO' if is_demo else 'LIVE'}")
    print(f"  MT5 Balance: {t['balance']:.2f} {t['currency']}")
    if trading_balance is not None:
        print(f"  Risk Balance:{trading_balance:.2f} USD  (used for lot sizing)")
    print(f"  Equity:      {t['equity']:.2f} {t['currency']}")
    print(f"  Free Margin: {t['free_margin']:.2f} {t['currency']}")
    print("=" * 50)


def has_open_position(symbol: str = DEFAULT_SYMBOL, magic: int = MAGIC_NUMBER) -> bool:
    """Return True if there is at least one open position for *symbol* with *magic*."""
    import MetaTrader5 as mt5
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.debug("positions_get returned None: %s", mt5.last_error())
        return False
    return any(p.magic == magic for p in positions)


def check_margin(symbol: str, lot: float, order_type: int, price: float) -> bool:
    """Pre-flight margin check via mt5.order_check().

    Returns True if the order passes margin validation without submitting it.
    """
    import MetaTrader5 as mt5
    request = {
        "action":        mt5.TRADE_ACTION_DEAL,
        "symbol":        symbol,
        "volume":        float(lot),
        "type":          order_type,
        "price":         float(price),
        "deviation":     DEFAULT_DEVIATION,
        "magic":         MAGIC_NUMBER,
        "type_filling":  mt5.ORDER_FILLING_IOC,
    }
    check = mt5.order_check(request)
    if check is None:
        logger.warning("order_check returned None: %s", mt5.last_error())
        return False
    ok = check.retcode == 0
    logger.debug("Margin check: retcode=%d  margin=%.2f  free_margin=%.2f",
                 check.retcode, check.margin, check.margin_free)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Signal derivation
# ─────────────────────────────────────────────────────────────────────────────


def _move_sl_to_breakeven(ticket: int, entry_price: float, mt5) -> None:
    """Modify an open position's SL to the entry price (break-even)."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return
    pos = positions[0]
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl":       round(entry_price, 2),
        "tp":       pos.tp,
    }
    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("Break-even SL set: ticket=%d  sl=%.2f", ticket, entry_price)
    else:
        logger.warning("Break-even SL failed: ticket=%d  retcode=%s",
                       ticket, res.retcode if res else "None")


def _apply_profit_guard(signal_tracker: dict, mt5) -> None:
    """Move SL to entry + 2×spread once price reaches 70% of TP1 distance.

    Fires once per signal (guarded by signal_tracker["guard_hit"]).  Applies
    to all open tickets in the tracker regardless of timeframe.  This protects
    profit before TP1 fills by effectively making the position risk-free.
    """
    if signal_tracker.get("guard_hit") or signal_tracker.get("tp1_hit"):
        return   # already protected or TP1 already hit
    tp1_level = signal_tracker.get("tp1_level")
    entry     = signal_tracker.get("entry_price", 0.0)
    direction = signal_tracker.get("direction")
    tickets   = signal_tracker.get("tickets", [])
    if not tp1_level or not tickets or not direction:
        return

    tick = mt5.symbol_info_tick(DEFAULT_SYMBOL)
    if not tick:
        return

    spread       = tick.ask - tick.bid
    tp1_dist     = abs(tp1_level - entry)
    guard_buffer = tp1_dist * 0.70   # trigger at 70% of the way to TP1

    if direction == "BUY":
        triggered = tick.bid >= entry + guard_buffer
        new_sl    = round(entry + spread * 2, 2)
    else:
        triggered = tick.ask <= entry - guard_buffer
        new_sl    = round(entry - spread * 2, 2)

    if triggered:
        for ticket in tickets:
            _move_sl_to_breakeven(ticket, new_sl, mt5)
        signal_tracker["guard_hit"] = True
        logger.info(
            "Profit guard triggered: entry=%.2f  new_sl=%.2f  (70%% of TP1 at %.2f)",
            entry, new_sl, tp1_level,
        )


def _set_trailing_sl(
    ticket: int, new_sl: float, current_sl: float, direction: str, mt5
) -> bool:
    """Update an open position's SL only if it improves on the current SL.

    For BUY positions: new_sl must be > current_sl (ratchet upward).
    For SELL positions: new_sl must be < current_sl (ratchet downward).
    Returns True if the SL was actually updated.
    """
    if direction == "BUY"  and new_sl <= current_sl:
        return False
    if direction == "SELL" and new_sl >= current_sl:
        return False
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False
    pos = positions[0]
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl":       round(new_sl, 2),
        "tp":       pos.tp,
    }
    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(
            "ATR trail SL updated: ticket=%d  sl=%.2f  (was %.2f)",
            ticket, new_sl, current_sl,
        )
        return True
    logger.warning(
        "ATR trail SL failed: ticket=%d  retcode=%s",
        ticket, res.retcode if res else "None",
    )
    return False


def _execute_partial_close(ticket: int, symbol: str, mt5, magic: int = MAGIC_NUMBER) -> bool:
    """Close 50 % of a position's volume to bank partial profit.

    Skips and returns False if volume <= MIN_LOT_GUARD (0.01) since MT5
    rejects close volumes below the broker minimum lot.
    """
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False
    pos = positions[0]
    close_vol = round(pos.volume / 2.0, 2)
    if close_vol < MIN_LOT_GUARD:
        logger.info(
            "Partial close skipped: ticket=%d  vol=%.2f  half=%.2f < MIN_LOT_GUARD %.2f",
            ticket, pos.volume, close_vol, MIN_LOT_GUARD,
        )
        return False
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick  = mt5.symbol_info_tick(symbol)
    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "position":     ticket,
        "symbol":       symbol,
        "volume":       close_vol,
        "type":         close_type,
        "price":        price,
        "magic":        magic,
        "comment":      "GRX_Partial_Profit",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(
            "Partial close: ticket=%d  vol_closed=%.2f  price=%.2f",
            ticket, close_vol, price,
        )
        return True
    logger.warning(
        "Partial close failed: ticket=%d  retcode=%s  vol=%.2f",
        ticket, res.retcode if res else "None", close_vol,
    )
    return False


def _close_position(ticket: int, mt5, comment: str = "GRX_close_chop", magic: int = MAGIC_NUMBER) -> None:
    """Close a specific open position at market price."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return
    pos   = positions[0]
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick  = mt5.symbol_info_tick(pos.symbol)
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "position":     ticket,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "price":        price,
        "deviation":    20,
        "magic":        magic,
        "comment":      comment,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("Position closed (%s): ticket=%d", comment, ticket)
    else:
        logger.warning("Position close failed: ticket=%d  retcode=%s",
                       ticket, res.retcode if res else "None")


def _log_closed_pnl(tickets: list, mt5, broker: str = "headway_cent", tf: str = "H1") -> None:
    """Query MT5 deal history for each closed ticket and log realized P&L.

    MT5 deal profits are reported in the account currency.  On Headway Cent
    accounts the currency is cUSD (cents); divide by CENT_MULTIPLIER to get
    real USD.  Retries up to 20 s waiting for the closing fill to appear.
    """
    from src.notifier import send_telegram_msg
    now   = datetime.now(timezone.utc)
    start = now - timedelta(hours=48)   # wide window covers overnight gaps

    # Cent accounts report P&L in cUSD — divide by 100 to get real USD.
    # Use the broker parameter directly; do NOT rely on raw balance because
    # demo standard accounts routinely have balances > 10 000 USD.
    is_cent = (broker == "headway_cent")

    for ticket in tickets:
        try:
            deals = None
            for _ in range(20):          # retry — exit deal can lag 10–15 s after close
                raw = mt5.history_deals_get(start, now, position=ticket)
                # Explicitly filter by position_id — mt5.history_deals_get with
                # position= can return all deals on some brokers/builds if the
                # filter is silently ignored.  Filtering here guarantees we only
                # process deals that belong to this specific position.
                deals = [d for d in (raw or []) if d.position_id == ticket]
                # Only accept once we have the closing fill (DEAL_ENTRY_OUT = 1)
                # The opening deal (entry=0) appears immediately; that is why a
                # shorter retry loop returns pnl=0.0 — it only finds the open fill.
                if deals and any(d.entry == 1 for d in deals):
                    break
                time.sleep(1.0)

            if not (deals and any(d.entry == 1 for d in deals)):
                logger.info(
                    "Position #%d CLOSED (exit deal not in history after 20 s).", ticket
                )
                continue

            pnl_raw    = sum(d.profit + d.commission for d in deals)
            pnl        = pnl_raw / CENT_MULTIPLIER if is_cent else pnl_raw
            in_deal    = next((d for d in deals if d.entry == 0), None)   # entry fill
            out_deal   = next((d for d in deals if d.entry == 1), None)   # exit fill
            entry_px   = in_deal.price   if in_deal  else 0.0
            exit_px    = out_deal.price  if out_deal else 0.0
            lot        = (out_deal or in_deal).volume if (out_deal or in_deal) else 0.0
            direction  = "BUY" if (in_deal and in_deal.type == 0) else "SELL"
            emoji      = "✅" if pnl > 0 else ("➡️" if pnl == 0 else "❌")
            tag        = "WIN" if pnl > 0 else ("BREAK-EVEN" if pnl == 0 else "LOSS")

            # Points moved: positive = trade went in our favour
            if entry_px > 0 and exit_px > 0:
                raw_pts = exit_px - entry_px if direction == "BUY" else entry_px - exit_px
                pts_str = f"{raw_pts:+.2f} pts"
            else:
                pts_str = "n/a pts"

            logger.info(
                "Position #%d CLOSED — P&L: %+.2f USD  [%s]  %s  "
                "entry=%.2f -> exit=%.2f  %s  lot=%.2f",
                ticket, pnl, tag, direction, entry_px, exit_px, pts_str, lot,
            )
            send_telegram_msg(
                f"{emoji} <b>[{tf}] Trade closed</b>  #{ticket}\n"
                f"{direction}  lot=<b>{lot:.2f}</b>  "
                f"entry: <b>{entry_px:.2f}</b> -> exit: <b>{exit_px:.2f}</b>\n"
                f"Move: <b>{pts_str}</b>  |  "
                f"Realized P&L: <b>{pnl:+.2f} USD</b>  [{emoji} {tag}]"
            )
        except Exception as exc:
            logger.warning("Could not fetch P&L for ticket #%d: %s", ticket, exc)


def send_daily_audit_report(mt5, broker: str = "headway_cent") -> None:
    """Query today's closed deals for all GRX magic numbers and send a P&L summary.

    Groups realized profit by timeframe (H1 / M15 / M5), then sends one
    consolidated Telegram message.  Called automatically at UTC midnight reset.
    """
    from src.notifier import send_telegram_msg
    now   = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)

    is_cent = (broker == "headway_cent")
    try:
        deals = mt5.history_deals_get(start, now) or []
    except Exception as exc:
        logger.warning("Daily audit: could not fetch MT5 deal history: %s", exc)
        return

    # Group closing fills (DEAL_ENTRY_OUT = 1) by TF magic number
    tf_results: dict[str, tuple[float, int]] = {}
    for tf_name, magic in TF_MAGIC_MAP.items():
        tf_deals = [d for d in deals if d.magic == magic and d.entry == 1]
        if tf_deals:
            pnl_raw = sum(d.profit + d.commission for d in tf_deals)
            pnl     = pnl_raw / CENT_MULTIPLIER if is_cent else pnl_raw
            tf_results[tf_name] = (pnl, len(tf_deals))

    if not tf_results:
        send_telegram_msg("📅 <b>Daily Performance Report</b>\nNo GRX trades closed today.")
        return

    lines = ["📅 <b>DAILY PERFORMANCE REPORT</b>"]
    for tf_name in ["H1", "M15", "M5"]:
        if tf_name in tf_results:
            pnl, count = tf_results[tf_name]
            sign = "+" if pnl >= 0 else ""
            trade_word = "trade" if count == 1 else "trades"
            lines.append(f"{tf_name}: <b>{sign}${pnl:.2f}</b> ({count} {trade_word})")

    total_pnl    = sum(v[0] for v in tf_results.values())
    total_trades = sum(v[1] for v in tf_results.values())
    sign = "+" if total_pnl >= 0 else ""
    lines.append(f"<b>TOTAL: {sign}${total_pnl:.2f} USD  ({total_trades} trades)</b>")

    logger.info(
        "Daily audit: total P&L=%+.2f USD  trades=%d  (broker=%s)",
        total_pnl, total_trades, broker,
    )
    send_telegram_msg("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Order execution
# ─────────────────────────────────────────────────────────────────────────────

def send_market_order(
    symbol: str   = DEFAULT_SYMBOL,
    order_type: int = None,
    lot: float    = 0.01,
    sl: float     = 0.0,
    tp: float     = 0.0,
    deviation: int = DEFAULT_DEVIATION,
    magic: int    = MAGIC_NUMBER,
    comment: str  = "GRX_Python",
) -> dict:
    """Send an IOC market order to the MT5 terminal.

    Uses ORDER_FILLING_IOC (Immediate or Cancel) — standard for ECN/Cent
    brokers.  If the fill price is outside the deviation window the order is
    cancelled rather than partially filled, preventing unintended exposure.

    Returns a dict with keys: retcode, order, comment, success.
    Does NOT raise on failure — inspect result['success'].
    """
    import MetaTrader5 as mt5
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("Cannot get tick for %s: %s", symbol, mt5.last_error())
        return {"retcode": -1, "order": 0, "comment": "No tick data", "success": False}

    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       float(lot),
        "type":         order_type,
        "price":        float(price),
        "sl":           float(sl),
        "tp":           float(tp),
        "deviation":    int(deviation),
        "magic":        int(magic),
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        logger.error("order_send returned None: %s", mt5.last_error())
        return {"retcode": -1, "order": 0, "comment": str(mt5.last_error()), "success": False}

    success = result.retcode == mt5.TRADE_RETCODE_DONE
    if success:
        logger.info(
            "Order filled: #%d  %s  lot=%.2f  price=%.2f  sl=%.2f  tp=%.2f  dev=%d",
            result.order,
            "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL",
            lot, price, sl, tp, deviation,
        )
    else:
        logger.error(
            "Order rejected: retcode=%d  comment=%s",
            result.retcode, result.comment,
        )
    return {
        "retcode": result.retcode,
        "order":   result.order,
        "comment": result.comment,
        "success": success,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Live execution loop
# ─────────────────────────────────────────────────────────────────────────────


def run_live_loop(
    tf: str          = "H1",
    broker: str      = "headway_cent",
    account_size: float = None,
    profit_target: float = None,
    use_tiered: bool = False,
) -> None:
    """Connect to MT5 and run the signal → order loop until interrupted.

    Fires on each newly completed bar.  Position sizing, session limits, and
    margin validation are enforced before every order.

    Args:
        tf:            Timeframe to trade — "H1", "M15", or "M5".
        broker:        Broker config key from risk_manager.BROKER_CONFIGS.
        account_size:  USD balance used for lot-sizing.  If None, reads from MT5
                       and normalises for cent accounts automatically.
        profit_target: Close early when floating P&L reaches this USD amount.
                       Defaults to PROFIT_ACTIVATION_USD for all TFs (legacy param).
        use_tiered:    Legacy param — retained for backward compat, unused.
    """
    import MetaTrader5 as mt5
    from src.mt5_sync import connect_mt5, disconnect_mt5

    if not connect_mt5():
        raise ConnectionError("Could not connect to MT5 terminal.")

    try:
        _run_loop_inner(tf, broker, account_size, mt5, profit_target, use_tiered=use_tiered)
    finally:
        disconnect_mt5()
        logger.info("Live loop terminated.  MT5 disconnected.")


def _run_loop_inner(tf: str, broker: str, account_size: float, mt5,
                    profit_target: float = None,
                    use_tiered: bool = False) -> None:
    """1:1 LIVE MIRROR of run_ml_filtered_backtest / _run_backtest_numba.

    Entries and exits are derived solely from strategy_backtest.latest_signal().
    SL/TP sizing, the asymmetric guard_factor, the leg A/B structure, the
    scale-in leg C, the regime-3 structural MR exit, and the mode-4 ATR trail /
    time stop all replicate the notebook backtest kernel exactly. Lot sizing
    keeps AdaptiveRiskManager. Real bid/ask fills already embed spread/slippage,
    so the kernel's synthetic spread/slippage adjustments are the one necessary
    live adaptation. PAPER-TRADE (demo) before risking capital.
    """
    from src.mt5_sync import ensure_data_updated

    tf_up  = tf.upper()
    tf_mt5 = _get_tf_map()[tf_up]
    symbol = DEFAULT_SYMBOL
    is_m5  = tf_up == "M5"
    PIP_SIZE_PRICE = 0.10   # XAUUSD: 1 pip = 0.10 price (kernel constant)

    telemetry = get_account_telemetry()
    if account_size is None:
        account_size = _normalise_balance(telemetry["balance"], broker)
        logger.info("Balance auto-detected: MT5 raw=%.2f  USD normalised=%.2f",
                    telemetry["balance"], account_size)
    display_account_info(trading_balance=account_size)

    magic = TF_MAGIC_MAP.get(tf_up, TF_MAGIC_MAP["H1"])
    logger.info("Magic number for [%s]: %d", tf_up, magic)

    # Notebook-engine live bundle (raises if --mode optimize was not run first).
    bundle = None
    if load_live_bundle is not None:
        for _a in ((tf_up, broker), (tf_up,), ()):
            try:
                bundle = load_live_bundle(*_a); break
            except TypeError:
                continue
    if latest_signal is None:
        raise RuntimeError("strategy_backtest.latest_signal unavailable — cannot trade.")

    arm = AdaptiveRiskManager(account_size, tf=tf_up, broker=broker)
    equity_gate = DailyEquityGate(tf=tf_up)
    try:
        equity_gate.reset_day(telemetry["equity"])
    except Exception:
        pass

    legs = {"A": None, "B": None, "C": None}
    leg_a_profit_hit = False
    leg_b_profit_hit = False
    last_bar_time = 0
    last_audit_day = None

    EXIT_MAP = {"fixed_tp": 0, "mr_exit": 1, "fixed_tp_plus_mr": 2,
                "partial_tp_plus_mr": 3, "partial_tp_mr_time_stop": 4}

    def _leg_lots(stop_dist):
        try:
            if getattr(arm, "is_small_account", False):
                return (0.02, 0.03) if broker == "headway_cent" else (0.01, 0.01)
            sl_pips = max(stop_dist / PIP_SIZE_PRICE, 1.0)
            base = float(arm.calculate_lot_size(sl_pips))
            per = max(MIN_LOT_GUARD, round(base / 2.0, 2))
            return per, per
        except Exception as exc:
            logger.warning("Lot sizing fallback (%s)", exc)
            return (0.02, 0.03) if broker == "headway_cent" else (0.01, 0.01)

    def _open_map():
        poss = mt5.positions_get(symbol=symbol) or []
        return {p.ticket: p for p in poss if p.magic == magic}

    def _closed_in_profit(ticket):
        try:
            now = datetime.now(timezone.utc); start = now - timedelta(hours=48)
            raw = mt5.history_deals_get(start, now, position=ticket) or []
            deals = [d for d in raw if d.position_id == ticket]
            if not any(d.entry == 1 for d in deals):
                return None
            return sum(d.profit + d.commission for d in deals) > 0
        except Exception:
            return None

    def reconcile():
        nonlocal leg_a_profit_hit, leg_b_profit_hit
        open_map = _open_map(); closed = []
        for key in ("A", "B", "C"):
            leg = legs[key]
            if leg is None:
                continue
            if leg["ticket"] not in open_map:
                prof = _closed_in_profit(leg["ticket"])
                closed.append(leg["ticket"])
                if key == "A" and prof:
                    leg_a_profit_hit = True
                if key == "B" and prof:
                    leg_b_profit_hit = True
                legs[key] = None
        if closed:
            _log_closed_pnl(closed, mt5, broker=broker, tf=tf_up)
        if legs["A"] is None and legs["B"] is None and legs["C"] is None:
            leg_a_profit_hit = False
            leg_b_profit_hit = False

    def _close_all(reason):
        nonlocal leg_a_profit_hit, leg_b_profit_hit
        tickets = []
        for key in ("A", "B", "C"):
            if legs[key] is not None:
                _close_position(legs[key]["ticket"], mt5, comment=reason, magic=magic)
                tickets.append(legs[key]["ticket"]); legs[key] = None
        if tickets:
            _log_closed_pnl(tickets, mt5, broker=broker, tf=tf_up)
        leg_a_profit_hit = False
        leg_b_profit_hit = False

    logger.info("[%s] Notebook-engine live loop started (1:1 backtest parity).", tf_up)
    send_telegram_msg(f"[{tf_up}] GoldRegimeX live loop online (engine-parity build).")

    while True:
        try:
            today = datetime.now(timezone.utc).date()
            if last_audit_day != today:
                last_audit_day = today
                try:
                    send_daily_audit_report(mt5, broker=broker)
                except Exception as exc:
                    logger.warning("Daily audit failed: %s", exc)
                try:
                    equity_gate.reset_day(get_account_telemetry()["equity"])
                except Exception:
                    pass

            try:
                tele = get_account_telemetry()
                equity_gate.check(tele["equity"])
                if getattr(equity_gate, "locked", False):
                    if any(legs[k] is not None for k in ("A", "B", "C")):
                        _close_all("GRX_daily_equity_gate")
                    logger.warning("[%s] Daily equity gate LOCKED — trading paused.", tf_up)
                    time.sleep(POLL_INTERVAL_SEC); continue
            except Exception as exc:
                logger.warning("Equity gate check failed: %s", exc)

            rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 1, 1)
            if rates is None or len(rates) == 0:
                time.sleep(POLL_INTERVAL_SEC); continue
            bar_time = int(rates[0]["time"])
            if bar_time == last_bar_time:
                reconcile()
                time.sleep(POLL_INTERVAL_SEC); continue
            last_bar_time = bar_time

            try:
                ensure_data_updated(tf_up, symbol)
            except Exception as exc:
                logger.warning("Data refresh skipped: %s", exc)

            sig = latest_signal(tf_up, bundle=bundle)
            if not sig:
                time.sleep(POLL_INTERVAL_SEC); continue

            signal      = int(sig.get("signal", 0))
            regime_code = int(sig.get("regime_code", 0))
            atr14       = float(sig.get("atr", 0.0) or 0.0)
            engine_close = float(sig.get("close", 0.0) or 0.0)
            bp = sig.get("base_params", {}) or {}
            atr_stop         = float(bp.get("atr_stop", 2.0))
            atr_target       = float(bp.get("atr_target", 2.0))
            leg_a_atr_target = float(bp.get("leg_a_atr_target", 1.0))
            exit_model       = str(bp.get("exit_model", "fixed_tp"))
            trail_mult       = float(bp.get("trail_mult", 0.0))
            time_stop_minutes = float(bp.get("time_stop_minutes", -1.0))
            code = EXIT_MAP.get(exit_model, 0)
            enable_fixed_tp  = code in (0, 2)
            enable_mr        = code in (1, 2, 3, 4)
            enable_time_stop = code == 4
            enable_trail     = code == 4

            reconcile()

            if enable_mr and regime_code == 3 and any(legs[k] is not None for k in ("A", "B", "C")):
                logger.info("[%s] Structural MR exit (regime==3) — flattening.", tf_up)
                send_telegram_msg(f"[{tf_up}] MR exit (regime 3) — closing all legs.")
                _close_all("GRX_MR_regime3")
                time.sleep(POLL_INTERVAL_SEC); continue

            if enable_time_stop and time_stop_minutes > 0 and legs["B"] is not None:
                now_min = int(datetime.now(timezone.utc).timestamp() // 60)
                if (now_min - legs["B"]["entry_min"]) >= time_stop_minutes:
                    logger.info("[%s] Time stop (%.0f min) — flattening.", tf_up, time_stop_minutes)
                    _close_all("GRX_time_stop")
                    time.sleep(POLL_INTERVAL_SEC); continue

            if enable_trail and legs["B"] is not None and atr14 > 0 and trail_mult > 0 and engine_close > 0:
                legB = legs["B"]; dist = trail_mult * atr14
                if legB["side"] == 1:
                    new_sl = engine_close - dist
                    if _set_trailing_sl(legB["ticket"], new_sl, legB["stop"], "BUY", mt5):
                        legB["stop"] = new_sl
                else:
                    new_sl = engine_close + dist
                    if _set_trailing_sl(legB["ticket"], new_sl, legB["stop"], "SELL", mt5):
                        legB["stop"] = new_sl

            is_flat = legs["A"] is None and legs["B"] is None and legs["C"] is None
            leg_a_closed_b_open = legs["A"] is None and legs["B"] is not None and leg_a_profit_hit
            leg_b_closed_a_open = legs["B"] is None and legs["A"] is not None and leg_b_profit_hit
            can_scale_in = legs["C"] is None and (leg_a_closed_b_open or leg_b_closed_a_open)

            if signal != 0 and (is_flat or can_scale_in) and atr14 > 0:
                info = mt5.symbol_info(symbol); tick = mt5.symbol_info_tick(symbol)
                if info is None or tick is None:
                    time.sleep(POLL_INTERVAL_SEC); continue
                spread_price = info.spread * info.point
                if spread_price > 0.8 * atr14:
                    logger.info("[%s] Spread %.3f > 0.8*ATR %.3f — skip entry.", tf_up, spread_price, 0.8 * atr14)
                    time.sleep(POLL_INTERVAL_SEC); continue
                all_pos = mt5.positions_get(symbol=symbol) or []
                if sum(1 for p in all_pos if p.magic in ALL_GRX_MAGICS) >= 4:
                    logger.info("[%s] Global exposure cap (4) reached — skip entry.", tf_up)
                    time.sleep(POLL_INTERVAL_SEC); continue

                side  = 1 if signal > 0 else -1
                guard = (0.85 if side == 1 else 0.75) if is_m5 else 0.65
                dev   = int(TF_DEFAULT_DEV.get(tf_up, DEFAULT_DEVIATION))
                otype = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
                entry_ref = tick.ask if side == 1 else tick.bid
                now_min = int(datetime.now(timezone.utc).timestamp() // 60)

                if is_flat:
                    actual_stop_dist = atr_stop * atr14 * guard
                    runner_tp_dist   = atr_target * atr14 * guard
                    leg_a_tp_dist    = leg_a_atr_target * atr14
                    lot_a, lot_b = _leg_lots(actual_stop_dist)
                    if side == 1:
                        stop_px = entry_ref - actual_stop_dist
                        a_tp = entry_ref + leg_a_tp_dist if leg_a_atr_target > 0 else 0.0
                        b_tp = entry_ref + runner_tp_dist if enable_fixed_tp else 0.0
                    else:
                        stop_px = entry_ref + actual_stop_dist
                        a_tp = entry_ref - leg_a_tp_dist if leg_a_atr_target > 0 else 0.0
                        b_tp = entry_ref - runner_tp_dist if enable_fixed_tp else 0.0
                    if not check_margin(symbol, lot_a + lot_b, otype, entry_ref):
                        logger.warning("[%s] Margin check failed — skip entry.", tf_up)
                        time.sleep(POLL_INTERVAL_SEC); continue
                    resA = send_market_order(symbol, otype, lot_a, round(stop_px, 2), round(a_tp, 2), dev, magic, f"GRX_{tf_up}_A")
                    if resA.get("success"):
                        legs["A"] = {"ticket": resA["order"], "side": side, "entry": entry_ref,
                                     "stop": round(stop_px, 2), "tp": round(a_tp, 2), "entry_min": now_min, "lot": lot_a}
                    resB = send_market_order(symbol, otype, lot_b, round(stop_px, 2), round(b_tp, 2), dev, magic, f"GRX_{tf_up}_B")
                    if resB.get("success"):
                        legs["B"] = {"ticket": resB["order"], "side": side, "entry": entry_ref,
                                     "stop": round(stop_px, 2), "tp": round(b_tp, 2), "entry_min": now_min, "lot": lot_b}
                    leg_a_profit_hit = False; leg_b_profit_hit = False
                    side_txt = "BUY" if side == 1 else "SELL"
                    b_desc = f"TP {b_tp:.2f}" if enable_fixed_tp else "runner (no TP)"
                    send_telegram_msg(
                        f"[{tf_up}] {side_txt} opened (legs A+B)\n"
                        f"entry~{entry_ref:.2f}  SL {stop_px:.2f}  guard {guard}\n"
                        f"legA TP {a_tp:.2f} lot {lot_a}  |  legB {b_desc} lot {lot_b}\n"
                        f"atr {atr14:.2f}  exit_model {exit_model}"
                    )
                elif can_scale_in:
                    runner = legs["B"] if leg_a_closed_b_open else legs["A"]
                    if side != runner["side"]:
                        time.sleep(POLL_INTERVAL_SEC); continue
                    stop_dist_c = 0.5 * atr14 * guard
                    tp_dist_c   = 0.5 * atr14 * guard
                    la, lb = _leg_lots(stop_dist_c)
                    lot_c = la if leg_a_closed_b_open else lb
                    if side == 1:
                        stop_px = entry_ref - stop_dist_c; c_tp = entry_ref + tp_dist_c
                    else:
                        stop_px = entry_ref + stop_dist_c; c_tp = entry_ref - tp_dist_c
                    if not check_margin(symbol, lot_c, otype, entry_ref):
                        time.sleep(POLL_INTERVAL_SEC); continue
                    resC = send_market_order(symbol, otype, lot_c, round(stop_px, 2), round(c_tp, 2), dev, magic, f"GRX_{tf_up}_C")
                    if resC.get("success"):
                        legs["C"] = {"ticket": resC["order"], "side": side, "entry": entry_ref,
                                     "stop": round(stop_px, 2), "tp": round(c_tp, 2), "entry_min": now_min, "lot": lot_c}
                        send_telegram_msg(f"[{tf_up}] Scale-in leg C {('BUY' if side==1 else 'SELL')} lot {lot_c}  SL {stop_px:.2f}  TP {c_tp:.2f}")

            time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("[%s] Interrupted — exiting live loop.", tf_up)
            break
        except Exception as exc:
            logger.exception("[%s] Loop iteration error: %s", tf_up, exc)
            time.sleep(POLL_INTERVAL_SEC)

