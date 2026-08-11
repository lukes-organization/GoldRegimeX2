"""src/mt5_trader.py -- live/demo execution engine (the mirror of the backtester).

This is the seam mt5_live_app.py imports.  Its entries/exits are driven by the
SAME deployed signal used by the Explorer backtester (strategy_backtest.latest_
signal) and the SAME deployed sizing policy (risk_manager.AdaptiveRiskManager
from the exported bundle), so a position taken live is the position the
backtester would have taken on that bar -- a true 1:1 mirror.

All MT5 access goes through the ``mt5`` handle the app passes in (a stoppable
wrapper that raises KeyboardInterrupt when the user hits Stop).  On the user's
machine this is the real MetaTrader5 module after a successful login; entries are
only ever reached AFTER the app has authenticated login/password/server.

PAPER-TRADE on a demo login first.
"""
from __future__ import annotations
import time
from datetime import datetime

from .risk_manager import AdaptiveRiskManager, broker_cent_multiplier, CENT_MULTIPLIER
from .strategy_backtest import load_live_bundle, latest_signal

DEFAULT_SYMBOL = "XAUUSD"

# One magic per timeframe cycle (kept in sync with mt5_live_app._fetch_trades).
MAGIC_BY_TF = {"H1": 123456, "M15": 123457, "M5": 123458}
ALL_GRX_MAGICS = frozenset(MAGIC_BY_TF.values())

POLL_SECONDS = 3.0

# Deployed exit geometry (Explorer config cell 3).  Used when the strategy
# base_params do not carry an explicit stop/target.
STOP_LOSS_PIPS = 15.0
RR_MULT = 2.0
PIP_SIZE_PRICE = 0.10
DEVIATION_POINTS = 20


def _tf_const(mt5, tf):
    return {
        "H1": getattr(mt5, "TIMEFRAME_H1", 16385),
        "M15": getattr(mt5, "TIMEFRAME_M15", 15),
        "M5": getattr(mt5, "TIMEFRAME_M5", 5),
    }.get(str(tf).upper())


def _ensure_symbol(mt5, symbol):
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return False
        if not getattr(info, "visible", True):
            mt5.symbol_select(symbol, True)
        return True
    except Exception:
        return False


def _last_closed_bar_time(mt5, symbol, tf):
    try:
        const = _tf_const(mt5, tf)
        rates = mt5.copy_rates_from_pos(symbol, const, 1, 1)  # index 1 = last CLOSED bar
        if rates is None or len(rates) == 0:
            return None
        return int(rates[-1]["time"])
    except Exception:
        return None


def _count_positions(mt5, symbol, magic):
    try:
        pos = mt5.positions_get(symbol=symbol)
        if not pos:
            return 0
        return sum(1 for p in pos if int(getattr(p, "magic", 0)) == int(magic))
    except Exception:
        return 0


def _realized_profit_cents(mt5, magics, cent_mult):
    """Sum today's closed-deal profit for our magics, expressed in account-cents."""
    try:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(start, datetime.now())
        if not deals:
            return 0.0
        total = sum(float(getattr(d, "profit", 0.0)) for d in deals
                    if int(getattr(d, "magic", 0)) in magics)
        return total * (cent_mult if cent_mult else 1.0)
    except Exception:
        return 0.0


def _send_order(mt5, symbol, is_buy, lot, sl, tp, magic, comment):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    price = tick.ask if is_buy else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": DEVIATION_POINTS,
        "magic": int(magic),
        "comment": comment,
        "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
        "type_filling": getattr(mt5, "ORDER_FILLING_IOC", 1),
    }
    return mt5.order_send(request)


def _stop_target(sig, is_buy):
    """Stop/target prices for a new entry, mirroring the deployed geometry.
    Prefers an ATR-based stop from the live signal, else fixed pips; target at
    RR_MULT x stop distance."""
    close = float(sig.get("close") or 0.0)
    atr_v = sig.get("atr")
    bp = sig.get("base_params", {}) or {}
    stop_dist = None
    for key in ("atr_stop_mult", "leg_c_atr_stop", "stop_atr"):
        if key in bp and atr_v and atr_v == atr_v:  # atr_v not NaN
            try:
                stop_dist = float(bp[key]) * float(atr_v)
                break
            except Exception:
                pass
    if not stop_dist or stop_dist <= 0:
        stop_dist = STOP_LOSS_PIPS * PIP_SIZE_PRICE
    rr = float(bp.get("rr_mult", bp.get("reward_risk", RR_MULT)) or RR_MULT)
    if is_buy:
        return close - stop_dist, close + rr * stop_dist
    return close + stop_dist, close - rr * stop_dist


def get_account_telemetry():
    """Return a flat dict of account fields for the app's Account panel.
    Called with no args -- reads the already-initialised MetaTrader5 session."""
    try:
        import MetaTrader5 as mt5
    except Exception:
        return {}
    try:
        info = mt5.account_info()
    except Exception:
        return {}
    if info is None:
        return {}
    return {
        "login": getattr(info, "login", "-"),
        "server": getattr(info, "server", "-"),
        "company": getattr(info, "company", "-"),
        "currency": getattr(info, "currency", "-"),
        "trade_mode": getattr(info, "trade_mode", "-"),
        "balance": getattr(info, "balance", 0.0),
        "equity": getattr(info, "equity", 0.0),
        "margin": getattr(info, "margin", 0.0),
        "free_margin": getattr(info, "margin_free", 0.0),
        "margin_level": getattr(info, "margin_level", 0.0),
    }


def _run_loop_inner(tf, broker, account_size, mt5, profit_target=None, use_tiered=False):
    """Bar-by-bar live/demo loop.  Runs until the app's stoppable mt5 wrapper
    raises KeyboardInterrupt (Stop button) or an unrecoverable error occurs.

    Auto-enters and manages positions using the deployed signal + sizing so the
    live behaviour is a 1:1 mirror of the Explorer backtester.
    """
    tf = str(tf).upper()
    symbol = DEFAULT_SYMBOL
    magic = MAGIC_BY_TF.get(tf, MAGIC_BY_TF["M5"])
    cent_mult = broker_cent_multiplier(broker)
    try:
        bundle = load_live_bundle()
    except Exception as e:
        print("[mt5_trader] cannot start: %s" % e)
        return
    rm = AdaptiveRiskManager.from_bundle(bundle)
    _ensure_symbol(mt5, symbol)
    last_bar = None
    print("[mt5_trader] live loop started  tf=%s broker=%s symbol=%s magic=%d"
          % (tf, broker, symbol, magic))
    try:
        while True:
            bar_time = _last_closed_bar_time(mt5, symbol, tf)
            if bar_time is not None and bar_time != last_bar:
                last_bar = bar_time
                sig = latest_signal(tf, bundle=bundle)
                if sig and sig.get("reason") == "ok" and int(sig.get("signal", 0)) != 0:
                    open_n = _count_positions(mt5, symbol, magic)
                    if open_n < rm.max_positions:
                        realized = _realized_profit_cents(mt5, ALL_GRX_MAGICS, cent_mult)
                        lots = rm.leg_lots(realized)
                        is_buy = int(sig["signal"]) > 0
                        sl, tp = _stop_target(sig, is_buy)
                        for i, lot in enumerate(lots[open_n:], start=open_n):
                            res = _send_order(mt5, symbol, is_buy, lot, sl, tp, magic,
                                              "GRX-%s-leg%d" % (tf, i + 1))
                            rc = getattr(res, "retcode", None)
                            print("[mt5_trader] %s leg%d %.2f @ sl=%.2f tp=%.2f -> %s"
                                  % ("BUY" if is_buy else "SELL", i + 1, lot, sl, tp, rc))
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("[mt5_trader] stop requested -- loop exiting (open positions are left as-is).")
        return
    except Exception as e:
        print("[mt5_trader] loop stopped on error: %s" % e)
        return


# Public convenience wrapper (used by main.py smoke checks / scripts).
def run_live_loop(tf="M5", broker="headway_cent", account_size=None):
    try:
        import MetaTrader5 as mt5
    except Exception as e:
        print("[mt5_trader] MetaTrader5 not available: %s" % e)
        return
    _run_loop_inner(tf, broker, account_size, mt5, None, use_tiered=False)
