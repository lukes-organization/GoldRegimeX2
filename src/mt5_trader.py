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
from src.processor import (
    TF_CONFIG,
    USDCHF_MASTER_PATH,
    load_usdchf_data,
    compute_log_returns,
    kalman_smooth,
    compute_volatility,
    compute_rsi,
    compute_atr,
)
from src.engine_hmm import load_model as load_hmm, predict_states, get_model_path as hmm_model_path, MODEL_PATH as HMM_GENERIC_PATH
from src.engine_xgb import (
    load_xgb_ensemble, get_predictions_ensemble, assign_vol_bucket, FEATURE_COLS,
    get_ensemble_path, ENSEMBLE_PKL_PATH as XGB_GENERIC_PATH,
)
from src.risk_manager import AdaptiveRiskManager, CENT_MULTIPLIER

logger = setup_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOL               = "XAUUSD"
MAGIC_NUMBER                 = 123456   # must match GoldRegimeX.mq5 MagicNumber
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

# Quick-profit target for M5 scalping: close early when floating P&L reaches
# this USD amount rather than waiting for the full TP.  Re-entry is allowed on
# the next bar if signal conditions are still met.  Set to None to disable.
QUICK_PROFIT_TARGET_M5 = 4.0

# Trailing P&L stop ("Chop Buffer"): once a position's peak floating P&L has
# reached TRAILING_ACTIVATION_USD, close it if P&L drops back below
# peak × TRAILING_DRAWDOWN_PCT.  Catches stalling/reversing scalps that never
# reach the fixed target, bypassing the 1-bar HMM state detection delay.
TRAILING_ACTIVATION_USD  = 2.0   # start protecting once P&L first exceeds this
TRAILING_DRAWDOWN_PCT    = 0.50  # close if P&L falls below 50 % of observed peak

# ── Per-timeframe signal thresholds and order parameters ──────────────────────
TF_PROB_THRESHOLD  = {"M5": 0.52, "M15": 0.54, "H1": 0.54}   # fallback — Optuna value used when available
TF_SHORT_THRESHOLD = {"M5": 0.48, "M15": 0.46, "H1": 0.46}   # fallback only
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

# Lazy MT5 timeframe map

# ── USDCHF fallback cache ─────────────────────────────────────────────────────
# When the live USDCHF bar fetch fails (e.g. symbol not subscribed), carry the
# last known return from the master CSV — far better than assuming 0.0.
_USDCHF_FALLBACK_CACHE: float | None = None

def _get_usdchf_fallback() -> float:
    """Return the last known USDCHF log return from the master CSV (cached)."""
    global _USDCHF_FALLBACK_CACHE
    if _USDCHF_FALLBACK_CACHE is None:
        try:
            usdchf_df = load_usdchf_data(USDCHF_MASTER_PATH)
            if usdchf_df is not None:
                _USDCHF_FALLBACK_CACHE = float(usdchf_df["usdchf_log_return"].dropna().iloc[-1])
            else:
                _USDCHF_FALLBACK_CACHE = 0.0
        except Exception:
            _USDCHF_FALLBACK_CACHE = 0.0
    return _USDCHF_FALLBACK_CACHE
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


def _tp_multipliers(tf: str, hmm_state: int) -> list[float]:
    """Return the TP multiplier list for this TF and HMM state."""
    cfg = TF_TP_CONFIG.get(tf.upper(), TF_TP_CONFIG["H1"])
    return cfg["chop"] if hmm_state == CHOP_STATE else cfg["trending"]


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

def _fetch_usdchf_log_return(tf_mt5: int, mt5) -> float | None:
    """Fetch the most recently completed USDCHF bar log return for live inference.

    USDCHF is always available on Headway as a standard Forex pair — unlike
    DXY/USDX which many brokers don't carry.  Returns ``None`` only if the
    data call fails (symbol not subscribed, no recent bars, etc.).
    """
    rates = mt5.copy_rates_from_pos("USDCHF", tf_mt5, 1, 3)   # 3 completed bars
    if rates is None or len(rates) < 2:
        return None
    closes = [r["close"] for r in rates]
    if closes[-2] <= 0:
        return None
    return float(np.log(closes[-1] / closes[-2]))


def compute_deviation(model_hmm, current_state: int, tf: str = "H1") -> int:
    """Select order deviation based on regime stability and timeframe.

    A low self-transition probability on the current HMM state means the regime
    is likely to flip next bar — a proxy for elevated market volatility.
    Deviation is widened to avoid requotes in those conditions.
    M5 uses a higher base deviation (30) than M15/H1 (20) for scalping fills.
    """
    self_prob    = model_hmm.transmat_[current_state, current_state]
    base_dev     = TF_DEFAULT_DEV.get(tf.upper(), 20)
    high_vol_dev = TF_HIGH_VOL_DEV.get(tf.upper(), 50)
    if self_prob < HIGH_VOL_SELF_TRANS_THRESHOLD:
        logger.debug(
            "High-vol deviation: state=%d  self_trans=%.3f < %.2f — using %d pts",
            current_state, self_prob, HIGH_VOL_SELF_TRANS_THRESHOLD, high_vol_dev,
        )
        return high_vol_dev
    return base_dev


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


def _close_position(ticket: int, mt5, comment: str = "GRX_close_chop") -> None:
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
        "magic":        MAGIC_NUMBER,
        "comment":      comment,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("Position closed (%s): ticket=%d", comment, ticket)
    else:
        logger.warning("Position close failed: ticket=%d  retcode=%s",
                       ticket, res.retcode if res else "None")


def _log_closed_pnl(tickets: list, mt5, broker: str = "headway_cent") -> None:
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
                f"{emoji} <b>Trade closed</b>  #{ticket}\n"
                f"{direction}  lot=<b>{lot:.2f}</b>  "
                f"entry: <b>{entry_px:.2f}</b> → exit: <b>{exit_px:.2f}</b>\n"
                f"Move: <b>{pts_str}</b>  |  "
                f"Realized P&L: <b>{pnl:+.2f} USD</b>  [{emoji} {tag}]"
            )
        except Exception as exc:
            logger.warning("Could not fetch P&L for ticket #%d: %s", ticket, exc)


def _build_live_df(
    symbol: str,
    tf_mt5: int,
    n_bars: int,
    obs_cov: float,
    trans_cov: float,
) -> pd.DataFrame:
    """Fetch the last *n_bars* completed bars and return a featurised DataFrame.

    The currently open (incomplete) bar is excluded via offset=1.
    Feature sequence mirrors processor.process_pipeline() exactly.
    """
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 1, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"copy_rates_from_pos returned no data: {mt5.last_error()}"
        )
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
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    df["log_return"]     = compute_log_returns(df["Close"])
    df["kalman_return"]  = kalman_smooth(df["log_return"].values, obs_cov, trans_cov)
    df["volatility"]     = compute_volatility(df["log_return"])
    df["rsi"]            = compute_rsi(df["Close"])
    df["rsi_slope"]      = df["rsi"].diff()
    df["atr_normalized"] = compute_atr(df)
    df.dropna(inplace=True)
    return df


def compute_live_features(
    symbol: str,
    tf_mt5: int,
    model_hmm,
    obs_cov: float,
    trans_cov: float,
    feature_cols: list | None = None,
    mt5=None,
):
    """Build the current-bar feature vector for XGBoost ensemble inference.

    Returns:
        features_df  (pd.DataFrame shape (1, n_features)): named feature row.
        hmm_state    (int):   current HMM regime index.
        atr_price    (float): ATR in price terms for SL calculation.

    ``feature_cols`` is loaded from the ensemble metadata so the live vector
    always matches what the models were trained on.  If ``usdchf_log_return`` is
    required and the fetch fails, the last known value from the master CSV is used.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = _build_live_df(symbol, tf_mt5, N_BARS_WARMUP, obs_cov, trans_cov)
    states = predict_states(model_hmm, df)

    current_state   = int(states[-1])
    rsi_slope       = float(df["rsi_slope"].iloc[-1])
    atr_normalized  = float(df["atr_normalized"].iloc[-1])
    prev_log_return = float(df["log_return"].iloc[-2])

    feature_dict = {
        "hmm_state":       float(current_state),
        "rsi_slope":       rsi_slope,
        "atr_normalized":  atr_normalized,
        "prev_log_return": prev_log_return,
    }

    if "usdchf_log_return" in feature_cols:
        if mt5 is not None:
            usdchf_ret = _fetch_usdchf_log_return(tf_mt5, mt5)
        else:
            usdchf_ret = None
        if usdchf_ret is None:
            usdchf_ret = _get_usdchf_fallback()
            logger.warning(
                "USDCHF return unavailable — using last known value %.6f as fallback.",
                usdchf_ret,
            )
        feature_dict["usdchf_log_return"] = usdchf_ret

    features_df = pd.DataFrame([feature_dict])[feature_cols]
    atr_price   = atr_normalized * float(df["Close"].iloc[-1])
    return features_df, current_state, atr_price


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
    prob_threshold_override: float = None,
    short_threshold_override: float = None,
    profit_target: float = None,
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
                       Defaults to QUICK_PROFIT_TARGET_M5 for M5, disabled for
                       other TFs.  Pass 0 to disable on M5 explicitly.
    """
    import MetaTrader5 as mt5
    from src.mt5_sync import connect_mt5, disconnect_mt5

    if not connect_mt5():
        raise ConnectionError("Could not connect to MT5 terminal.")

    try:
        _run_loop_inner(tf, broker, account_size, mt5,
                        prob_threshold_override, short_threshold_override,
                        profit_target)
    finally:
        disconnect_mt5()
        logger.info("Live loop terminated.  MT5 disconnected.")


def _run_loop_inner(tf: str, broker: str, account_size: float, mt5,
                    prob_threshold_override: float = None,
                    short_threshold_override: float = None,
                    profit_target: float = None) -> None:
    """Inner loop extracted to allow clean finally / disconnect in run_live_loop."""
    tf_mt5 = _get_tf_map()[tf.upper()]

    # Resolve live balance
    telemetry = get_account_telemetry()
    if account_size is None:
        account_size = _normalise_balance(telemetry["balance"], broker)
        logger.info(
            "Balance auto-detected: MT5 raw=%.2f  USD normalised=%.2f",
            telemetry["balance"], account_size,
        )

    display_account_info(trading_balance=account_size)

    # Load models — prefer broker+TF specific file; fall back to generic
    hmm_path = hmm_model_path(tf, broker)
    if not hmm_path.exists():
        hmm_path = HMM_GENERIC_PATH
    try:
        model_hmm = load_hmm(hmm_path)
    except FileNotFoundError:
        raise FileNotFoundError("HMM model not found. Run --mode train first.")

    xgb_path = get_ensemble_path(tf, broker)
    if not xgb_path.exists():
        xgb_path = XGB_GENERIC_PATH
    try:
        models_xgb, thresholds_xgb, xgb_meta = load_xgb_ensemble(xgb_path)
        feature_cols = xgb_meta.get("feature_cols", list(FEATURE_COLS))
    except FileNotFoundError:
        raise FileNotFoundError("XGB ensemble model not found. Run --mode train first.")

    # Resolve Kalman params from Optuna study or TF defaults
    try:
        from src.optimizer import get_best_params
        params    = get_best_params(balance=account_size, broker=broker, tf=tf)
        obs_cov   = params.get("obs_cov")
        trans_cov = params.get("trans_cov")
        prob_threshold_opt = params.get("prob_threshold")
    except Exception:
        params    = {}
        obs_cov   = None
        trans_cov = None
        prob_threshold_opt = None

    cfg_tf    = TF_CONFIG[tf.upper()]
    obs_cov   = obs_cov   if obs_cov   is not None else cfg_tf["obs_cov_default"]
    trans_cov = trans_cov if trans_cov is not None else cfg_tf["trans_cov_default"]

    # Use optimized thresholds if available; otherwise fall back to TF hardcoded values.
    # Both prob_threshold (BUY) and short_threshold (SELL) are now Optuna-tuned
    # so the optimizer finds the best asymmetric no-trade zone for each TF.
    if prob_threshold_opt is not None:
        prob_threshold  = prob_threshold_opt
        short_threshold = params.get("short_threshold", 1.0 - prob_threshold_opt)
    else:
        prob_threshold  = TF_PROB_THRESHOLD.get(tf.upper(), 0.65)
        short_threshold = TF_SHORT_THRESHOLD.get(tf.upper(), 0.35)
    # CLI override has highest priority — applied after Optuna resolution
    if prob_threshold_override is not None:
        prob_threshold  = prob_threshold_override
    if short_threshold_override is not None:
        short_threshold = short_threshold_override

    # Session state (persists across bars within a day)
    daily_trades  = 0
    last_bar_time = None
    current_day   = None
    # Tracks the tickets and entry context of the most recently placed signal
    # so the break-even SL logic can fire when TP1 closes position 1.
    signal_tracker = {"tickets": [], "entry_price": 0.0,
                      "direction": None, "tp1_hit": False,
                      "tp1_level": None, "guard_hit": False}
    # Per-ticket high-water-mark for trailing P&L stop  {ticket: peak_usd}
    peak_pnl_tracker: dict = {}

    arm = AdaptiveRiskManager(account_size, tf=tf, broker=broker)
    logger.info(
        "Live loop started — TF=%s  broker=%s  balance=$%.2f  %s",
        tf, broker, account_size, arm,
    )
    logger.info(
        "Signal thresholds — BUY>%.3f  SELL<%.3f  (source: %s)",
        prob_threshold, short_threshold,
        "override" if (prob_threshold_override or short_threshold_override)
        else ("optuna" if prob_threshold_opt is not None else "hardcoded"),
    )

    # Resolve quick-profit target: CLI override > M5 default > disabled
    if profit_target is None:
        profit_target = QUICK_PROFIT_TARGET_M5 if tf.upper() == "M5" else None
    elif profit_target <= 0:
        profit_target = None   # explicit CLI disable (--profit_target 0)
    if profit_target is not None:
        logger.info(
            "Hybrid Scalp Protection — Fixed target: $%.2f | "
            "Trailing guard: activation $%.2f / drawdown %.0f%%.",
            profit_target, TRAILING_ACTIVATION_USD, TRAILING_DRAWDOWN_PCT * 100,
        )

    # Warn if the MQL5 EA (same MAGIC_NUMBER) already has open positions.
    # Running both simultaneously causes Python to see EA positions as its own
    # and block all signal generation via the "Open position — holding" guard.
    _startup_positions = mt5.positions_get(symbol=DEFAULT_SYMBOL) or []
    _ea_conflict = [p for p in _startup_positions if p.magic == MAGIC_NUMBER]
    if _ea_conflict:
        logger.warning(
            "CONFLICT: %d open position(s) with MAGIC_NUMBER=%d detected at startup. "
            "These were likely placed by the MQL5 EA (GoldRegimeX.mq5). "
            "Running both EA and Python bridge simultaneously causes signal blocking. "
            "Detach the EA from the chart before continuing.",
            len(_ea_conflict), MAGIC_NUMBER,
        )

    while True:
        try:
            # ── 1. Daily reset at UTC midnight ────────────────────────────
            today = datetime.now(timezone.utc).date()
            if today != current_day:
                daily_trades = 0
                current_day  = today
                logger.info("New UTC day %s — session counter reset.", today)

            # ── 2. Bar-change detection ───────────────────────────────────
            bars = mt5.copy_rates_from_pos(DEFAULT_SYMBOL, tf_mt5, 1, 1)
            if bars is None or len(bars) == 0:
                time.sleep(POLL_INTERVAL_SEC)
                continue
            bar_time = bars[0]["time"]
            if bar_time == last_bar_time:
                # Between bars: detect position closures and report P&L immediately
                if signal_tracker["tickets"]:
                    open_set = {
                        p.ticket
                        for p in (mt5.positions_get(symbol=DEFAULT_SYMBOL) or [])
                        if p.magic == MAGIC_NUMBER
                    }
                    closed = [t for t in signal_tracker["tickets"] if t not in open_set]
                    if closed:
                        _log_closed_pnl(closed, mt5)
                        for _t in closed:
                            peak_pnl_tracker.pop(_t, None)
                        signal_tracker["tickets"] = [
                            t for t in signal_tracker["tickets"] if t in open_set
                        ]
                        if not signal_tracker["tickets"]:
                            signal_tracker["tp1_hit"] = False

                # ── Hybrid Scalp Protection (per-position) ────────────────
                # Runs every poll cycle (5 s) — bypasses the 5-min bar delay.
                # Condition A — Fixed target  : close when P&L >= profit_target
                # Condition B — Trailing guard: once P&L peaked >= $2, close if
                #               it pulls back to ≤ 50 % of that peak (chop buffer)
                if profit_target is not None and signal_tracker["tickets"]:
                    _pnl_divisor = 100 if broker == "headway_cent" else 1
                    _open_pos = [
                        p for p in (mt5.positions_get(symbol=DEFAULT_SYMBOL) or [])
                        if p.magic == MAGIC_NUMBER
                        and p.ticket in set(signal_tracker["tickets"])
                    ]
                    for _pos in _open_pos:
                        _cur  = _pos.profit / _pnl_divisor
                        _peak = peak_pnl_tracker.get(_pos.ticket, _cur)

                        # Update high-water mark
                        if _cur > _peak:
                            _peak = _cur
                            peak_pnl_tracker[_pos.ticket] = _peak

                        # Condition A: fixed scalp target (per position)
                        if _cur >= profit_target:
                            logger.info(
                                "Fixed scalp target #%d: P&L=+$%.2f >= $%.2f — closing.",
                                _pos.ticket, _cur, profit_target,
                            )
                            _close_position(_pos.ticket, mt5,
                                            comment="GRX_Fixed_Scalp_Target")
                            continue

                        # Condition B: trailing guard — activation $2, drawdown 50 %
                        if (_peak >= TRAILING_ACTIVATION_USD
                                and _cur <= _peak * TRAILING_DRAWDOWN_PCT):
                            logger.info(
                                "Trailing guard #%d: P&L=$%.2f <= 50%% of peak $%.2f "
                                "(activation $%.2f) — closing.",
                                _pos.ticket, _cur, _peak, TRAILING_ACTIVATION_USD,
                            )
                            _close_position(_pos.ticket, mt5,
                                            comment="GRX_Trailing_Profit_Guard")
                time.sleep(POLL_INTERVAL_SEC)
                continue
            last_bar_time = bar_time

            # ── 3. Refresh telemetry for margin/display (risk sizing always
            #        uses the caller-supplied account_size, not the live MT5
            #        balance, because demo accounts carry arbitrary balances).
            try:
                telemetry = get_account_telemetry()
            except Exception:
                telemetry = {}

            arm              = AdaptiveRiskManager(account_size, tf=tf, broker=broker)
            arm.daily_trades = daily_trades   # restore session count

            # ── 4. Compute live features + probability ────────────────────
            features_df, hmm_state, atr_price = compute_live_features(
                DEFAULT_SYMBOL, tf_mt5, model_hmm, obs_cov, trans_cov,
                feature_cols=feature_cols, mt5=mt5,
            )
            _, _probs = get_predictions_ensemble(models_xgb, thresholds_xgb, features_df)
            prob = float(_probs[0])

            # ── 5. Log bar info on every new bar (always visible) ─────────
            bar_str          = datetime.fromtimestamp(bar_time, timezone.utc).strftime("%Y-%m-%d %H:%M")
            state_lbl        = {0: "Bull", 1: "Bear", 2: "Chop"}.get(hmm_state, str(hmm_state))
            max_trades_today = arm.get_trade_limits(hmm_state)["max_daily_trades"]
            logger.info(
                "Bar %s | state=%s | prob=%.3f | trades=%d/%d",
                bar_str, hmm_state, prob,
                arm.daily_trades, max_trades_today,
            )
            send_telegram_msg(
                f"📊 <b>Bar</b> {bar_str} [{tf}]\n"
                f"Regime: <b>{state_lbl}</b> ({hmm_state})  |  "
                f"Prob: <b>{prob:.3f}</b>\n"
                f"Trades today: <b>{arm.daily_trades}/{max_trades_today}</b>"
            )

            # ── 6. Position management (always runs — P&L, break-even, chop-exit)
            if signal_tracker["tickets"]:
                open_set = {
                    p.ticket
                    for p in (mt5.positions_get(symbol=DEFAULT_SYMBOL) or [])
                    if p.magic == MAGIC_NUMBER
                }
                active = [t for t in signal_tracker["tickets"] if t in open_set]
                closed = [t for t in signal_tracker["tickets"] if t not in open_set]
                # Log P&L for any positions that closed since last bar
                if closed:
                    _log_closed_pnl(closed, mt5, broker=broker)
                # Profit guard: move SL to entry+spread when 70% to TP1 (all TFs)
                if active:
                    _apply_profit_guard(signal_tracker, mt5)
                # TP1 hit: first position gone → move runner's SL to break-even
                if len(active) < len(signal_tracker["tickets"]) and not signal_tracker["tp1_hit"]:
                    signal_tracker["tp1_hit"] = True
                    for ticket in active:
                        _move_sl_to_breakeven(ticket, signal_tracker["entry_price"], mt5)
                # Regime shifted to Chop while runner active → close immediately
                if hmm_state == CHOP_STATE and active:
                    for ticket in active:
                        _close_position(ticket, mt5)
                    logger.info("Runner(s) closed: Chop state shift while position open.")
                    active = []
                signal_tracker["tickets"] = active

            # ── 7. Session limit check ────────────────────────────────────
            if not arm.can_trade(hmm_state):
                logger.info(
                    "Session limit reached (state=%d  daily=%d/%d). Skipping bar.",
                    hmm_state, arm.daily_trades,
                    arm.get_trade_limits(hmm_state)["max_daily_trades"],
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 8. Skip new signals if a position is still open ──────────
            if has_open_position(DEFAULT_SYMBOL, MAGIC_NUMBER):
                logger.info(
                    "Open position — holding (prob=%.3f  state=%s).",
                    prob, state_lbl,
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 9. Signal routing — regime-aligned (Bull→BUY, Bear→SELL) ──
            # Mirrors compute_signals() in backtester exactly.
            # Chop generates no signal regardless of probability.
            if prob > prob_threshold and hmm_state == BULL_STATE:
                direction  = "BUY"
                order_type = mt5.ORDER_TYPE_BUY
            elif prob < short_threshold and hmm_state == BEAR_STATE:
                direction  = "SELL"
                order_type = mt5.ORDER_TYPE_SELL
            else:
                # Explain exactly why no signal fired
                if hmm_state == BULL_STATE:
                    reason = f"Bull state but prob={prob:.3f} not >{prob_threshold:.3f}"
                elif hmm_state == BEAR_STATE:
                    reason = f"Bear state but prob={prob:.3f} not <{short_threshold:.3f}"
                else:
                    reason = f"Chop state — no signal regardless of prob"
                logger.info("No signal (%s).", reason)
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 9. Position sizing ────────────────────────────────────────
            tp_mults      = _tp_multipliers(tf, hmm_state)
            limits        = arm.get_trade_limits(hmm_state)
            pos_per_trade = min(limits["pos_per_trade"], len(tp_mults))
            sl_distance   = max(atr_price * TF_ATR_MULTIPLIER.get(tf.upper(), 2.0), 0.01)
            lot_total     = arm.calculate_lot_size(stop_loss_pips=sl_distance)
            lot_per_pos   = max(0.01, round(lot_total / pos_per_trade, 2))

            # ── 10. Deviation (TF-specific + regime stability) ────────────
            deviation = compute_deviation(model_hmm, hmm_state, tf)

            # ── 11. SL / TP prices ────────────────────────────────────────
            tick = mt5.symbol_info_tick(DEFAULT_SYMBOL)
            if direction == "BUY":
                entry_price = tick.ask
                sl_price    = round(entry_price - sl_distance, 2)
                tp_levels   = [round(entry_price + sl_distance * m, 2) for m in tp_mults]
            else:
                entry_price = tick.bid
                sl_price    = round(entry_price + sl_distance, 2)
                tp_levels   = [round(entry_price - sl_distance * m, 2) for m in tp_mults]

            # ── 11b. Spread viability guard (M5 scalps + all TFs on standard) ─
            spread_ratio = MIN_SPREAD_RATIO.get(broker, 1.5)
            if tf.upper() == "M5" or broker == "standard":
                sym_info     = mt5.symbol_info(DEFAULT_SYMBOL)
                spread_price = (sym_info.spread * sym_info.point) if sym_info else 0.0
                tp1_distance = sl_distance * tp_mults[0]
                if spread_price > 0 and tp1_distance < spread_price * spread_ratio:
                    logger.warning(
                        "Unviable trade [%s/%s]: TP1=%.4f < %.1fx spread=%.4f. Skipping.",
                        tf, broker, tp1_distance, spread_ratio, spread_price,
                    )
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

            # ── 12. Margin check ──────────────────────────────────────────
            if not check_margin(DEFAULT_SYMBOL, lot_per_pos, order_type, entry_price):
                logger.warning(
                    "Insufficient margin for %s lot=%.2f. Skipping. Free margin: %.2f",
                    direction, lot_per_pos, telemetry.get("free_margin", 0),
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 13. Send order(s) with state-aware staged TPs ─────────────
            signal_tracker = {"tickets": [], "entry_price": entry_price,
                              "direction": direction, "tp1_hit": False,
                              "tp1_level": tp_levels[0], "guard_hit": False}
            peak_pnl_tracker = {}
            logger.info(
                "SIGNAL %s | state=%d | prob=%.3f | lot×%d=%.2f | sl=%.2f | tp=%s | dev=%d",
                direction, hmm_state, prob, pos_per_trade, lot_per_pos,
                sl_price, "/".join(f"{t:.2f}" for t in tp_levels), deviation,
            )
            for p in range(pos_per_trade):
                tp_price = tp_levels[p]
                comment  = f"GRX_{direction}_p{p+1}of{pos_per_trade}_s{hmm_state}_tp{p+1}"
                result = send_market_order(
                    symbol=DEFAULT_SYMBOL,
                    order_type=order_type,
                    lot=lot_per_pos,
                    sl=sl_price,
                    tp=tp_price,
                    deviation=deviation,
                    magic=MAGIC_NUMBER,
                    comment=comment,
                )
                if result["success"]:
                    signal_tracker["tickets"].append(result["order"])
                    daily_trades += 1
                    arm.log_trade()
                else:
                    logger.error(
                        "Order %d/%d failed — retcode=%d  %s",
                        p + 1, pos_per_trade,
                        result["retcode"], result["comment"],
                    )

            # Send one Telegram message per signal summarising all filled positions
            if signal_tracker["tickets"]:
                _emoji  = "🟢" if direction == "BUY" else "🔴"
                _tp_str = " / ".join(f"{t:.2f}" for t in tp_levels)
                _tix    = "  ".join(f"#{t}" for t in signal_tracker["tickets"])
                send_telegram_msg(
                    f"{_emoji} <b>Trade opened</b> [{tf}]\n"
                    f"<b>{direction}</b>  Regime: <b>{state_lbl}</b>  Prob: <b>{prob:.3f}</b>\n"
                    f"Lot: <b>{pos_per_trade}×{lot_per_pos:.2f}</b>  "
                    f"SL: <b>{sl_price:.2f}</b>  TP: <b>{_tp_str}</b>\n"
                    f"Tickets: {_tix}"
                )

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping live loop.")
            break
        except Exception as exc:
            logger.error(
                "Unhandled loop error: %s — sleeping 30 s before retry.",
                exc, exc_info=True,
            )
            time.sleep(30)
