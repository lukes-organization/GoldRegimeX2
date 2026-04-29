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
    compute_gmm_vol_cluster,
    load_gmm_model,
    load_feature_scaler,
    CONTINUOUS_FEATURE_COLS,
)
from src.engine_hmm import load_model as load_hmm, predict_states, get_model_path as hmm_model_path, MODEL_PATH as HMM_GENERIC_PATH, STATE_NAMES_2, STATE_NAMES_3, STATE_NAMES_4
from src.engine_xgb import (
    load_xgb_ensemble, get_predictions_ensemble, assign_vol_bucket, FEATURE_COLS,
    get_ensemble_path, ENSEMBLE_PKL_PATH as XGB_GENERIC_PATH,
    load_regime_classifiers, predict_regime_proba,
)
from src.risk_manager import AdaptiveRiskManager, BROKER_CONFIGS, CENT_MULTIPLIER, DailyEquityGate
from src.signal_evaluator import SignalEvaluator
from src.rcev_scorer import RCEVScorer, get_rcev_path, RCEV_DEFAULT_THRESHOLDS

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

# ── Regime-aware TP/SL multipliers (all values are direct ATR multiples) ──────
# Used by calculate_tp_sl() when RCEV signal mode is active.
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


def calculate_tp_sl(
    tf: str,
    regime: int,
    atr: float,
    entry_price: float,
    is_buy: bool,
) -> tuple:
    """Calculate regime-aware TP and SL price levels.

    Args:
        tf:          Timeframe string (H1, M15, M5).
        regime:      HMM state (0=Bull, 1=Bear, 2/3=Chop).
        atr:         Current ATR value in price units.
        entry_price: Trade entry price.
        is_buy:      True for BUY, False for SELL.

    Returns:
        ``(sl, tp1, tp2)`` — tp2 is None when not applicable.
    """
    cfg     = TP_SL_CONFIG.get(tf.upper(), TP_SL_CONFIG["H1"])
    r_type  = "trend" if regime in (0, 1) else "chop"
    params  = cfg[r_type]

    sl_dist  = params["sl_mult"]  * atr
    tp1_dist = params["tp1_mult"] * atr
    tp2_mult = params["tp2_mult"]

    if is_buy:
        sl  = entry_price - sl_dist
        tp1 = entry_price + tp1_dist
        tp2 = (entry_price + tp2_mult * atr) if tp2_mult else None
    else:
        sl  = entry_price + sl_dist
        tp1 = entry_price - tp1_dist
        tp2 = (entry_price - tp2_mult * atr) if tp2_mult else None

    return round(sl, 2), round(tp1, 2), (round(tp2, 2) if tp2 is not None else None)

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
    """Return the TP multiplier list for this TF and HMM state.

    Any Chop state (2 or 3 in 4-state model) uses the tighter chop TPs —
    this covers both trend-regime and mean-reversion trades that open in Chop.
    """
    cfg = TF_TP_CONFIG.get(tf.upper(), TF_TP_CONFIG["H1"])
    return cfg["chop"] if hmm_state >= CHOP_STATE else cfg["trending"]


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
    tf: str = "H1",
    broker: str = "headway_cent",
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
                "USDCHF live fetch failed — falling back to last value in "
                "USDCHF_master.csv (%.6f).  This value is STALE (CSV ends "
                "31/12/2025).  XGBoost is running on outdated USD-strength data. "
                "Export a fresh USDCHF CSV from MT5 and run --mode consolidate "
                "to restore live accuracy.",
                usdchf_ret,
            )
        else:
            logger.debug("USDCHF live return: %.6f (source: MT5 bar)", usdchf_ret)
        feature_dict["usdchf_log_return"] = usdchf_ret

    if "gmm_vol_cluster" in feature_cols:
        try:
            _gmm, _scaler = load_gmm_model(tf, broker)
            _cluster = compute_gmm_vol_cluster(
                df["volatility"].values, fitted_gmm=_gmm, fitted_scaler=_scaler
            )
            feature_dict["gmm_vol_cluster"] = float(_cluster[-1])
            df["gmm_vol_cluster"] = _cluster.astype(float)   # LSTM needs this column
        except FileNotFoundError:
            logger.warning("GMM model missing for [%s/%s] — using cluster=0", tf, broker)
            feature_dict["gmm_vol_cluster"] = 0.0
            df["gmm_vol_cluster"] = 0.0                       # LSTM fallback

    features_df = pd.DataFrame([feature_dict])[feature_cols]

    # Apply the 10-year feature scaler so live values match the training distribution
    cont_cols = [c for c in CONTINUOUS_FEATURE_COLS if c in features_df.columns]
    try:
        _feat_scaler = load_feature_scaler(tf, broker)
        features_df[cont_cols] = _feat_scaler.transform(features_df[cont_cols])
    except FileNotFoundError:
        logger.warning(
            "Feature scaler not found for [%s/%s] — running without scaling. "
            "Re-train with --mode train to generate it.",
            tf, broker,
        )

    # NaN/Inf guard: abort this bar rather than feed garbage to XGBoost
    _bad = features_df.isnull().any(axis=1) | (np.isinf(features_df.values).any(axis=1))
    if _bad.any():
        bad_cols = features_df.columns[features_df.isnull().any() | np.isinf(features_df.values).any(axis=0)].tolist()
        raise ValueError(f"Live feature NaN/Inf in columns {bad_cols} — skipping bar.")

    atr_price   = atr_normalized * float(df["Close"].iloc[-1])
    # Return raw (pre-scaler) atr_normalized separately so the live ER filter
    # can compare it against spread_frac without using the scaled value.
    # raw_df is the full live DataFrame (200 bars) passed to the TCN confidence scorer.
    return features_df, current_state, atr_price, atr_normalized, df


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

def _update_regime_stability(tracker: dict, hmm_state: int) -> dict:
    """Update per-bar regime stability state and return a stability info dict.

    The *tracker* dict is mutated in-place and must be initialised once per
    loop run as::

        tracker = {"current_state": None, "consecutive_bars": 0,
                   "previous_state": None, "exited_from_state": None}
    """
    if tracker["current_state"] is None:
        tracker["current_state"]   = hmm_state
        tracker["consecutive_bars"] = 1
        tracker["previous_state"]  = hmm_state
        tracker["exited_from_state"] = None
    elif hmm_state != tracker["current_state"]:
        logger.info(
            "[REGIME CHANGE] %d → %d after %d bars",
            tracker["current_state"], hmm_state, tracker["consecutive_bars"],
        )
        tracker["exited_from_state"] = tracker["current_state"]
        tracker["previous_state"]    = tracker["current_state"]
        tracker["current_state"]     = hmm_state
        tracker["consecutive_bars"]  = 1
    else:
        tracker["consecutive_bars"] += 1

    is_chop = hmm_state in (2, 3)
    return {
        "is_chop":              is_chop,
        "consecutive_bars":     tracker["consecutive_bars"],
        "is_stable_chop":       is_chop and tracker["consecutive_bars"] >= 3,
        "just_entered_state":   tracker["consecutive_bars"] == 1,
        "regime_changed_this_bar": tracker["consecutive_bars"] == 1,
        "exited_from_state":    tracker["exited_from_state"],
        "previous_state":       tracker["previous_state"],
    }


def _get_transition_prob(model_hmm, hmm_state: int) -> float:
    """Return P(state → state) from the HMM transition matrix (self-transition)."""
    try:
        transmat = model_hmm.transmat_
        if hmm_state < transmat.shape[0]:
            return float(transmat[hmm_state, hmm_state])
    except Exception:
        pass
    return 0.70


def _calculate_bb_position(close_prices: np.ndarray, period: int = 20,
                            num_std: float = 2.0) -> float:
    """Return price position within Bollinger Bands: 0 = lower band, 1 = upper band.

    Returns 0.5 (neutral) if there are insufficient bars or the band is too narrow.
    """
    if len(close_prices) < period:
        return 0.5
    try:
        sma    = np.mean(close_prices[-period:])
        std    = np.std(close_prices[-period:])
        upper  = sma + num_std * std
        lower  = sma - num_std * std
        bw     = upper - lower
        if bw < 1e-6:
            return 0.5
        pos = (close_prices[-1] - lower) / bw
        return float(np.clip(pos, 0.0, 1.0))
    except Exception:
        return 0.5


def run_live_loop(
    tf: str          = "H1",
    broker: str      = "headway_cent",
    account_size: float = None,
    profit_target: float = None,
    use_tiered: bool = False,
    rcev_threshold: float = None,
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
                       Defaults to PROFIT_ACTIVATION_USD for all TFs (legacy param,
                       other TFs.  Pass 0 to disable on M5 explicitly.
        use_tiered:    When True, pass tiered Z-Score override to SignalEvaluator
                       so strong XGBoost conviction can reduce the Z cutoff
                       (floor 1.0).  MR signals are unaffected.
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

    # Resolve TF-specific magic number — each TF instance must use its own number
    # so H1/M15/M5 running simultaneously don't see each other's positions.
    magic = TF_MAGIC_MAP.get(tf.upper(), TF_MAGIC_MAP["H1"])
    logger.info("Magic number for [%s]: %d", tf.upper(), magic)

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

    # Load regime statistics and build the adaptive signal evaluator
    regime_stats = xgb_meta.get("regime_stats", {})

    # Resolve Kalman params and Z-cutoff from Optuna study or TF defaults
    try:
        from src.optimizer import get_best_params
        params    = get_best_params(balance=account_size, broker=broker, tf=tf)
        obs_cov   = params.get("obs_cov")
        trans_cov = params.get("trans_cov")
    except Exception:
        params    = {}
        obs_cov   = None
        trans_cov = None

    _z_cut = params.get("z_cutoff_bull")
    _eval_cfg = {"Z_CUTOFF_BULL": _z_cut, "Z_CUTOFF_BEAR": -_z_cut} if _z_cut else None
    signal_evaluator = SignalEvaluator(regime_stats, tf=tf, config=_eval_cfg)

    # Load optional TCN confidence scorer — absent until --mode train_tcn is run
    from src.engine_tcn import load_tcn_classifier as _load_tcn_clf
    tcn_classifier = _load_tcn_clf(tf, broker)
    if tcn_classifier is not None:
        logger.info(
            "TCN confidence model loaded [%s/%s] — seq_len=%d  n_states=%d.",
            tf, broker, tcn_classifier.seq_len, tcn_classifier.n_states,
        )
    else:
        logger.info(
            "TCN confidence model not found [%s/%s] — static Z cutoffs active. "
            "Run --mode train_tcn to enable dynamic Z-Score adjustment.",
            tf, broker,
        )
    if regime_stats:
        logger.info(
            "Z-Score signal mode: regime stats loaded for %d states", len(regime_stats)
        )
        for _s, _st in regime_stats.items():
            logger.info(
                "  State %d: mean=%.4f  std=%.4f  n=%d",
                _s, _st["mean"], _st["std"], _st["count"],
            )
    else:
        logger.warning(
            "No regime_stats found in model — fixed-threshold fallback active. "
            "Re-train to enable Z-Score calibration."
        )

    # ── RCEV scorer (optional — Z-Score fallback when absent) ─────────────
    _rcev_path = get_rcev_path(tf, broker)
    rcev_scorer_live = RCEVScorer.from_file(_rcev_path, broker=broker)
    _rcev_threshold  = (
        rcev_threshold
        if rcev_threshold is not None
        else RCEV_DEFAULT_THRESHOLDS.get(tf.upper(), 0.50)
    )
    if rcev_scorer_live is not None:
        logger.info(
            "[RCEV] Scorer loaded [%s/%s] — threshold=$%.2f  "
            "(pass --rcev_threshold to override during live).",
            tf, broker, _rcev_threshold,
        )
    else:
        logger.info(
            "[RCEV] No calibration found [%s/%s] — Z-Score fallback active. "
            "Run --mode train to enable RCEV.",
            tf, broker,
        )

    # ── Per-regime XGBoost classifiers (optional) ─────────────────────────
    _xgb_base_path  = get_ensemble_path(tf, broker)
    regime_models_live = load_regime_classifiers(_xgb_base_path)
    if regime_models_live is not None:
        logger.info("[RCEV] Per-regime classifiers loaded [%s/%s].", tf, broker)
    else:
        logger.info(
            "[RCEV] Per-regime classifiers not found [%s/%s] — "
            "uniform 0.33/0.33/0.33 distribution used when RCEV is active.",
            tf, broker,
        )

    cfg_tf    = TF_CONFIG[tf.upper()]
    obs_cov   = obs_cov   if obs_cov   is not None else cfg_tf["obs_cov_default"]
    trans_cov = trans_cov if trans_cov is not None else cfg_tf["trans_cov_default"]

    # ── TCN startup health check ───────────────────────────────────────────
    if tcn_classifier is not None:
        _hc_ok, _hc_msg = tcn_classifier.health_check()
        if not _hc_ok:
            logger.warning(
                "[TCN HEALTH] FAILED (%s) — using static Z cutoffs. "
                "Retrain with --mode train_tcn.",
                _hc_msg,
            )
            tcn_classifier = None
        else:
            # Quick multiplier probe on real warmup data
            try:
                _hc_df = _build_live_df(DEFAULT_SYMBOL, tf_mt5, 150, obs_cov, trans_cov)
                try:
                    _hc_gmm, _hc_gscaler = load_gmm_model(tf, broker)
                    _hc_df["gmm_vol_cluster"] = compute_gmm_vol_cluster(
                        _hc_df["volatility"].values,
                        fitted_gmm=_hc_gmm, fitted_scaler=_hc_gscaler,
                    ).astype(float)
                except Exception:
                    _hc_df["gmm_vol_cluster"] = 0.0
                _hc_mult = tcn_classifier.predict_confidence(_hc_df)
                if _hc_mult is not None:
                    logger.info(
                        "[TCN HEALTH] OK — multiplier=%.2f  (Z %.0f%% of base)",
                        _hc_mult, _hc_mult * 100,
                    )
                else:
                    logger.warning(
                        "[TCN HEALTH] warmup probe returned None — "
                        "using static Z cutoffs."
                    )
                    tcn_classifier = None
            except Exception as _hc_exc:
                logger.warning("[TCN HEALTH] warmup probe error: %s", _hc_exc)
                tcn_classifier = None

    # Session state (persists across bars within a day)
    last_bar_time = None
    current_day   = None
    # Tracks the tickets and entry context of the most recently placed signal
    # so the break-even SL logic can fire when TP1 closes position 1.
    signal_tracker = {"tickets": [], "entry_price": 0.0,
                      "direction": None, "tp1_hit": False,
                      "tp1_level": None, "guard_hit": False,
                      "signal_type": None,
                      "atr_price": 0.0}   # cached price-denom ATR for between-bar trail
    # Per-ticket ATR trail state: {ticket: {"activated": bool, "partial_done": bool, "current_sl": float}}
    atr_state_tracker: dict = {}
    # Deprecated peak-pnl tracker kept for cleanup compat (atr_state_tracker supersedes it)
    peak_pnl_tracker: dict = {}
    # Two-sided equity gate: loss side (5%) + Trailing Daily Equity Lock (profit side)
    equity_gate = DailyEquityGate(tf=tf)
    equity_gate.reset_day(account_size)

    # Regime stability tracker for adaptive Z-Score gating
    regime_stability_tracker: dict = {
        "current_state":    None,
        "consecutive_bars": 0,
        "previous_state":   None,
        "exited_from_state": None,
    }
    # Rolling close-price cache for Bollinger Band confluence filter (live only)
    close_prices_cache: list = []

    arm = AdaptiveRiskManager(account_size, tf=tf, broker=broker)
    logger.info(
        "Live loop started — TF=%s  broker=%s  balance=$%.2f  %s",
        tf, broker, account_size, arm,
    )

    # ATR-linked trailing stop applies to all TFs unconditionally.
    # The legacy profit_target CLI param is kept for backward compat but is no
    # longer used — the ATR trail supersedes the old fixed-exit logic.
    _trail_cfg     = ATR_TRAIL_CONFIG.get(tf.upper(), ATR_TRAIL_CONFIG["H1"])
    _atr_mult_log  = _trail_cfg["trail_mult"]
    _activ_pnl_log = _trail_cfg["activation_pnl"]
    logger.info(
        "ATR Trailing Stop — activation $%.2f | "
        "multiplier %.1fx ATR [%s] | partial close (lot>%.2f).",
        _activ_pnl_log, _atr_mult_log, tf.upper(), MIN_LOT_GUARD,
    )

    # Warn if the MQL5 EA (same magic number) already has open positions.
    _startup_positions = mt5.positions_get(symbol=DEFAULT_SYMBOL) or []
    _ea_conflict = [p for p in _startup_positions if p.magic == magic]
    if _ea_conflict:
        logger.warning(
            "CONFLICT: %d open position(s) with magic=%d detected at startup. "
            "These were likely placed by the MQL5 EA (GoldRegimeX.mq5). "
            "Running both EA and Python bridge simultaneously causes signal blocking. "
            "Detach the EA from the chart before continuing.",
            len(_ea_conflict), magic,
        )

    # Pre-seed the Bollinger Band close-price cache from MT5 history so BB
    # is live from bar 1 instead of returning the neutral 0.5 fallback for
    # the first 20 hours of every session.
    try:
        _seed_rates = mt5.copy_rates_from_pos(DEFAULT_SYMBOL, tf_mt5, 1, 50)
        if _seed_rates is not None and len(_seed_rates) > 0:
            close_prices_cache = [float(r["close"]) for r in _seed_rates]
            logger.info("BB cache pre-seeded: %d historical closes loaded.", len(close_prices_cache))
    except Exception as _seed_exc:
        logger.debug("BB cache pre-seed failed (non-critical): %s", _seed_exc)

    while True:
        try:
            # ── 1. Daily reset at UTC midnight ────────────────────────────
            today = datetime.now(timezone.utc).date()
            if today != current_day:
                current_day  = today
                # Anchor the equity gate to the fixed risk-sizing balance so loss%
                # and profit-lock% are computed against the same reference that
                # check() receives (account_size + open_pnl).  Using the live MT5
                # balance here would mismatch the baseline and trigger a false lock.
                equity_gate.reset_day(account_size)
                logger.info("New UTC day %s — equity gate reset (anchor $%.2f USD).", today, account_size)
                # Send yesterday's P&L summary to Telegram at day rollover
                try:
                    send_daily_audit_report(mt5, broker=broker)
                except Exception as _audit_exc:
                    logger.warning("Daily audit report failed: %s", _audit_exc)

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
                        if p.magic == magic
                    }
                    closed = [t for t in signal_tracker["tickets"] if t not in open_set]
                    if closed:
                        _log_closed_pnl(closed, mt5, broker=broker, tf=tf)
                        for _t in closed:
                            peak_pnl_tracker.pop(_t, None)
                            atr_state_tracker.pop(_t, None)
                        signal_tracker["tickets"] = [
                            t for t in signal_tracker["tickets"] if t in open_set
                        ]
                        if not signal_tracker["tickets"]:
                            signal_tracker["tp1_hit"] = False

                # ── Daily equity protection gate ──────────────────────────
                _pnl_divisor = 100 if broker == "headway_cent" else 1
                _open_pnl = sum(
                    p.profit for p in (mt5.positions_get(symbol=DEFAULT_SYMBOL) or [])
                    if p.magic == magic
                ) / _pnl_divisor
                if equity_gate.check(account_size + _open_pnl):
                    if equity_gate.needs_loss_notification:
                        _loss_usd = -_open_pnl
                        logger.warning(
                            "Daily loss limit hit: equity=%.2f  loss=%.2f  "
                            "(%.0f%% of $%.2f) -- closing all & locking until UTC midnight.",
                            account_size + _open_pnl, _loss_usd,
                            equity_gate.loss_pct * 100, account_size,
                        )
                        for _gp in (mt5.positions_get(symbol=DEFAULT_SYMBOL) or []):
                            if _gp.magic == magic:
                                _close_position(_gp.ticket, mt5,
                                                comment="GRX_Daily_Loss_Limit",
                                                magic=magic)
                        try:
                            send_telegram_msg(
                                f"🚨 UNIVERSAL SAFETY TRIGGERED [{tf}]: "
                                f"Daily loss limit hit (${_loss_usd:.2f}). "
                                "Bot locked for recovery."
                            )
                        except Exception:
                            pass
                    elif equity_gate.needs_profit_notification:
                        logger.info(
                            "Trailing Daily Equity Lock engaged [%s]: day gain ≥%.0f%% — "
                            "profits banked, no new signals until UTC midnight.",
                            tf, equity_gate.profit_lock_pct * 100,
                        )
                        try:
                            send_telegram_msg(
                                f"🔒 <b>Equity Lock [{tf} / {broker}]</b>\n"
                                f"Day gain ≥ <b>{equity_gate.profit_lock_pct*100:.0f}%</b> — profits banked.\n"
                                f"No new signals until midnight UTC."
                            )
                        except Exception:
                            pass
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                # ── M5 Quick Scalp Exit ────────────────────────────────────
                # Close individual positions that reach the $4 scalp target.
                # If regime hasn't changed and daily cap not hit, allow re-entry.
                if tf.upper() == "M5" and signal_tracker["tickets"]:
                    _scalp_target = ATR_TRAIL_CONFIG["M5"].get("scalp_target", 4.00)
                    for _st in list(signal_tracker["tickets"]):
                        _sp = mt5.positions_get(ticket=_st)
                        if not _sp:
                            continue
                        _s_pnl = _sp[0].profit / _pnl_divisor
                        if _s_pnl >= _scalp_target:
                            _close_position(_st, mt5, comment="GRX_Fixed_Scalp_Target", magic=magic)
                            logger.info(
                                "[M5 SCALP] Position closed at target: ticket=%d  P&L=+$%.2f",
                                _st, _s_pnl,
                            )
                            signal_tracker["tickets"] = [
                                t for t in signal_tracker["tickets"] if t != _st
                            ]
                            peak_pnl_tracker.pop(_st, None)
                            atr_state_tracker.pop(_st, None)

                # ── ATR-linked Hybrid Trailing Stop ───────────────────────
                # Phase 1 (one-time per ticket): when P&L >= PROFIT_ACTIVATION_USD,
                #   move SL to breakeven+2×spread; optionally close 50% volume.
                # Phase 2 (every poll): trail SL at price ∓ (ATR_mult × cached ATR).
                if signal_tracker["tickets"]:
                    _trail_cfg_inner = ATR_TRAIL_CONFIG.get(tf.upper(), ATR_TRAIL_CONFIG["H1"])
                    _atr_mult  = _trail_cfg_inner["trail_mult"]
                    _activ_pnl = _trail_cfg_inner["activation_pnl"]
                    _atr_cache = signal_tracker.get("atr_price", 0.0)
                    _open_pos  = [
                        p for p in (mt5.positions_get(symbol=DEFAULT_SYMBOL) or [])
                        if p.magic == magic
                        and p.ticket in set(signal_tracker["tickets"])
                    ]
                    for _pos in _open_pos:
                        _cur    = _pos.profit / _pnl_divisor
                        _ticket = _pos.ticket
                        _atr_st = atr_state_tracker.setdefault(
                            _ticket,
                            {"activated": False, "partial_done": False,
                             "current_sl": _pos.sl},
                        )

                        if not _atr_st["activated"] and _cur >= _activ_pnl:
                            # Phase 1: lock in break-even + 2× spread (one time)
                            _tick   = mt5.symbol_info_tick(DEFAULT_SYMBOL)
                            _spread = _tick.ask - _tick.bid
                            _dir    = signal_tracker["direction"]
                            if _dir == "BUY":
                                _be_sl = round(signal_tracker["entry_price"] + _spread * 2, 2)
                            else:
                                _be_sl = round(signal_tracker["entry_price"] - _spread * 2, 2)
                            _move_sl_to_breakeven(_ticket, _be_sl, mt5)
                            _atr_st["activated"]  = True
                            _atr_st["current_sl"] = _be_sl
                            logger.info(
                                "ATR trail activated: ticket=%d  P&L=+$%.2f  BE_SL=%.2f",
                                _ticket, _cur, _be_sl,
                            )
                            # Partial close — skipped automatically if lot <= MIN_LOT_GUARD
                            if not _atr_st["partial_done"]:
                                if _execute_partial_close(_ticket, DEFAULT_SYMBOL, mt5, magic=magic):
                                    _atr_st["partial_done"] = True

                        elif _atr_st["activated"] and _atr_cache > 0:
                            # Phase 2: ratchet SL toward price at ATR distance
                            _tick = mt5.symbol_info_tick(DEFAULT_SYMBOL)
                            _dir  = signal_tracker["direction"]
                            if _dir == "BUY":
                                _trail_sl = round(_tick.bid - _atr_mult * _atr_cache, 2)
                            else:
                                _trail_sl = round(_tick.ask + _atr_mult * _atr_cache, 2)
                            if _set_trailing_sl(_ticket, _trail_sl,
                                                _atr_st["current_sl"], _dir, mt5):
                                _atr_st["current_sl"] = _trail_sl

                time.sleep(POLL_INTERVAL_SEC)
                continue
            last_bar_time = bar_time

            # Update rolling close cache for Bollinger Band confluence filter
            close_prices_cache.append(float(bars[0]["close"]))
            if len(close_prices_cache) > 50:
                close_prices_cache.pop(0)

            # ── 3. Refresh telemetry for margin/display (risk sizing always
            #        uses the caller-supplied account_size, not the live MT5
            #        balance, because demo accounts carry arbitrary balances).
            try:
                telemetry = get_account_telemetry()
            except Exception:
                telemetry = {}

            arm = AdaptiveRiskManager(account_size, tf=tf, broker=broker)

            # ── 4. Compute live features + probability ────────────────────
            features_df, hmm_state, atr_price, raw_atr_norm, live_df = compute_live_features(
                DEFAULT_SYMBOL, tf_mt5, model_hmm, obs_cov, trans_cov,
                feature_cols=feature_cols, mt5=mt5, tf=tf, broker=broker,
            )
            signal_tracker["atr_price"] = atr_price   # cache for between-bar ATR trail

            # ── 4b. Hourly maintenance: weekly data pull + TCN staleness ──
            _now_hour = datetime.now(timezone.utc).replace(
                minute=0, second=0, microsecond=0
            )
            if not hasattr(run_live_loop, "_last_maint_hour") or \
                    run_live_loop._last_maint_hour != _now_hour:
                run_live_loop._last_maint_hour = _now_hour
                try:
                    from src.data_updater import WeeklyDataUpdater
                    _upd = WeeklyDataUpdater()
                    if _upd.should_update():
                        logger.info("Sunday detected — running weekly data update.")
                        _upd.update_all_timeframes()
                except Exception as _upd_exc:
                    logger.debug("Weekly data updater: %s", _upd_exc)
                try:
                    from src.tcn_maintenance import TCNMaintenanceScheduler
                    _maint = TCNMaintenanceScheduler(broker=broker, balance=account_size)
                    if _maint.should_run_check():
                        _maint.run_maintenance_cycle()
                except Exception as _maint_exc:
                    logger.debug("TCN maintenance check: %s", _maint_exc)
            _, _probs = get_predictions_ensemble(models_xgb, thresholds_xgb, features_df)
            prob        = float(_probs[0])
            gmm_cluster = int(features_df["gmm_vol_cluster"].iloc[0]) if "gmm_vol_cluster" in features_df.columns else -1

            # ── Spread efficiency filter ──────────────────────────────────────
            # Use the RAW (pre-scaler) atr_normalized fraction — the StandardScaler
            # can produce negative values which make the ratio meaningless.
            _spread_frac = BROKER_CONFIGS.get(broker, {}).get("spread_frac", 0.0004)
            _er          = raw_atr_norm / _spread_frac if _spread_frac > 0 else 999.0
            if _er < 1.25:
                logger.info(
                    "Efficiency ratio %.2f < 1.25 (ATR=%.5f / spread=%.5f) — no signal.",
                    _er, raw_atr_norm, _spread_frac,
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 5. Log bar info on every new bar (always visible) ─────────
            bar_str   = datetime.fromtimestamp(bar_time, timezone.utc).strftime("%Y-%m-%d %H:%M")
            _snames   = {2: STATE_NAMES_2, 3: STATE_NAMES_3, 4: STATE_NAMES_4}
            state_lbl = _snames.get(model_hmm.n_components, STATE_NAMES_4).get(hmm_state, str(hmm_state))
            logger.info(
                "Bar %s | state=%s | prob=%.3f | Efficiency=%.1fx",
                bar_str, hmm_state, prob, _er,
            )
            send_telegram_msg(
                f"📊 <b>Bar</b> {bar_str} [{tf}]\n"
                f"Regime: <b>{state_lbl}</b> ({hmm_state})  |  "
                f"Prob: <b>{prob:.3f}</b>  |  "
                f"Efficiency: <b>{_er:.1f}x</b>"
            )

            # ── 6. Position management (always runs — P&L, break-even, chop-exit)
            if signal_tracker["tickets"]:
                open_set = {
                    p.ticket
                    for p in (mt5.positions_get(symbol=DEFAULT_SYMBOL) or [])
                    if p.magic == magic
                }
                active = [t for t in signal_tracker["tickets"] if t in open_set]
                closed = [t for t in signal_tracker["tickets"] if t not in open_set]
                # Log P&L for any positions that closed since last bar
                if closed:
                    _log_closed_pnl(closed, mt5, broker=broker, tf=tf)
                # Profit guard: move SL to entry+spread when 70% to TP1 (all TFs)
                if active:
                    _apply_profit_guard(signal_tracker, mt5)
                # TP1 hit: first position gone → move runner's SL to break-even
                if len(active) < len(signal_tracker["tickets"]) and not signal_tracker["tp1_hit"]:
                    signal_tracker["tp1_hit"] = True
                    for ticket in active:
                        _move_sl_to_breakeven(ticket, signal_tracker["entry_price"], mt5)
                # Regime shifted to any Chop state while a TREND runner is active → close immediately.
                # Use >= CHOP_STATE to cover both Chop_Low (2) and Chop_High (3) in 4-state models.
                if signal_tracker.get("signal_type") == "trend" and hmm_state >= CHOP_STATE and active:
                    for ticket in active:
                        _close_position(ticket, mt5, magic=magic)
                    logger.info("Trend runner(s) closed: regime shifted to Chop.")
                    active = []
                elif signal_tracker.get("signal_type") == "mean_reversion" and hmm_state < CHOP_STATE and active:
                    for ticket in active:
                        _close_position(ticket, mt5, comment="GRX_close_mr_breakout", magic=magic)
                    logger.info("MR position(s) closed: Chop ended, regime broke out to state %d.", hmm_state)
                    active = []
                signal_tracker["tickets"] = active

            # ── 7. Skip new signals if equity gate is locked ─────────────
            if equity_gate.locked:
                logger.info(
                    "Equity gate locked — no new signal (state=%s).",
                    state_lbl,
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 8. Skip new signals if a position is still open ──────────
            if has_open_position(DEFAULT_SYMBOL, magic):
                logger.info(
                    "Open position -- holding (state=%s).",
                    state_lbl,
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 8. Global multi-TF exposure guard ────────────────────────
            # Count open positions across ALL three GRX magic numbers so that
            # H1 + M15 + M5 running simultaneously on the same $15 account
            # can't exceed 4 open positions total.
            _all_grx_open = sum(
                1 for p in (mt5.positions_get(symbol=DEFAULT_SYMBOL) or [])
                if p.magic in ALL_GRX_MAGICS
            )
            if _all_grx_open >= 4:
                logger.info(
                    "[GLOBAL GUARD] Max account exposure reached (%d positions across all TFs). "
                    "Skipping %s signal.", _all_grx_open, tf,
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 9. Signal routing — RCEV or Z-Score evaluation ───────────
            # RCEV path: when calibration file is present and per-regime
            #   classifiers are loaded, use evaluate_signal_rcev().
            # Z-Score fallback: use evaluate_signal() (existing logic).
            _stability  = _update_regime_stability(regime_stability_tracker, hmm_state)
            _t_prob     = _get_transition_prob(model_hmm, hmm_state)
            _bb_pos     = _calculate_bb_position(np.array(close_prices_cache))
            _gmc        = gmm_cluster if gmm_cluster >= 0 else 1

            # Regime stability from HMM transmat diagonal P(stay in state)
            _regime_stability = float(model_hmm.transmat_[hmm_state, hmm_state])
            if not (0.0 < _regime_stability <= 1.0):
                _regime_stability = 0.70

            # Per-regime XGBoost probabilities (uniform fallback if not loaded)
            _regime_probs = predict_regime_proba(regime_models_live, features_df)

            # TCN multiplier
            _tcn_mult_val = 1.0
            if tcn_classifier is not None:
                try:
                    _tcn_r = tcn_classifier.predict_confidence(live_df)
                    if _tcn_r is not None:
                        _tcn_mult_val = float(_tcn_r)
                except Exception:
                    pass

            if rcev_scorer_live is not None:
                _sig_eval = signal_evaluator.evaluate_signal_rcev(
                    rcev             = rcev_scorer_live,
                    regime_probs     = _regime_probs,
                    hmm_state        = hmm_state,
                    volatility       = float(raw_atr_norm),
                    spread           = BROKER_CONFIGS.get(broker, {}).get("spread_frac", 0.0004),
                    atr              = float(atr_price),
                    hour             = datetime.now(timezone.utc).hour,
                    bb_position      = _bb_pos,
                    tcn_multiplier   = _tcn_mult_val,
                    rcev_threshold   = _rcev_threshold,
                    tiered           = use_tiered,
                    regime_stability = _regime_stability,
                    tf               = tf,
                )
                _sig_str = _sig_eval.get("signal", "WAIT")
            else:
                # Z-Score fallback path (existing logic)
                _active_evaluator = signal_evaluator
                if tcn_classifier is not None:
                    try:
                        _tcn_mult2 = tcn_classifier.predict_confidence(live_df)
                        if _tcn_mult2 is not None:
                            _base_cut = signal_evaluator.config["Z_CUTOFF_BULL"]
                            _eff_cut  = _base_cut * _tcn_mult2
                            from src.signal_evaluator import SignalEvaluator as _SE
                            _active_evaluator = _SE(
                                regime_stats, tf=tf,
                                config={"Z_CUTOFF_BULL": _eff_cut, "Z_CUTOFF_BEAR": -_eff_cut},
                            )
                    except Exception:
                        pass
                _sig_eval = _active_evaluator.evaluate_signal(
                    prob_buy=prob,
                    hmm_state=hmm_state,
                    gmm_cluster=_gmc,
                    stability=_stability,
                    bb_position=_bb_pos,
                    transition_prob=_t_prob,
                    use_tiered=use_tiered,
                )
                _sig_str = _sig_eval.get("signal")

            if _sig_str == "BUY":
                direction   = "BUY"
                order_type  = mt5.ORDER_TYPE_BUY
                signal_type = "trend"
                _conf_str = (f"RCEV=${_sig_eval.get('expected_pnl', 0):.2f}"
                             if rcev_scorer_live else f"z={_sig_eval.get('confidence', 0):.2f}")
                logger.info("[SIGNAL] BUY (trend) | %s | %s",
                            _conf_str, _sig_eval.get("reason", ""))
            elif _sig_str == "SELL":
                direction   = "SELL"
                order_type  = mt5.ORDER_TYPE_SELL
                signal_type = "trend"
                _conf_str = (f"RCEV=${_sig_eval.get('expected_pnl', 0):.2f}"
                             if rcev_scorer_live else f"z={_sig_eval.get('confidence', 0):.2f}")
                logger.info("[SIGNAL] SELL (trend) | %s | %s",
                            _conf_str, _sig_eval.get("reason", ""))
            elif _sig_str == "MR_BUY":
                direction   = "BUY"
                order_type  = mt5.ORDER_TYPE_BUY
                signal_type = "mean_reversion"
                _conf_str = (f"RCEV=${_sig_eval.get('expected_pnl', 0):.2f}"
                             if rcev_scorer_live else f"z={_sig_eval.get('confidence', 0):.2f}")
                logger.info("[SIGNAL] MR_BUY | %s | %s",
                            _conf_str, _sig_eval.get("reason", ""))
                if gmm_cluster == 2:
                    logger.warning(
                        "[MR WARNING] High-vol environment (GMM cluster=2) — "
                        "MR has lower edge in breakout conditions."
                    )
            elif _sig_str == "MR_SELL":
                direction   = "SELL"
                order_type  = mt5.ORDER_TYPE_SELL
                signal_type = "mean_reversion"
                _conf_str = (f"RCEV=${_sig_eval.get('expected_pnl', 0):.2f}"
                             if rcev_scorer_live else f"z={_sig_eval.get('confidence', 0):.2f}")
                logger.info("[SIGNAL] MR_SELL | %s | %s",
                            _conf_str, _sig_eval.get("reason", ""))
                if gmm_cluster == 2:
                    logger.warning(
                        "[MR WARNING] High-vol environment (GMM cluster=2) — "
                        "MR has lower edge in breakout conditions."
                    )
            else:
                # ── Logic Audit ───────────────────────────────────────────
                regime_desc = ("BULL" if hmm_state == BULL_STATE
                               else "BEAR" if hmm_state == BEAR_STATE
                               else "CHOP")
                if rcev_scorer_live:
                    logger.info(
                        "[LOGIC AUDIT] %s Regime | state=%d  RCEV=$%.2f<$%.2f  "
                        "bars=%d  P(stay)=%.2f  BB=%.2f  regime_probs=%s",
                        regime_desc, hmm_state,
                        _sig_eval.get("expected_pnl", 0), _sig_eval.get("threshold", _rcev_threshold),
                        _stability["consecutive_bars"], _regime_stability, _bb_pos,
                        {k: f"{v:.2f}" for k, v in _regime_probs.items()},
                    )
                else:
                    _z = _sig_eval.get("confidence", 0)
                    logger.info(
                        "[LOGIC AUDIT] %s Regime | state=%d  z=%.2f  "
                        "bars=%d  P(stay)=%.2f  BB=%.2f | %s",
                        regime_desc, hmm_state, _z,
                        _stability["consecutive_bars"], _t_prob, _bb_pos,
                        _sig_eval.get("reason", ""),
                    )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 9. Position sizing ────────────────────────────────────────
            tp_mults      = _tp_multipliers(tf, hmm_state)
            limits        = arm.get_trade_limits(hmm_state)
            pos_per_trade = min(limits["pos_per_trade"], len(tp_mults))
            sl_distance   = max(atr_price * TF_ATR_MULTIPLIER.get(tf.upper(), 2.0), 0.01)
            if signal_type == "mean_reversion":
                sl_distance *= 0.70   # MR trades use tighter SL — mean-reversion has defined exit
                logger.info("[MR RISK ADJ] SL tightened to 70%% of base: %.4f", sl_distance)

            # $15 accounts use hardcoded lot splits — bypass AdaptiveRiskManager.
            # Cent:     0.02 (pos1) + 0.03 (pos2) = 0.05 micro-lots total.
            # Standard: 0.01 single position only — margin safety on $15 standard.
            # Accounts > $50: dynamic ARM sizing as normal.
            if round(account_size) == 15:
                if broker == "headway_cent":
                    forced_lots = [0.02, 0.03]
                else:
                    forced_lots   = [0.01]
                    pos_per_trade = 1
                pos_per_trade = min(pos_per_trade, len(forced_lots))
                logger.info(
                    "[SPECIAL] $15 Account Detected: Enforcing forced lot sizing "
                    "(Broker: %s)  lots=%s  pos_per_trade=%d",
                    broker, forced_lots, pos_per_trade,
                )
            else:
                lot_total   = arm.calculate_lot_size(stop_loss_pips=sl_distance)
                lot_per_pos = max(0.01, round(lot_total / pos_per_trade, 2))
                forced_lots = [lot_per_pos] * pos_per_trade

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
            if not check_margin(DEFAULT_SYMBOL, max(forced_lots), order_type, entry_price):
                logger.warning(
                    "Insufficient margin for %s lot=%.2f. Skipping. Free margin: %.2f",
                    direction, max(forced_lots), telemetry.get("free_margin", 0),
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 13. Send order(s) with state-aware staged TPs ─────────────
            signal_tracker = {"tickets": [], "entry_price": entry_price,
                              "direction": direction, "tp1_hit": False,
                              "tp1_level": tp_levels[0], "guard_hit": False,
                              "signal_type": signal_type}
            peak_pnl_tracker = {}
            logger.info(
                "SIGNAL %s | state=%d | z=%.2f | lots=%s | sl=%.2f | tp=%s | dev=%d",
                direction, hmm_state, _sig_eval["confidence"],
                "+".join(f"{l:.2f}" for l in forced_lots[:pos_per_trade]),
                sl_price, "/".join(f"{t:.2f}" for t in tp_levels), deviation,
            )
            for p in range(pos_per_trade):
                tp_price    = tp_levels[p]
                current_lot = forced_lots[p] if p < len(forced_lots) else forced_lots[-1]
                _trade_tag  = "MR" if signal_type == "mean_reversion" else "TREND"
                comment     = f"GRX_{tf}_{_trade_tag}_{direction}_s{hmm_state}_tp{p+1}"
                result = send_market_order(
                    symbol=DEFAULT_SYMBOL,
                    order_type=order_type,
                    lot=current_lot,
                    sl=sl_price,
                    tp=tp_price,
                    deviation=deviation,
                    magic=magic,
                    comment=comment,
                )
                if result["success"]:
                    signal_tracker["tickets"].append(result["order"])
                else:
                    logger.error(
                        "Order %d/%d failed — retcode=%d  %s",
                        p + 1, pos_per_trade,
                        result["retcode"], result["comment"],
                    )

            # Send one Telegram message per signal summarising all filled positions
            if signal_tracker["tickets"]:
                _emoji  = "🟢" if direction == "BUY" else "🔴"
                _tag    = "📐 MEAN REVERSION" if signal_type == "mean_reversion" else "📈 TREND"
                _tp_str = " / ".join(f"{t:.2f}" for t in tp_levels)
                _tix    = "  ".join(f"#{t}" for t in signal_tracker["tickets"])
                send_telegram_msg(
                    f"{_emoji} <b>Trade opened</b> [{tf}]  {_tag}\n"
                    f"<b>{direction}</b>  Regime: <b>{state_lbl}</b>  Prob: <b>{prob:.3f}</b>\n"
                    f"Lots: <b>{'+'.join(f'{l:.2f}' for l in forced_lots[:pos_per_trade])}</b>  "
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
