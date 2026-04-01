"""MT5 Live Execution Engine.

Provides account telemetry, feature engineering parity with the training
pipeline, margin validation, and live order placement through the
MetaTrader5 Python package.

Usage (via main.py):
    python main.py --mode live --account demo --broker headway_cent --balance 15 --tf H1

IMPORTANT: Remove the GoldRegimeX.mq5 EA from the XAUUSD chart before running
this script.  Both the EA and the Python bridge use MAGIC_NUMBER = 123456 and
their session counters are independent — running both simultaneously will
double-count trades.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.notifier import send_telegram_msg

from src.logger import setup_logger
from src.processor import (
    TF_CONFIG,
    DXY_RAW_PATH,
    load_dxy_data,
    compute_log_returns,
    kalman_smooth,
    compute_volatility,
    compute_rsi,
    compute_atr,
)
from src.engine_hmm import load_model as load_hmm, predict_states
from src.engine_xgb import (
    load_xgb_ensemble, get_predictions_ensemble, assign_vol_bucket, FEATURE_COLS,
)
from src.mt5_sync import DXY_SYMBOL_ALIASES, _find_dxy_symbol
from src.risk_manager import AdaptiveRiskManager, CENT_MULTIPLIER

logger = setup_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOL               = "XAUUSD"
MAGIC_NUMBER                 = 123456   # must match GoldRegimeX.mq5 MagicNumber
CHOP_STATE                   = 2        # HMM Chop state index
ATR_MULTIPLIER               = 2.0      # SL = entry ± ATR × 2.0  (all TFs)
DEFAULT_DEVIATION            = 20       # fallback deviation for check_margin / send_market_order
N_BARS_WARMUP                = 200      # bars fetched for Kalman/HMM warm-up
POLL_INTERVAL_SEC            = 5        # seconds between bar-change checks
HIGH_VOL_SELF_TRANS_THRESHOLD = 0.70    # self-transition prob below this -> elevated deviation
MIN_SPREAD_RATIO             = 1.5      # TP1 must be at least 1.5× spread to be viable

# ── Per-timeframe signal thresholds and order parameters ──────────────────────
TF_PROB_THRESHOLD  = {"M5": 0.70, "M15": 0.65, "H1": 0.65}
TF_SHORT_THRESHOLD = {"M5": 0.30, "M15": 0.35, "H1": 0.35}
TF_DEFAULT_DEV     = {"M5": 30,   "M15": 20,   "H1": 20}
TF_HIGH_VOL_DEV    = {"M5": 50,   "M15": 50,   "H1": 50}

# ── State-aware multi-stage TP multipliers (relative to SL distance) ──────────
# Bull / Bear: TP1 quick partial, TP2 runner.
# Chop: tighter single TP — position 2 (runner) is skipped.
# Format: {regime: [tp1_mult, tp2_mult]}; single-element list = one TP, close all.
TF_TP_CONFIG = {
    "M5":  {"trending": [1.0, 3.0], "chop": [0.8]},
    "M15": {"trending": [1.5, 3.0], "chop": [1.5]},
    "H1":  {"trending": [1.5, 3.0], "chop": [1.5]},
}

# Lazy MT5 timeframe map

# ── DXY fallback cache ────────────────────────────────────────────────────────
# When the broker doesn't carry DXY as a live symbol, we carry forward the
# last known daily log return from DXY_data.csv (much closer to reality than 0.0).
_DXY_FALLBACK_CACHE: float | None = None

def _get_dxy_fallback() -> float:
    """Return the last known DXY log return from DXY_data.csv (cached after first load)."""
    global _DXY_FALLBACK_CACHE
    if _DXY_FALLBACK_CACHE is None:
        try:
            dxy_df = load_dxy_data(DXY_RAW_PATH)
            if dxy_df is not None:
                _DXY_FALLBACK_CACHE = float(dxy_df["dxy_log_return"].dropna().iloc[-1])
            else:
                _DXY_FALLBACK_CACHE = 0.0
        except Exception:
            _DXY_FALLBACK_CACHE = 0.0
    return _DXY_FALLBACK_CACHE
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

def _fetch_dxy_log_return(tf_mt5: int, mt5) -> float | None:
    """Fetch the most recently completed DXY bar log return for live inference.

    Returns ``None`` if no DXY symbol is found on the broker or data is
    unavailable — callers should substitute ``0.0`` in that case.
    """
    dxy_sym = _find_dxy_symbol(mt5)
    if dxy_sym is None:
        return None
    rates = mt5.copy_rates_from_pos(dxy_sym, tf_mt5, 1, 3)  # 3 completed bars
    if rates is None or len(rates) < 2:
        return None
    closes = [r["close"] for r in rates]
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


def _close_position(ticket: int, mt5) -> None:
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
        "comment":      "GRX_close_chop",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("Position closed (Chop exit): ticket=%d", ticket)
    else:
        logger.warning("Position close failed: ticket=%d  retcode=%s",
                       ticket, res.retcode if res else "None")


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
    always matches what the models were trained on.  If ``dxy_log_return`` is
    required and unavailable from the broker, it is filled with 0.0.
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

    if "dxy_log_return" in feature_cols:
        if mt5 is not None:
            dxy_ret = _fetch_dxy_log_return(tf_mt5, mt5)
        else:
            dxy_ret = None
        if dxy_ret is None:
            dxy_ret = _get_dxy_fallback()
            logger.warning(
                "DXY return unavailable — using last known value %.6f as fallback for dxy_log_return.",
                dxy_ret,
            )
        feature_dict["dxy_log_return"] = dxy_ret

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
    dry_run: bool    = False,
) -> None:
    """Connect to MT5 and run the signal → order loop until interrupted.

    Fires on each newly completed bar.  Position sizing, session limits, and
    margin validation are enforced before every order.

    Args:
        tf:           Timeframe to trade — "H1" or "M15".
        broker:       Broker config key from risk_manager.BROKER_CONFIGS.
        account_size: USD balance used for lot-sizing.  If None, reads from MT5
                      and normalises for cent accounts automatically.
        dry_run:      When True, signals are logged but no orders are placed.
                      Use with --account demo for paper-trading simulation.
    """
    import MetaTrader5 as mt5
    from src.mt5_sync import connect_mt5, disconnect_mt5

    if not connect_mt5():
        raise ConnectionError("Could not connect to MT5 terminal.")

    try:
        _run_loop_inner(tf, broker, account_size, dry_run, mt5)
    finally:
        disconnect_mt5()
        logger.info("Live loop terminated.  MT5 disconnected.")


def _run_loop_inner(tf: str, broker: str, account_size: float, dry_run: bool, mt5) -> None:
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

    # Load models
    try:
        model_hmm = load_hmm()
    except FileNotFoundError:
        raise FileNotFoundError("HMM model not found. Run --mode train first.")
    try:
        models_xgb, thresholds_xgb, xgb_meta = load_xgb_ensemble()
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

    # Use optimized prob threshold if available; otherwise fall back to TF hardcoded values.
    # Short threshold is symmetric: 1 - prob_threshold.
    if prob_threshold_opt is not None:
        prob_threshold  = prob_threshold_opt
        short_threshold = 1.0 - prob_threshold_opt
    else:
        prob_threshold  = TF_PROB_THRESHOLD.get(tf.upper(), 0.65)
        short_threshold = TF_SHORT_THRESHOLD.get(tf.upper(), 0.35)

    # Session state (persists across bars within a day)
    daily_trades  = 0
    last_bar_time = None
    current_day   = None
    # Tracks the tickets and entry context of the most recently placed signal
    # so the break-even SL logic can fire when TP1 closes position 1.
    signal_tracker = {"tickets": [], "entry_price": 0.0,
                      "direction": None, "tp1_hit": False}

    arm = AdaptiveRiskManager(account_size)
    logger.info(
        "Live loop started — TF=%s  broker=%s  balance=$%.2f  dry_run=%s  %s",
        tf, broker, account_size, dry_run, arm,
    )
    logger.info(
        "Signal thresholds — BUY>%.3f  SELL<%.3f  (source: %s)",
        prob_threshold, short_threshold,
        "optuna" if prob_threshold_opt is not None else "hardcoded",
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

            arm              = AdaptiveRiskManager(account_size)
            arm.daily_trades = daily_trades   # restore session count

            # ── 4. Compute live features ──────────────────────────────────
            features_df, hmm_state, atr_price = compute_live_features(
                DEFAULT_SYMBOL, tf_mt5, model_hmm, obs_cov, trans_cov,
                feature_cols=feature_cols, mt5=mt5,
            )

            # ── 5. Session limit check ────────────────────────────────────
            if not arm.can_trade(hmm_state):
                logger.info(
                    "Session limit reached (state=%d  daily=%d/%d). Skipping bar.",
                    hmm_state, arm.daily_trades,
                    arm.get_trade_limits(hmm_state)["max_daily_trades"],
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 6. Skip if position already open ─────────────────────────
            if has_open_position(DEFAULT_SYMBOL, MAGIC_NUMBER):
                logger.debug("Open position exists — skipping bar.")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            bar_str   = datetime.fromtimestamp(bar_time, timezone.utc).strftime("%Y-%m-%d %H:%M")
            state_lbl = {0: "Bull", 1: "Bear", 2: "Chop"}.get(hmm_state, str(hmm_state))
            max_trades_today = arm.get_trade_limits(hmm_state)["max_daily_trades"]
            _, _probs = get_predictions_ensemble(models_xgb, thresholds_xgb, features_df)
            prob = float(_probs[0])
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
                + ("  🔴 <i>DRY RUN</i>" if dry_run else "")
            )

            # ── 7b. Break-even SL and Chop-exit position management ───────
            if not dry_run and signal_tracker["tickets"]:
                open_tickets = {
                    p.ticket
                    for p in (mt5.positions_get(symbol=DEFAULT_SYMBOL) or [])
                    if p.magic == MAGIC_NUMBER
                }
                active = [t for t in signal_tracker["tickets"] if t in open_tickets]
                # TP1 hit: first position closed → move runner's SL to break-even
                if len(active) < len(signal_tracker["tickets"]) and not signal_tracker["tp1_hit"]:
                    signal_tracker["tp1_hit"] = True
                    for ticket in active:
                        _move_sl_to_breakeven(
                            ticket, signal_tracker["entry_price"], mt5,
                        )
                # Regime shifted to Chop while runner is active → close immediately
                if hmm_state == CHOP_STATE and active:
                    for ticket in active:
                        _close_position(ticket, mt5)
                    logger.info("Runner(s) closed: Chop state shift while position open.")
                    active = []
                signal_tracker["tickets"] = active

            # ── 8. Signal routing (TF-specific thresholds) ───────────────
            if prob > prob_threshold and hmm_state != CHOP_STATE:
                direction  = "BUY"
                order_type = mt5.ORDER_TYPE_BUY
            elif prob < short_threshold and hmm_state != CHOP_STATE:
                direction  = "SELL"
                order_type = mt5.ORDER_TYPE_SELL
            else:
                logger.debug("No signal (prob=%.3f  state=%d). Holding.", prob, hmm_state)
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ── 9. Position sizing ────────────────────────────────────────
            tp_mults      = _tp_multipliers(tf, hmm_state)
            limits        = arm.get_trade_limits(hmm_state)
            pos_per_trade = min(limits["pos_per_trade"], len(tp_mults))
            sl_distance   = max(atr_price * ATR_MULTIPLIER, 0.01)
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

            # ── 11b. Spread viability guard (M5 scalping only) ───────────
            if tf.upper() == "M5":
                sym_info     = mt5.symbol_info(DEFAULT_SYMBOL)
                spread_price = (sym_info.spread * sym_info.point) if sym_info else 0.0
                tp1_distance = sl_distance * tp_mults[0]
                if spread_price > 0 and tp1_distance < spread_price * MIN_SPREAD_RATIO:
                    logger.warning(
                        "Unviable M5 trade: TP1=%.4f < %.1fx spread=%.4f. Skipping.",
                        tp1_distance, MIN_SPREAD_RATIO, spread_price,
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
                              "direction": direction, "tp1_hit": False}
            logger.info(
                "SIGNAL %s | state=%d | prob=%.3f | lot×%d=%.2f | sl=%.2f | tp=%s | dev=%d",
                direction, hmm_state, prob, pos_per_trade, lot_per_pos,
                sl_price, "/".join(f"{t:.2f}" for t in tp_levels), deviation,
            )
            for p in range(pos_per_trade):
                tp_price = tp_levels[p]
                comment  = f"GRX_{direction}_p{p+1}of{pos_per_trade}_s{hmm_state}_tp{p+1}"
                if dry_run:
                    logger.info(
                        "[DRY RUN] Would send: %s  lot=%.2f  sl=%.2f  tp=%.2f  dev=%d  comment=%s",
                        direction, lot_per_pos, sl_price, tp_price, deviation, comment,
                    )
                else:
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

            if dry_run:
                daily_trades += 1   # count dry-run signals for session limit testing

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping live loop.")
            break
        except Exception as exc:
            logger.error(
                "Unhandled loop error: %s — sleeping 30 s before retry.",
                exc, exc_info=True,
            )
            time.sleep(30)
