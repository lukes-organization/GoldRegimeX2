from src.logger import setup_logger

logger = setup_logger(__name__)

# Spread and commission fractions of price per trade (normalized cost)
BROKER_CONFIGS = {
    "standard": {
        "spread_frac": 0.0002,
        "commission_frac": 0.0001,
    },
    "headway_cent": {
        # XAUUSD on Headway Cent: ~$0.30 spread + ~$0.03/micro-lot commission
        "spread_frac": 0.0002,
        "commission_frac": 0.0002,
    },
}

MIN_CAPITAL_USD = 15.0
CENT_MULTIPLIER = 100       # Headway Cent: 1 USD displayed as 100 cents
SMALL_ACCOUNT_THRESHOLD = 50.0  # USD — boundary between small and growth tiers


class CentConverter:
    """Lot sizing for Headway Cent accounts (micro-lot floor = 0.01)."""

    MIN_LOT = 0.01

    def __init__(self, account_balance_usd: float):
        self.account_balance_usd = max(float(account_balance_usd), MIN_CAPITAL_USD)

    @property
    def displayed_balance(self) -> float:
        return self.account_balance_usd * CENT_MULTIPLIER

    def calculate_lot(self, risk_pct: float, stop_distance_norm: float) -> float:
        if stop_distance_norm <= 0:
            return self.MIN_LOT
        risk_amount = self.account_balance_usd * risk_pct
        lot = risk_amount / stop_distance_norm
        lot = max(self.MIN_LOT, (lot // self.MIN_LOT) * self.MIN_LOT)
        return round(lot, 2)

    def __repr__(self) -> str:
        return (
            f"CentConverter(balance=${self.account_balance_usd:.2f}, "
            f"displayed={self.displayed_balance:.0f} cents)"
        )


class SessionManager:
    """Legacy per-session limiter (kept for backwards compat).

    Prefer ``AdaptiveRiskManager`` for new code.
    """

    def __init__(self, account_size: float):
        self.account_size = float(account_size)
        # Delegate to AdaptiveRiskManager for consistent limits
        arm = AdaptiveRiskManager(account_size)
        base = arm.get_trade_limits()
        self.max_trades = base["max_daily_trades"]
        self.trades_this_session = 0

    def can_trade(self) -> bool:
        return self.trades_this_session < self.max_trades

    def log_trade(self):
        self.trades_this_session += 1

    def reset_session(self):
        self.trades_this_session = 0

    def __repr__(self) -> str:
        return (
            f"SessionManager(account=${self.account_size:.0f}, "
            f"max={self.max_trades}, used={self.trades_this_session})"
        )


class AdaptiveRiskManager:
    """Dynamic trade limits and lot sizing for Headway Cent and Standard accounts.

    Account tiers
    ─────────────
    ≤ $50 USD  — small account:
        • max_daily_trades : 2  (4 for M5/M15; 2 for H1 on headway_cent)
        • pos_per_trade    : 1  (single position per signal; 2 for M5/M15)
        • total daily pos  : same as max_daily_trades

    > $50 USD  — growth account:
        • max_daily_trades : 3 in Bull/Bear, 2 in Chop  (market-dependent)
        • pos_per_trade    : 2  (dual positions per signal)
        • total daily pos  : 4 or 6

    Standard account guard
    ──────────────────────
    When ``broker == "standard"`` and balance < $50, ``pos_per_trade`` is
    forced to 1 regardless of tier.  A 0.01 standard lot on XAUUSD has a
    notional of ~$48 and a margin requirement that makes dual positions
    unsafe on a $15 account.
    """

    CHOP_STATE = 2

    def __init__(self, balance: float, tf: str = "H1", broker: str = "headway_cent"):
        self.balance = max(float(balance), MIN_CAPITAL_USD)
        self.tf      = tf.upper()
        self.broker  = broker
        self.daily_trades = 0
        tier = "small" if self.balance <= SMALL_ACCOUNT_THRESHOLD else "growth"
        logger.debug(
            "AdaptiveRiskManager: balance=$%.2f tier=%s tf=%s broker=%s",
            self.balance, tier, self.tf, self.broker,
        )

    @property
    def is_small_account(self) -> bool:
        return self.balance <= SMALL_ACCOUNT_THRESHOLD

    def get_trade_limits(self, market_state: int = None, tf: str = None) -> dict:
        """Return trade limits for the current balance and optional HMM state.

        Args:
            market_state: Current HMM state index. If None, uses the most
                          permissive limit for the account tier.
            tf: Timeframe string (e.g. ``"M5"``). Defaults to ``self.tf``
                set at construction. M5 small accounts get a higher daily
                cap (4) to match the higher bar frequency.

        Lot sizing note: pos_per_trade controls how many orders are placed per
        signal.  Each order is individually floored to 0.01 lots in the trader
        (``max(0.01, lot_per_pos)``), so even on a $15 account every position
        uses minimum notional sizing — 2 positions simply gives two independent
        entries with separate TPs for performance monitoring.
        """
        _tf = (tf or self.tf).upper()
        if self.is_small_account:
            # H1 on a small cent account: cap at 2 trades per day.
            # Allows recovery if the first setup is a minor stop-out while still
            # preventing over-trading the slow hourly timeframe on a $15 account.
            if _tf == "H1" and self.broker == "headway_cent":
                return {"max_daily_trades": 2, "pos_per_trade": 1, "total_daily_pos": 2}
            # M5/M15 small accounts: 4 daily position slots (2 signals × 2 positions).
            # Both standard and cent use pos_per_trade=2; lot floor (0.01) in mt5_trader
            # ensures notional stays minimal per position.
            max_daily = 4 if _tf in ("M5", "M15") else 2
            return {"max_daily_trades": max_daily, "pos_per_trade": 2, "total_daily_pos": max_daily}

        # Growth account (>$50): 3 positions per signal for staged TPs.
        in_chop = (market_state == self.CHOP_STATE) if market_state is not None else False
        max_daily = 2 if in_chop else 3
        return {
            "max_daily_trades": max_daily,
            "pos_per_trade": 3,
            "total_daily_pos": max_daily * 3,
        }

    def calculate_lot_size(self, stop_loss_pips: float) -> float:
        """1% risk rule for Headway cent XAUUSD.

        On the Headway cent account the effective contract is 1 oz per lot, so
        each $1 price move = $1 P&L per lot.  Formula:

            lot = (balance × 1%) / sl_price_distance

        Example: $15 balance, 14.27-pt SL → $0.15 / 14.27 = 0.0105 → 0.01 lot.
        This mirrors CentConverter.calculate_lot() which is already correct.
        """
        risk_per_trade = self.balance * 0.01
        if stop_loss_pips <= 0:
            return 0.01
        lot_size = risk_per_trade / stop_loss_pips
        return max(0.01, round(lot_size, 2))

    def can_trade(self, market_state: int = None) -> bool:
        limits = self.get_trade_limits(market_state, tf=self.tf)
        return self.daily_trades < limits["max_daily_trades"]

    def log_trade(self):
        self.daily_trades += 1

    def reset_daily(self):
        self.daily_trades = 0

    def __repr__(self) -> str:
        tier = "small" if self.is_small_account else "growth"
        return (
            f"AdaptiveRiskManager(balance=${self.balance:.0f}, tier={tier}, "
            f"broker={self.broker}, daily_trades={self.daily_trades})"
        )


class DailyEquityGate:
    """Floating-equity safety switch for live trading.

    Monitors the sum of realised balance and open floating P&L.  If that
    running equity drops >= ``loss_pct`` below the start-of-day baseline, the
    gate locks and remains locked until ``reset_day`` is called (UTC midnight).

    Usage in the live loop::

        gate = DailyEquityGate()
        gate.reset_day(account_size)   # call once at start of day

        # each poll cycle:
        current_eq = account_size + open_pnl
        if gate.check(current_eq):
            if gate.needs_notification:
                <close all positions + send Telegram alert>
            continue   # skip new signals until tomorrow
    """

    DAILY_LOSS_PCT = 0.05   # 5 % default loss limit

    def __init__(self, loss_pct: float = DAILY_LOSS_PCT):
        self.loss_pct      = loss_pct
        self._start_equity: float | None = None
        self._locked       = False
        self._notified     = False

    def reset_day(self, equity: float) -> None:
        """Initialise or reset for a new UTC trading day."""
        self._start_equity = float(equity)
        self._locked       = False
        self._notified     = False
        logger.debug("DailyEquityGate reset: start_equity=%.2f  limit=%.0f%%",
                     equity, self.loss_pct * 100)

    def check(self, current_equity: float) -> bool:
        """Return True if the daily loss limit has been breached.

        Calling this repeatedly after lock is safe — it stays True and does
        NOT re-trigger ``needs_notification``.
        """
        if self._start_equity is None or self._start_equity <= 0:
            return False
        dd_pct = (self._start_equity - current_equity) / self._start_equity
        if dd_pct >= self.loss_pct:
            self._locked = True
        return self._locked

    @property
    def needs_notification(self) -> bool:
        """True exactly once — the first poll cycle after the lock engages.

        Designed so the caller sends a single Telegram alert without repeated
        messages on subsequent polls.
        """
        if self._locked and not self._notified:
            self._notified = True
            return True
        return False

    @property
    def locked(self) -> bool:
        return self._locked

    def __repr__(self) -> str:
        return (
            f"DailyEquityGate(loss_pct={self.loss_pct*100:.0f}%, "
            f"locked={self._locked}, start=${self._start_equity or 0:.2f})"
        )
