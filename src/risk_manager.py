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


class AdaptiveRiskManager:
    """Dynamic lot sizing for Headway Cent and Standard accounts.

    Returns ``pos_per_trade`` — the number of positions to open per signal —
    based on account balance tier.  Trade-frequency limits are removed; the
    Daily Equity Gate (loss side) and Trailing Daily Equity Lock (profit side)
    inside ``DailyEquityGate`` now handle all stop-trading decisions.

    Account tiers
    ─────────────
    ≤ $50 USD  — small account:
        • pos_per_trade : 2  (standard accounts get 1 — margin safety on $15)

    > $50 USD  — growth account:
        • pos_per_trade : 3  (staged TPs across three independent positions)
    """

    CHOP_STATE = 2

    def __init__(self, balance: float, tf: str = "H1", broker: str = "headway_cent"):
        self.balance = max(float(balance), MIN_CAPITAL_USD)
        self.tf      = tf.upper()
        self.broker  = broker
        tier = "small" if self.balance <= SMALL_ACCOUNT_THRESHOLD else "growth"
        logger.debug(
            "AdaptiveRiskManager: balance=$%.2f tier=%s tf=%s broker=%s",
            self.balance, tier, self.tf, self.broker,
        )

    @property
    def is_small_account(self) -> bool:
        return self.balance <= SMALL_ACCOUNT_THRESHOLD

    def get_trade_limits(self, market_state: int = None, tf: str = None) -> dict:
        """Return ``pos_per_trade`` for the current balance tier.

        Args:
            market_state: Unused — kept for call-site compatibility.
            tf: Timeframe string (e.g. ``"M5"``). Defaults to ``self.tf``.

        Returns:
            dict with key ``pos_per_trade``.
        """
        if self.is_small_account:
            # Standard accounts: single position on micro-balance for margin safety
            if self.broker == "standard":
                return {"pos_per_trade": 1}
            return {"pos_per_trade": 2}
        return {"pos_per_trade": 3}

    def calculate_lot_size(self, stop_loss_pips: float) -> float:
        """1% risk rule for Headway cent XAUUSD.

        On the Headway cent account the effective contract is 1 oz per lot, so
        each $1 price move = $1 P&L per lot.  Formula:

            lot = (balance × 1%) / sl_price_distance

        Example: $15 balance, 14.27-pt SL → $0.15 / 14.27 = 0.0105 → 0.01 lot.
        """
        risk_per_trade = self.balance * 0.01
        if stop_loss_pips <= 0:
            return 0.01
        lot_size = risk_per_trade / stop_loss_pips
        return max(0.01, round(lot_size, 2))

    def __repr__(self) -> str:
        tier = "small" if self.is_small_account else "growth"
        return (
            f"AdaptiveRiskManager(balance=${self.balance:.0f}, tier={tier}, "
            f"broker={self.broker})"
        )


class DailyEquityGate:
    """Floating-equity safety switch for live trading.

    Two-sided gate that blocks new signals when either limit is breached:

    Loss side (universal):
        If running equity (balance + open floating P&L) drops ≥ ``loss_pct``
        below the start-of-day baseline, the loss gate locks and all open
        positions are closed.

    Profit side (Trailing Daily Equity Lock):
        If running equity rises ≥ ``PROFIT_LOCK_PCT[tf]`` above the
        start-of-day baseline, the profit gate locks — banking the day's gains
        before a late-session regime shift can give them back.

        Thresholds by TF:
            M5:     20% day gain  (scalp sessions can spike fast)
            M15:    10% day gain  (intraday sessions)
            H1:     10% day gain  (swing sessions)

    Both gates reset at UTC midnight via ``reset_day``.

    Usage in the live loop::

        gate = DailyEquityGate(tf="M5")
        gate.reset_day(live_account_balance_usd)   # call once at start of day

        # each poll cycle:
        current_eq = account_size + open_pnl
        if gate.check(current_eq):
            if gate.needs_loss_notification:
                <close all positions + send Telegram loss alert>
            elif gate.needs_profit_notification:
                <send Telegram equity-locked alert>
            continue   # skip new signals until tomorrow
    """

    DAILY_LOSS_PCT = 0.05   # 5% default loss limit

    # Trailing Daily Equity Lock thresholds per timeframe
    PROFIT_LOCK_PCT = {
        "M5":  0.20,   # 20% — fast-moving scalp sessions
        "M15": 0.10,   # 10% — intraday sessions
        "H1":  0.10,   # 10% — swing sessions
    }

    def __init__(self, loss_pct: float = DAILY_LOSS_PCT, tf: str = "H1"):
        self.loss_pct         = loss_pct
        self.profit_lock_pct  = self.PROFIT_LOCK_PCT.get(tf.upper(), 0.10)
        self._tf              = tf.upper()
        self._start_equity: float | None = None
        self._loss_locked     = False
        self._profit_locked   = False
        self._loss_notified   = False
        self._profit_notified = False

    def reset_day(self, equity: float) -> None:
        """Initialise or reset for a new UTC trading day."""
        self._start_equity    = float(equity)
        self._loss_locked     = False
        self._profit_locked   = False
        self._loss_notified   = False
        self._profit_notified = False
        logger.debug(
            "DailyEquityGate [%s] reset: start_equity=%.2f  loss=%.0f%%  profit_lock=%.0f%%",
            self._tf, equity, self.loss_pct * 100, self.profit_lock_pct * 100,
        )

    def check(self, current_equity: float) -> bool:
        """Return True if either the loss or profit gate has been triggered.

        Calling this repeatedly after lock is safe — it stays True and does
        NOT re-trigger notifications.
        """
        if self._start_equity is None or self._start_equity <= 0:
            return False
        change_pct = (current_equity - self._start_equity) / self._start_equity
        if -change_pct >= self.loss_pct:
            self._loss_locked = True
        if change_pct >= self.profit_lock_pct:
            self._profit_locked = True
        return self._loss_locked or self._profit_locked

    @property
    def locked(self) -> bool:
        """True if either gate is active."""
        return self._loss_locked or self._profit_locked

    @property
    def loss_locked(self) -> bool:
        return self._loss_locked

    @property
    def profit_locked(self) -> bool:
        return self._profit_locked

    @property
    def needs_notification(self) -> bool:
        """Backwards-compat alias → ``needs_loss_notification``."""
        return self.needs_loss_notification

    @property
    def needs_loss_notification(self) -> bool:
        """True exactly once — the first poll cycle after the loss gate engages."""
        if self._loss_locked and not self._loss_notified:
            self._loss_notified = True
            return True
        return False

    @property
    def needs_profit_notification(self) -> bool:
        """True exactly once — the first poll cycle after the profit gate engages."""
        if self._profit_locked and not self._profit_notified:
            self._profit_notified = True
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"DailyEquityGate(tf={self._tf}, loss={self.loss_pct*100:.0f}%, "
            f"profit_lock={self.profit_lock_pct*100:.0f}%, "
            f"loss_locked={self._loss_locked}, profit_locked={self._profit_locked}, "
            f"start=${self._start_equity or 0:.2f})"
        )
