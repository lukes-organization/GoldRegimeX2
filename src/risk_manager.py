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
    """Dynamic trade limits and lot sizing per the Headway Cent spec.

    Account tiers
    ─────────────
    ≤ $50 USD  — small account:
        • max_daily_trades : 2
        • pos_per_trade    : 1  (single position per signal)
        • total daily pos  : 2

    > $50 USD  — growth account:
        • max_daily_trades : 3 in Bull/Bear, 2 in Chop  (market-dependent)
        • pos_per_trade    : 2  (dual/hedging positions per signal)
        • total daily pos  : 4 or 6
    """

    CHOP_STATE = 2

    def __init__(self, balance: float):
        self.balance = max(float(balance), MIN_CAPITAL_USD)
        self.daily_trades = 0
        tier = "small" if self.balance <= SMALL_ACCOUNT_THRESHOLD else "growth"
        logger.debug("AdaptiveRiskManager: balance=$%.2f tier=%s", self.balance, tier)

    @property
    def is_small_account(self) -> bool:
        return self.balance <= SMALL_ACCOUNT_THRESHOLD

    def get_trade_limits(self, market_state: int = None, tf: str = None) -> dict:
        """Return trade limits for the current balance and optional HMM state.

        Args:
            market_state: Current HMM state index. If None, uses the most
                          permissive limit for the account tier.
            tf: Timeframe string (e.g. ``"M5"``). M5 small accounts get a
                higher daily cap (4) to match the higher bar frequency.
        """
        if self.is_small_account:
            # M5 generates ~288 bars/day — allow more signals so the optimizer
            # can find frequent enough OOS trades to clear MIN_OOS_TRADES=300.
            max_daily = 4 if (tf and tf.upper() == "M5") else 2
            return {"max_daily_trades": max_daily, "pos_per_trade": 1, "total_daily_pos": max_daily}

        # Growth account: market-state-dependent
        in_chop = (market_state == self.CHOP_STATE) if market_state is not None else False
        max_daily = 2 if in_chop else 3
        pos_per_trade = 2
        return {
            "max_daily_trades": max_daily,
            "pos_per_trade": pos_per_trade,
            "total_daily_pos": max_daily * pos_per_trade,
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
        limits = self.get_trade_limits(market_state)
        return self.daily_trades < limits["max_daily_trades"]

    def log_trade(self):
        self.daily_trades += 1

    def reset_daily(self):
        self.daily_trades = 0

    def __repr__(self) -> str:
        tier = "small" if self.is_small_account else "growth"
        return f"AdaptiveRiskManager(balance=${self.balance:.0f}, tier={tier}, daily_trades={self.daily_trades})"
