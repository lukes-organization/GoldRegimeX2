"""src/risk_manager.py -- position sizing + equity guard for the live/demo engine.

Mirrors the Explorer notebook's DEPLOYED sizing policy (config cell 3 + the
live_settings block exported inside the model bundle): base per-leg lots
LOT_CYCLE_SMALL, escalated to LIVE_SCALED_LOT once realized profit clears
PROFIT_SCALE_THRESHOLD_CENTS.  Headway 'cent' accounts express balance / P&L in
cents (x100), hence CENT_MULTIPLIER.  The live loop uses these EXACT numbers so
live sizing equals the backtester's sizing.
"""
from __future__ import annotations

CENT_MULTIPLIER = 100.0  # headway cent account: 1 USD == 100 account-cents

BROKER_CONFIGS = {
    "headway_cent": {"cent_account": True, "cent_multiplier": CENT_MULTIPLIER},
    "standard": {"cent_account": False, "cent_multiplier": 1.0},
}


def broker_cent_multiplier(broker):
    cfg = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["standard"])
    return float(cfg["cent_multiplier"]) if cfg["cent_account"] else 1.0


class DailyEquityGate:
    """Blocks NEW entries once the day's realized drawdown breaches a fraction of
    the day's starting equity (same intent as Explorer cell 20's
    ProductionRiskCircuitBreaker).  Reset per calendar day."""

    def __init__(self, start_equity, max_daily_loss_frac=0.10):
        self.start_equity = float(start_equity)
        self.max_daily_loss_frac = float(max_daily_loss_frac)
        self.day = None
        self.day_start_equity = float(start_equity)

    def update_day(self, today, equity):
        if self.day != today:
            self.day = today
            self.day_start_equity = float(equity)

    def can_enter(self, equity):
        floor = self.day_start_equity * (1.0 - self.max_daily_loss_frac)
        return float(equity) > floor


class AdaptiveRiskManager:
    """Per-leg lot sizing identical to the deployed live-simulation policy."""

    def __init__(self, settings):
        s = settings or {}
        self.lot_cycle_small = list(s.get("lot_cycle_small", [0.02, 0.02]))
        self.max_positions = int(s.get("max_positions_per_cycle", len(self.lot_cycle_small)))
        self.profit_scale_threshold_cents = float(s.get("profit_scale_threshold_cents", 5000.0))
        self.enable_scaling = bool(s.get("live_enable_lot_scaling", True))
        self.scaled_lot = float(s.get("live_scaled_lot", 0.03))
        self.initial_balance_cents = float(s.get("initial_balance_cents", 1500.0))

    def leg_lots(self, realized_profit_cents=0.0):
        "Return the per-leg lot list for the current cycle."
        if self.enable_scaling and float(realized_profit_cents) >= self.profit_scale_threshold_cents:
            return [self.scaled_lot for _ in self.lot_cycle_small]
        return list(self.lot_cycle_small)

    @classmethod
    def from_bundle(cls, bundle):
        return cls((bundle or {}).get("settings", {}))
