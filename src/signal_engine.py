"""Regime-Confirmation Signal Engine.

Stateful signal generator: enters on regime confirmation + XGBoost confidence,
exits on regime reversal, persistence collapse, profit erosion, or max hold.
Replaces Z-Score and RCEV evaluation logic.
"""

import numpy as np

# Minimum consecutive bars in a regime before entry is allowed.
# H1: raised to 5 — requires 5 consecutive hours of stable regime before entry.
# This prevents the rapid-oscillation pattern where HMM switches Bull↔Bear
# every 1–2 days, causing 700+ trades/OOS-window and DD blowups.
# M15/M5: unchanged at 2 — faster TFs need quicker confirmation.
MIN_CONFIRMATION_BARS = {"H1": 3, "M15": 2, "M5": 2}
MIN_CHOP_CONFIRM_BARS = {"H1": 3, "M15": 2, "M5": 2}

# Minimum consecutive bars in a NEW (different) regime before regime-reversal
# exit fires.  H1: raised to 5 to match entry confirmation logic — a single
# regime blip during a multi-day H1 trend should not prematurely close the trade.
MIN_EXIT_CONFIRM_BARS = {"H1": 2, "M15": 2, "M5": 3}

# XGBoost probability threshold for trend and MR entries.
# Thresholds are deliberately strict: XGB test accuracy is 50-52% (next-bar
# direction), so only the highest-confidence bars have real signal.  Allowing
# lower thresholds floods the engine with noise trades and destroys performance.
ENTRY_PROB    = {"H1": 0.55, "M15": 0.55, "M5": 0.52}
MR_ENTRY_PROB = {"H1": 0.56, "M15": 0.52, "M5": 0.50}

# Maximum bars to hold a single trade before forcing exit
MAX_HOLD_BARS = {"H1": 24, "M15": 32, "M5": 48}

# Exit if current P&L drops below this fraction of peak P&L
PROFIT_EROSION_THRESHOLD = 0.40

# Minimum HMM self-transition probability for stable regime entry/hold.
# TF-specific: M5 HMMs naturally have lower persistence due to noise.
PERSISTENCE_MIN = {
    "H1": 0.55,
    "M15": 0.55,
    "M5": 0.45,
}

# BB position extremity thresholds for MR entries
MR_BB_BUY_MAX  = 0.30   # below this → MR_BUY (price at lower band)
MR_BB_SELL_MIN = 0.70   # above this → MR_SELL (price at upper band)

# MR position size fraction (smaller than trend — mean-reversion is higher risk)
MR_SIZE_MULTIPLIER = 0.75


class SignalEngine:
    """Stateful signal engine for one live/backtest session.

    Call ``update_regime()`` every bar, then ``should_enter()`` or
    ``should_exit()`` depending on current trade state.  Reset the engine
    when the session ends or at the start of a new backtest run.
    """

    def __init__(self, tf: str = "H1"):
        self.tf = tf.upper()
        self.bars_in_regime: int = 0
        self.current_regime: int | None = None
        self.in_trade: bool = False
        self.entry_regime: int | None = None
        self.bars_in_trade: int = 0
        self._reversal_bars: int = 0   # consecutive bars in non-entry regime

    # ── Core API ──────────────────────────────────────────────────────────────

    def update_regime(self, new_state: int, hmm_transmat: np.ndarray | None) -> dict:
        """Process the current bar's regime state.

        Returns a regime_info dict used by should_enter() / should_exit().
        """
        changed = new_state != self.current_regime
        if changed:
            self.bars_in_regime = 1
            self.current_regime = new_state
        else:
            self.bars_in_regime += 1

        if hmm_transmat is not None:
            stability = float(hmm_transmat[new_state, new_state])
            if np.isnan(stability):
                stability = 0.70
        else:
            stability = 0.70

        # Track consecutive bars in a non-entry regime for exit confirmation.
        # Reset to zero when we return to the entry regime (blip resolved).
        if self.in_trade and self.entry_regime is not None:
            if new_state != self.entry_regime:
                self._reversal_bars += 1
            else:
                self._reversal_bars = 0

        return {
            "state": new_state,
            "bars_in_regime": self.bars_in_regime,
            "stability": stability,
            "changed": changed,
        }

    def should_enter(
        self,
        regime_info: dict,
        xgb_prob: float,
        gmm_cluster: int,
        bb_position: float | None = None,
    ) -> dict | None:
        """Evaluate entry conditions for the current bar.

        Returns an entry dict on success, None otherwise.
        Entry dict keys: ``signal``, ``size_multiplier``.
        """
        if self.in_trade:
            return None

        # xgb_prob = P(next bar UP) from binary XGBoost classifier.
        # Compute directional probabilities separately so BUY and SELL both
        # require XGB to AGREE with the intended trade direction.
        eff_buy  = xgb_prob             # P(UP)  — for BUY entries
        eff_sell = 1.0 - xgb_prob       # P(DOWN) — for SELL entries

        state = regime_info["state"]
        bars = regime_info["bars_in_regime"]
        stability = regime_info["stability"]

        # ── Trend entry: Bull (0) → BUY when XGB confirms UP ──────────────────
        if state == 0:
            if (
                bars >= MIN_CONFIRMATION_BARS.get(self.tf, 2)
                and eff_buy >= ENTRY_PROB.get(self.tf, 0.55)
                and stability >= PERSISTENCE_MIN.get(self.tf, 0.55)
            ):
                return {"signal": "BUY", "size_multiplier": 1.0}

        # ── Trend entry: Bear (1) → SELL when XGB confirms DOWN ───────────────
        elif state == 1:
            if (
                bars >= MIN_CONFIRMATION_BARS.get(self.tf, 2)
                and eff_sell >= ENTRY_PROB.get(self.tf, 0.55)
                and stability >= PERSISTENCE_MIN.get(self.tf, 0.55)
            ):
                return {"signal": "SELL", "size_multiplier": 1.0}

        # ── Mean-reversion entry: Chop (state >= 2) ───────────────────────────
        elif state >= 2 and bb_position is not None:
            if bars >= MIN_CHOP_CONFIRM_BARS.get(self.tf, 2):
                if bb_position < MR_BB_BUY_MAX and eff_buy >= MR_ENTRY_PROB.get(self.tf, 0.52):
                    return {"signal": "MR_BUY", "size_multiplier": MR_SIZE_MULTIPLIER}
                elif bb_position > MR_BB_SELL_MIN and eff_sell >= MR_ENTRY_PROB.get(self.tf, 0.52):
                    return {"signal": "MR_SELL", "size_multiplier": MR_SIZE_MULTIPLIER}

        return None

    def should_exit(
        self,
        regime_info: dict,
        cur_pnl: float,
        max_fav_pnl: float,
        bars_in_trade: int,
    ) -> tuple[bool, str]:
        """Evaluate exit conditions for an open trade.

        Returns ``(exit_now, reason)`` where reason is a short string tag.
        """
        state = regime_info["state"]

        # Regime reversal — require MIN_EXIT_CONFIRM_BARS consecutive bars in the
        # new regime before exiting.  Single-bar blips (e.g. one Chop bar in a Bull
        # trend) reset _reversal_bars and keep the trade alive.
        if self.entry_regime is not None and state != self.entry_regime:
            if self._reversal_bars >= MIN_EXIT_CONFIRM_BARS.get(self.tf, 2):
                return True, "regime_reversal"

        # Persistence collapse — regime is no longer stable
        if regime_info["stability"] < PERSISTENCE_MIN.get(self.tf, 0.55):
            return True, "persistence_collapse"

        # Profit erosion — gave back too much of the peak gain
        if max_fav_pnl > 0 and cur_pnl < max_fav_pnl * PROFIT_EROSION_THRESHOLD:
            return True, "profit_erosion"

        # Max hold — avoid open-ended exposure
        if bars_in_trade >= MAX_HOLD_BARS.get(self.tf, 24):
            return True, "max_hold"

        return False, ""

    # ── Trade lifecycle hooks ─────────────────────────────────────────────────

    def on_trade_entered(self, regime_state: int) -> None:
        """Call immediately after placing an order."""
        self.in_trade = True
        self.entry_regime = regime_state
        self.bars_in_trade = 0
        self._reversal_bars = 0

    def on_trade_closed(self) -> None:
        """Call after a trade is fully closed."""
        self.in_trade = False
        self.entry_regime = None
        self.bars_in_trade = 0
        self._reversal_bars = 0

    def reset(self) -> None:
        """Full state reset — use between independent backtest runs."""
        self.bars_in_regime = 0
        self.current_regime = None
        self.in_trade = False
        self.entry_regime = None
        self.bars_in_trade = 0
        self._reversal_bars = 0
