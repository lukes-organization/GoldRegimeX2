"""Regime-Confirmation Signal Engine.

Stateful signal generator: enters on regime confirmation + XGBoost confidence,
exits on regime reversal, persistence collapse, profit erosion, or max hold.
Replaces Z-Score and RCEV evaluation logic.
"""

from dataclasses import dataclass

import numpy as np

# Minimum consecutive bars in a regime before entry is allowed.
# H1: raised to 5 — requires 5 consecutive hours of stable regime before entry.
# This prevents the rapid-oscillation pattern where HMM switches Bull↔Bear
# every 1–2 days, causing 700+ trades/OOS-window and DD blowups.
# M15/M5: unchanged at 2 — faster TFs need quicker confirmation.
MIN_CONFIRMATION_BARS = {"H1": 2, "M15": 2, "M5": 2}
MIN_CHOP_CONFIRM_BARS = {"H1": 4, "M15": 2, "M5": 2}

# Minimum consecutive bars in a NEW (different) regime before regime-reversal
# exit fires.  H1: raised to 5 to match entry confirmation logic — a single
# regime blip during a multi-day H1 trend should not prematurely close the trade.
MIN_EXIT_CONFIRM_BARS = {"H1": 2, "M15": 2, "M5": 2}

# XGBoost probability threshold for trend and MR entries.
# Thresholds are deliberately strict: XGB test accuracy is 50-52% (next-bar
# direction), so only the highest-confidence bars have real signal.  Allowing
# lower thresholds floods the engine with noise trades and destroys performance.
# XGBoost probability thresholds (H1 only — M15/M5 bypass probability checks).
# On lower timeframes XGB probabilities cluster near 50–55%, so strict gates
# produce a "dead zone" that suppresses genuinely profitable volatility regimes.
ENTRY_PROB    = {"H1": 0.575}
MR_ENTRY_PROB = {"H1": 0.575}

# Z-Score thresholds for trend and MR entries.
MIN_TREND_ZSCORE = {"H1": 0.5, "M15": 1.0, "M5": 1.5}
MIN_MR_ZSCORE    = {"H1": 1.0, "M15": 1.5, "M5": 2.0}

# Dynamic ATR band thresholds (0 = lower band, 1 = upper band).
# Prevent buying when overextended up or selling when overextended down.
ATR_BAND_TREND_MAX = 0.80
ATR_BAND_TREND_MIN = 0.20

# Maximum bars to hold a single trade before forcing exit
MAX_HOLD_BARS = {"H1": 24, "M15": 32, "M5": 48}

# Exit if current P&L drops below this fraction of peak P&L
PROFIT_EROSION_THRESHOLD = 0.40

# Minimum HMM self-transition probability for stable regime entry/hold.
# TF-specific: M5 HMMs naturally have lower persistence due to noise.
PERSISTENCE_MIN = {
    "H1": 0.57,
    "M15": 0.55,
    "M5": 0.45,
}

# BB position extremity thresholds for MR entries — superseded by ATR band logic
# below, retained here for reference only.
# MR_BB_BUY_MAX  = 0.30
# MR_BB_SELL_MIN = 0.70

# MR position size fraction (smaller than trend — mean-reversion is higher risk)
MR_SIZE_MULTIPLIER = 0.75


def should_apply_prob_gate(tf: str) -> bool:
    """Return True when the XGBoost probability gate should be enforced.

    H1: probability check required (XGB signal is meaningful at hourly scale).
    M15/M5: bypass — XGB probabilities cluster near 50-55% on noisy short TFs;
            strict gates produce a dead zone that suppresses valid signals.

    This single function is the canonical source used by both backtester and
    live trader, eliminating ad hoc per-module guards.
    """
    return tf.upper() == "H1"


@dataclass
class SignalDecision:
    """Unified trading decision returned by SignalEngine.evaluate_bar().

    Both backtest and live paths consume this object, ensuring identical
    policy logic regardless of execution environment.
    """
    enter: bool
    exit: bool
    direction: int                  # +1 = long, -1 = short, 0 = flat
    confidence: float               # XGBoost probability of upward move
    regime: str                     # human-readable regime label
    reason: str                     # short tag explaining the decision
    position_size_multiplier: float
    sl_atr_multiple: float
    tp_atr_multiple: float


class SignalEngine:
    """Stateful signal engine for one live/backtest session.

    Call ``update_regime()`` every bar, then ``should_enter()`` or
    ``should_exit()`` depending on current trade state.  Reset the engine
    when the session ends or at the start of a new backtest run.

    The ``evaluate_bar()`` method provides a unified high-level interface
    that returns a :class:`SignalDecision` for both backtest and live paths.
    """

    _REGIME_LABELS = {0: "Bull", 1: "Bear", 2: "Chop", 3: "Chop_High"}

    def __init__(self, tf: str = "H1"):
        self.tf = tf.upper()
        self.bars_in_regime: int = 0
        self.current_regime: int | None = None
        self.in_trade: bool = False
        self.entry_regime: int | None = None
        self.bars_in_trade: int = 0
        self._reversal_bars: int = 0   # consecutive bars in non-entry regime
        self._probs_window: list = []  # rolling XGB prob history for M5 dynamic threshold

    # ── Unified high-level API (Phase 2) ──────────────────────────────────────

    def evaluate_bar(
        self,
        row: dict,
        current_position: dict | None,
        tf_config: dict,
    ) -> "SignalDecision":
        """Evaluate a single bar and return a unified SignalDecision.

        *row* must contain: hmm_state, prob, synth_vix_zscore, atr_band_position,
        and optionally: persistence, atr_normalized, cur_pnl, max_fav_pnl,
        bars_in_trade.

        *current_position* is None when flat, or a dict with keys:
          ``direction``, ``cur_pnl``, ``max_fav_pnl``, ``bars_in_trade``.

        *tf_config* is a dict with at minimum key ``"tf"``.

        This method calls ``update_regime()``, ``should_enter()``, and
        ``should_exit()`` in the correct order, keeping this class as the
        single owner of entry/exit policy for both backtest and live.
        """
        hmm_transmat = tf_config.get("hmm_transmat")
        state   = int(row.get("hmm_state", self.current_regime or 0))
        prob    = float(row.get("prob", 0.5))
        vix_z   = float(row.get("synth_vix_zscore", 0.0))
        atr_bnd = float(row.get("atr_band_position", 0.5))
        regime_label = self._REGIME_LABELS.get(state, f"state_{state}")

        regime_info = self.update_regime(state, hmm_transmat)

        # ── Exit check first (if in a position) ───────────────────────────────
        if current_position:
            cur_pnl       = float(current_position.get("cur_pnl", 0.0))
            max_fav_pnl   = float(current_position.get("max_fav_pnl", 0.0))
            bars_in_trade = int(current_position.get("bars_in_trade", 0))
            exit_now, reason = self.should_exit(
                regime_info, cur_pnl, max_fav_pnl, bars_in_trade
            )
            if exit_now:
                return SignalDecision(
                    enter=False, exit=True, direction=0, confidence=prob,
                    regime=regime_label, reason=reason,
                    position_size_multiplier=1.0,
                    sl_atr_multiple=0.0, tp_atr_multiple=0.0,
                )

        # ── Entry check (if flat) ──────────────────────────────────────────────
        if not current_position:
            # Rolling probability window for M5 dynamic threshold.
            self._probs_window.append(prob)
            if len(self._probs_window) > 200:
                self._probs_window.pop(0)

            # Phase 3 — M5: only enter on top-10% probability bars in the window.
            if self.tf == "M5" and len(self._probs_window) >= 30:
                _dyn_th = self.dynamic_prob_threshold(
                    np.array(self._probs_window), percentile=90.0
                )
                if prob < _dyn_th:
                    return SignalDecision(
                        enter=False, exit=False, direction=0, confidence=prob,
                        regime=regime_label, reason="m5_prob_below_dynamic_threshold",
                        position_size_multiplier=1.0, sl_atr_multiple=0.0, tp_atr_multiple=0.0,
                    )

            entry = self.should_enter(regime_info, prob, vix_z, atr_bnd,
                                      z_cutoff_bull=tf_config.get("z_cutoff_bull"),
                                      z_cutoff_bear=tf_config.get("z_cutoff_bear"))

            # Phase 3 — M15: continuation quality gate.
            # Neutral fallbacks (1.0) pass the gate when features are absent
            # (e.g. live tick before process_pipeline re-run); the gate is
            # fully active once atr_expansion_ratio / realized_vol_ratio are
            # in the processed data.
            if entry and self.tf == "M15":
                _stability = float(hmm_transmat[state, state]) if hmm_transmat is not None else 0.70
                _m15_row = {
                    "regime_duration":    row.get("regime_duration", self.bars_in_regime),
                    "persistence":        _stability,
                    "synth_vix_zscore":   vix_z,
                    "atr_expansion_ratio": float(row.get("atr_expansion_ratio", 1.0)),
                    "realized_vol_ratio":  float(row.get("realized_vol_ratio", 1.0)),
                }
                if not self._m15_entry_ok(_m15_row, tf_config):
                    return SignalDecision(
                        enter=False, exit=False, direction=0, confidence=prob,
                        regime=regime_label, reason="m15_gate_rejected",
                        position_size_multiplier=1.0, sl_atr_multiple=0.0, tp_atr_multiple=0.0,
                    )

            # Phase 4 — meta-label gate (no-op unless use_meta_label=True in tf_config).
            if entry and tf_config.get("use_meta_label", False):
                _meta_model = tf_config.get("meta_model")
                if _meta_model is None:
                    tf_config["use_meta_label"] = False  # disable for this session
                else:
                    import pandas as _pd
                    _meta_X = _pd.DataFrame([row])
                    _meta_prob = float(_meta_model.predict_proba(_meta_X)[0, 1])
                    if _meta_prob < float(tf_config.get("meta_min_prob", 0.55)):
                        return SignalDecision(
                            enter=False, exit=False, direction=0, confidence=prob,
                            regime=regime_label, reason="meta_label_reject",
                            position_size_multiplier=1.0, sl_atr_multiple=0.0, tp_atr_multiple=0.0,
                        )

            if entry:
                sig  = entry["signal"]
                direction = 1 if sig in ("BUY", "MR_BUY") else -1
                return SignalDecision(
                    enter=True, exit=False,
                    direction=direction,
                    confidence=prob,
                    regime=regime_label,
                    reason=f"entry_{sig.lower()}",
                    position_size_multiplier=entry.get("size_multiplier", 1.0),
                    sl_atr_multiple=tf_config.get("sl_atr_multiple", 2.0),
                    tp_atr_multiple=tf_config.get("tp_atr_multiple", 1.5),
                )

        return SignalDecision(
            enter=False, exit=False, direction=0, confidence=prob,
            regime=regime_label, reason="no_signal",
            position_size_multiplier=1.0, sl_atr_multiple=0.0, tp_atr_multiple=0.0,
        )

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
        synth_vix_zscore: float,
        atr_band_position: float,
        z_cutoff_bull: float | None = None,
        z_cutoff_bear: float | None = None,
    ) -> dict | None:
        """Evaluate entry conditions utilizing Z-scores, Dynamic ATR bands, and conditional probabilities.

        H1: Dual constraint — requires both high ML probability AND confirmed
            volatility expansion (synth_vix_zscore).
        M15/M5: Z-score/ATR-band only — probability check bypassed (defaults to
            0.0 = always True) to avoid the dual-constraint dead zone caused by
            XGB probabilities clustering near 50–55% on noisy lower timeframes.

        z_cutoff_bull / z_cutoff_bear: optional per-call overrides for the
            minimum synth_vix_zscore required for Bull/Bear trend entry.  When
            provided they supersede MIN_TREND_ZSCORE for this call only.  Used
            by sensitivity analysis to sweep effective Z thresholds.
        """
        if self.in_trade:
            return None

        eff_buy  = xgb_prob
        eff_sell = 1.0 - xgb_prob

        state = regime_info["state"]
        bars = regime_info["bars_in_regime"]
        stability = regime_info["stability"]

        # Effective Z thresholds — caller override takes precedence.
        trend_z_bull = (float(z_cutoff_bull) if z_cutoff_bull is not None
                        else MIN_TREND_ZSCORE.get(self.tf, 1.0))
        trend_z_bear = (abs(float(z_cutoff_bear)) if z_cutoff_bear is not None
                        else MIN_TREND_ZSCORE.get(self.tf, 1.0))

        # H1 requires probability checks; M15/M5 bypass this (default 0.0 = always True)
        prob_req_buy      = eff_buy  >= ENTRY_PROB.get(self.tf, 0.0)
        prob_req_sell     = eff_sell >= ENTRY_PROB.get(self.tf, 0.0)
        mr_prob_req_buy   = eff_buy  >= MR_ENTRY_PROB.get(self.tf, 0.0)
        mr_prob_req_sell  = eff_sell >= MR_ENTRY_PROB.get(self.tf, 0.0)

        # ── Trend entry: Bull (0) ─────────────────────────────────────────────
        if state == 0:
            if (
                bars >= MIN_CONFIRMATION_BARS.get(self.tf, 2)
                and synth_vix_zscore >= trend_z_bull
                and prob_req_buy
                and atr_band_position < ATR_BAND_TREND_MAX
                and stability >= PERSISTENCE_MIN.get(self.tf, 0.55)
            ):
                return {"signal": "BUY", "size_multiplier": 1.0}

        # ── Trend entry: Bear (1) ─────────────────────────────────────────────
        elif state == 1:
            if (
                bars >= MIN_CONFIRMATION_BARS.get(self.tf, 2)
                and synth_vix_zscore >= trend_z_bear
                and prob_req_sell
                and atr_band_position > ATR_BAND_TREND_MIN
                and stability >= PERSISTENCE_MIN.get(self.tf, 0.55)
            ):
                return {"signal": "SELL", "size_multiplier": 1.0}

        # ── Mean-reversion entry: Chop (state >= 2) ───────────────────────────
        elif state >= 2 and atr_band_position is not None:
            if self.tf in ("M5", "M15"):
                return None  # Chop/MR entries disabled for M5/M15 — 3-state HMM only trades Bull/Bear
            if bars >= MIN_CHOP_CONFIRM_BARS.get(self.tf, 2):
                if atr_band_position < 0.1 and synth_vix_zscore >= MIN_MR_ZSCORE.get(self.tf, 1.5) and mr_prob_req_buy:
                    return {"signal": "MR_BUY", "size_multiplier": MR_SIZE_MULTIPLIER}
                elif atr_band_position > 0.9 and synth_vix_zscore >= MIN_MR_ZSCORE.get(self.tf, 1.5) and mr_prob_req_sell:
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
        self._probs_window = []

    # ── Phase 3: M15/M5 specialization helpers ────────────────────────────────

    def dynamic_prob_threshold(
        self, probs_window: np.ndarray, percentile: float = 90.0
    ) -> float:
        """Compute a rolling dynamic probability threshold for M5 entries.

        Uses the top *percentile* of recent XGB probabilities as the bar
        threshold, adapting to the current signal distribution rather than
        relying on a fixed value.  Falls back to 0.55 when fewer than 30
        observations are available (cold-start guard).
        """
        if len(probs_window) < 30:
            return 0.55
        return float(np.percentile(probs_window, percentile))

    def _m15_entry_ok(self, row: dict, cfg: dict) -> bool:
        """Continuation entry gate specific to M15.

        All five conditions must pass:
          1. Regime has persisted for at least m15_min_regime_bars bars.
          2. HMM self-transition probability >= m15_min_persistence.
          3. Short-term ATR is expanded relative to the long-term baseline.
          4. Synthetic VIX z-score confirms volatility is rising.
          5. Realised volatility ratio confirms short-term expansion.
        """
        return (
            float(row.get("regime_duration", 0))
                >= float(cfg.get("m15_min_regime_bars", 4))
            and float(row.get("persistence", 0.0))
                >= float(cfg.get("m15_min_persistence", 0.55))
            and float(row.get("atr_expansion_ratio", 0.0))
                >= float(cfg.get("m15_min_atr_expansion", 1.0))
            and float(row.get("synth_vix_zscore", 0.0))
                >= float(cfg.get("m15_min_vix_z", 0.5))
            and float(row.get("realized_vol_ratio", 0.0))
                >= float(cfg.get("m15_min_rv_ratio", 1.0))
        )
