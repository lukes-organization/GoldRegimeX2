"""Regime-Confirmation Signal Engine.

Stateful signal generator: enters on regime confirmation + XGBoost confidence,
exits on regime reversal, persistence collapse, profit erosion, or max hold.
Replaces Z-Score and RCEV evaluation logic.
"""

from dataclasses import dataclass, field

import numpy as np

from src.engine_hmm import (
    REGIME_TREND, REGIME_MR, REGIME_SHOCK,
    CANONICAL_REGIME_ID, STATE_NAMES, STATE_POLICY,
)

# Canonical state integer IDs (sourced from engine_hmm to stay in sync)
TREND_STATE = CANONICAL_REGIME_ID[REGIME_TREND]   # 0
MR_STATE    = CANONICAL_REGIME_ID[REGIME_MR]       # 1
SHOCK_STATE = CANONICAL_REGIME_ID[REGIME_SHOCK]    # 2

# ── Trade-allowed policy (sourced from engine_hmm) ──────────────────────────
# TREND  → True  (directional move, use trend_model)
# MR     → False (strict no-trade: mean-reversion = noise regime)
# SHOCK  → True  (explosive volatility, use shock_model)
# This is the single canonical source of state-to-trade permission.

# Minimum consecutive bars in a regime before entry is allowed.
MIN_CONFIRMATION_BARS = {"H1": 4, "M15": 3, "M5": 2}

# Minimum consecutive bars in a NEW (different) regime before regime-reversal
# exit fires.
MIN_EXIT_CONFIRM_BARS = {"H1": 2, "M15": 2, "M5": 2}

# XGBoost probability threshold for trend and SHOCK entries.
# M5 keeps a dynamic rolling percentile gate in evaluate_bar().
ENTRY_PROB = {"H1": 0.58, "M15": 0.53}

# Z-Score thresholds for TREND and SHOCK entries.
MIN_TREND_ZSCORE = {"H1": 0.5, "M15": 1.0, "M5": 1.5}
MIN_SHOCK_ZSCORE = {"H1": 0.8, "M15": 1.2, "M5": 1.8}   # SHOCK needs higher VIX expansion

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

    H1/M15: fixed threshold gate required.
    M5: dynamic rolling percentile gate handled separately.

    This single function is the canonical source used by both backtester and
    live trader, eliminating ad hoc per-module guards.
    """
    return tf.upper() in ENTRY_PROB


@dataclass
class SignalDecision:
    """Unified trading decision returned by SignalEngine.evaluate_bar().

    Both backtest and live paths consume this object, ensuring identical
    policy logic regardless of execution environment.
    """
    action: str
    enter: bool
    exit: bool
    direction: int                  # +1 = long, -1 = short, 0 = flat
    confidence: float               # XGBoost probability of upward move
    regime: str                     # human-readable regime label
    reason: str                     # short tag explaining the decision
    position_size_multiplier: float
    sl_atr_multiple: float
    tp_atr_multiple: float
    metadata: dict = field(default_factory=dict)

    @classmethod
    def no_trade(
        cls,
        confidence: float,
        regime: str,
        reason: str,
        metadata: dict | None = None,
    ) -> "SignalDecision":
        return cls(
            action="NO_TRADE",
            enter=False,
            exit=False,
            direction=0,
            confidence=float(confidence),
            regime=regime,
            reason=reason,
            metadata=metadata or {},
            position_size_multiplier=1.0,
            sl_atr_multiple=0.0,
            tp_atr_multiple=0.0,
        )

    @classmethod
    def exit_trade(
        cls,
        confidence: float,
        regime: str,
        reason: str,
        metadata: dict | None = None,
    ) -> "SignalDecision":
        return cls(
            action="EXIT",
            enter=False,
            exit=True,
            direction=0,
            confidence=float(confidence),
            regime=regime,
            reason=reason,
            metadata=metadata or {},
            position_size_multiplier=1.0,
            sl_atr_multiple=0.0,
            tp_atr_multiple=0.0,
        )

    @classmethod
    def enter_trade(
        cls,
        direction: int,
        confidence: float,
        regime: str,
        reason: str,
        size_multiplier: float,
        sl_atr_multiple: float,
        tp_atr_multiple: float,
        metadata: dict | None = None,
    ) -> "SignalDecision":
        return cls(
            action="ENTER_LONG" if direction > 0 else "ENTER_SHORT",
            enter=True,
            exit=False,
            direction=int(direction),
            confidence=float(confidence),
            regime=regime,
            reason=reason,
            metadata=metadata or {},
            position_size_multiplier=float(size_multiplier),
            sl_atr_multiple=float(sl_atr_multiple),
            tp_atr_multiple=float(tp_atr_multiple),
        )


class SignalEngine:
    """Stateful signal engine for one live/backtest session.

    Call ``update_regime()`` every bar, then ``should_enter()`` or
    ``should_exit()`` depending on current trade state.  Reset the engine
    when the session ends or at the start of a new backtest run.

    The ``evaluate_bar()`` method provides a unified high-level interface
    that returns a :class:`SignalDecision` for both backtest and live paths.
    """

    # Regime label lookup: canonical int ID → semantic string
    _REGIME_LABELS = STATE_NAMES   # {0: "TREND", 1: "MEAN_REVERSION", 2: "VOLATILITY_SHOCK"}

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
        base_meta = {
            "tf": self.tf,
            "state": state,
            "bars_in_regime": int(regime_info.get("bars_in_regime", 0)),
            "stability": float(regime_info.get("stability", 0.0)),
            "probability": prob,
            "synth_vix_zscore": vix_z,
            "atr_band_position": atr_bnd,
        }

        # ── Exit check first (if in a position) ───────────────────────────────
        if current_position:
            cur_pnl       = float(current_position.get("cur_pnl", 0.0))
            max_fav_pnl   = float(current_position.get("max_fav_pnl", 0.0))
            bars_in_trade = int(current_position.get("bars_in_trade", 0))
            exit_now, reason = self.should_exit(
                regime_info, cur_pnl, max_fav_pnl, bars_in_trade
            )
            if exit_now:
                exit_meta = dict(base_meta)
                exit_meta.update({
                    "cur_pnl": cur_pnl,
                    "max_fav_pnl": max_fav_pnl,
                    "bars_in_trade": bars_in_trade,
                })
                return SignalDecision.exit_trade(
                    confidence=prob,
                    regime=regime_label,
                    reason=reason,
                    metadata=exit_meta,
                )

        # ── Entry check (if flat) ──────────────────────────────────────────────
        if not current_position:
            # Hard block: MEAN_REVERSION always returns no-trade before other filters.
            if regime_label == REGIME_MR:
                return SignalDecision.no_trade(
                    confidence=prob,
                    regime=regime_label,
                    reason="mean_reversion_state",
                    metadata=base_meta,
                )

            # Strict STATE_POLICY fallback for unknown labels.
            if not STATE_POLICY.get(regime_label, False):
                return SignalDecision.no_trade(
                    confidence=prob,
                    regime=regime_label,
                    reason="state_disabled",
                    metadata=base_meta,
                )

            if should_apply_prob_gate(self.tf):
                _min_prob = float(ENTRY_PROB.get(self.tf, 0.0))
                _confidence = max(prob, 1.0 - prob)
                if _confidence < _min_prob:
                    gate_meta = dict(base_meta)
                    gate_meta["required_confidence"] = _min_prob
                    gate_meta["directional_confidence"] = _confidence
                    return SignalDecision.no_trade(
                        confidence=prob,
                        regime=regime_label,
                        reason="probability_below_threshold",
                        metadata=gate_meta,
                    )

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
                    dyn_meta = dict(base_meta)
                    dyn_meta["dynamic_threshold"] = _dyn_th
                    return SignalDecision.no_trade(
                        confidence=prob,
                        regime=regime_label,
                        reason="m5_prob_below_dynamic_threshold",
                        metadata=dyn_meta,
                    )

            entry = self.should_enter(regime_info, prob, vix_z, atr_bnd,
                                      z_cutoff_bull=tf_config.get("z_cutoff_bull"),
                                      z_cutoff_bear=tf_config.get("z_cutoff_bear"),
                                      entry_probability_threshold=tf_config.get(
                                          "entry_probability_threshold",
                                          tf_config.get("h1_entry_prob"),
                                      ))

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
                    return SignalDecision.no_trade(
                        confidence=prob,
                        regime=regime_label,
                        reason="m15_gate_rejected",
                        metadata=base_meta,
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
                        meta_reject = dict(base_meta)
                        meta_reject["meta_probability"] = _meta_prob
                        return SignalDecision.no_trade(
                            confidence=prob,
                            regime=regime_label,
                            reason="meta_label_reject",
                            metadata=meta_reject,
                        )

            if entry:
                sig       = entry["signal"]
                direction = 1 if sig == "BUY" else -1
                enter_meta = dict(base_meta)
                enter_meta.update({
                    "entry_regime": entry.get("regime"),
                    "size_multiplier": float(entry.get("size_multiplier", 1.0)),
                })
                return SignalDecision.enter_trade(
                    direction=direction,
                    confidence=prob,
                    regime=regime_label,
                    reason=f"entry_{entry.get('regime', 'unknown').lower()}_{sig.lower()}",
                    size_multiplier=entry.get("size_multiplier", 1.0),
                    sl_atr_multiple=tf_config.get("sl_atr_multiple", 2.0),
                    tp_atr_multiple=tf_config.get("tp_atr_multiple", 1.5),
                    metadata=enter_meta,
                )

        return SignalDecision.no_trade(
            confidence=prob,
            regime=regime_label,
            reason="no_signal",
            metadata=base_meta,
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
        entry_probability_threshold: float | None = None,
    ) -> dict | None:
        """Evaluate entry conditions for the current bar.

        Routing:
            TREND  (state 0) → trend_model probability gate + VIX z-score
            SHOCK  (state 2) → shock_model probability gate + higher VIX z-score
            MR     (state 1) → always returns None (strict no-trade per STATE_POLICY)

        z_cutoff_bull / z_cutoff_bear: optional per-call overrides for minimum
            synth_vix_zscore used by sensitivity sweeps.  Supersede constants.
        """
        if self.in_trade:
            return None

        state     = regime_info["state"]
        bars      = regime_info["bars_in_regime"]
        stability = regime_info["stability"]
        label     = self._REGIME_LABELS.get(state, f"state_{state}")

        # ── Strict STATE_POLICY gate ─────────────────────────────────────────────
        # MEAN_REVERSION is unconditionally disabled.  Any other unknown state
        # also defaults to no-trade.
        if not STATE_POLICY.get(label, False):
            return None   # reason tracked at evaluate_bar level

        eff_buy  = xgb_prob
        eff_sell = 1.0 - xgb_prob

        if should_apply_prob_gate(self.tf):
            thr = (
                float(entry_probability_threshold)
                if entry_probability_threshold is not None
                else float(ENTRY_PROB.get(self.tf, 0.0))
            )
            prob_req_buy = eff_buy >= thr
            prob_req_sell = eff_sell >= thr
        else:
            prob_req_buy = True
            prob_req_sell = True

        # Effective Z thresholds — caller override takes precedence
        trend_z_bull = (float(z_cutoff_bull) if z_cutoff_bull is not None
                        else MIN_TREND_ZSCORE.get(self.tf, 1.0))
        trend_z_bear = (abs(float(z_cutoff_bear)) if z_cutoff_bear is not None
                        else MIN_TREND_ZSCORE.get(self.tf, 1.0))
        shock_z      = MIN_SHOCK_ZSCORE.get(self.tf, 1.0)

        # ── TREND entry ───────────────────────────────────────────────────────────
        if state == TREND_STATE:
            # Direction is determined by XGB probability
            # prob > 0.5 → upward trend predicted → BUY
            # prob < 0.5 → downward trend predicted → SELL
            if (
                bars >= MIN_CONFIRMATION_BARS.get(self.tf, 2)
                and stability >= PERSISTENCE_MIN.get(self.tf, 0.55)
                and synth_vix_zscore >= trend_z_bull
                and prob_req_buy
                and atr_band_position < ATR_BAND_TREND_MAX
            ):
                return {"signal": "BUY", "size_multiplier": 1.0, "regime": REGIME_TREND}
            if (
                bars >= MIN_CONFIRMATION_BARS.get(self.tf, 2)
                and stability >= PERSISTENCE_MIN.get(self.tf, 0.55)
                and synth_vix_zscore >= trend_z_bear
                and prob_req_sell
                and atr_band_position > ATR_BAND_TREND_MIN
            ):
                return {"signal": "SELL", "size_multiplier": 1.0, "regime": REGIME_TREND}

        # ── VOLATILITY_SHOCK entry ────────────────────────────────────────────
        elif state == SHOCK_STATE:
            # SHOCK: require higher VIX expansion (shock_z) before entering.
            # Use reduced position size — SHOCK regime is volatile, risk is elevated.
            if (
                bars >= MIN_CONFIRMATION_BARS.get(self.tf, 2)
                and stability >= PERSISTENCE_MIN.get(self.tf, 0.55)
                and synth_vix_zscore >= shock_z
                and prob_req_buy
                and atr_band_position < ATR_BAND_TREND_MAX
            ):
                return {"signal": "BUY",  "size_multiplier": 0.75, "regime": REGIME_SHOCK}
            if (
                bars >= MIN_CONFIRMATION_BARS.get(self.tf, 2)
                and stability >= PERSISTENCE_MIN.get(self.tf, 0.55)
                and synth_vix_zscore >= shock_z
                and prob_req_sell
                and atr_band_position > ATR_BAND_TREND_MIN
            ):
                return {"signal": "SELL", "size_multiplier": 0.75, "regime": REGIME_SHOCK}

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

    def _effective_h1_prob_thresholds(self, tf_config: dict | None = None) -> "tuple[float, float]":
        """Return (buy_thr, sell_thr) for H1 entry gate, respecting per-run overrides.

        Priority: h1_entry_prob_buy/sell > h1_entry_prob > ENTRY_PROB constant.
        Both thresholds are clamped to [0.50, 0.90].
        """
        tf_config   = tf_config or {}
        default_thr = float(ENTRY_PROB.get("H1", 0.575))
        buy_thr  = float(tf_config.get("h1_entry_prob_buy",
                         tf_config.get("h1_entry_prob", default_thr)))
        sell_thr = float(tf_config.get("h1_entry_prob_sell",
                         tf_config.get("h1_entry_prob", default_thr)))
        buy_thr  = min(max(buy_thr,  0.50), 0.90)
        sell_thr = min(max(sell_thr, 0.50), 0.90)
        return buy_thr, sell_thr

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
