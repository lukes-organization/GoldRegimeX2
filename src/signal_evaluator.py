"""
Shared adaptive signal evaluation logic — single source of truth for
backtesting, optimisation, and live trading.

The backtester/optimizer use :meth:`SignalEvaluator.evaluate_signal_fast` which
applies pure Z-Score logic with no live-safety gates (vectorisation-friendly).
The live bridge uses :meth:`SignalEvaluator.evaluate_signal` which adds three
sequential gates (chop-stability, transition-probability, Bollinger-Band
confluence) on top of the same Z-Score core.

This design ensures Optuna selects models based on pure signal quality while
live trading adds protective filters without invalidating the optimisation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Default Z-Score configuration ─────────────────────────────────────────────
_DEFAULT_CONFIG: Dict[str, Any] = {
    # Trend-following cutoffs
    "Z_CUTOFF_BULL":  2.5,   # BUY  when z > +2.5σ in Bull regime
    "Z_CUTOFF_BEAR": -2.5,   # SELL when z < −2.5σ in Bear regime
    # Per-chop-state MR cutoffs (Chop_Low is easier to fade than Chop_High)
    # TF-specific overrides applied in __init__ via _TF_CUTOFF_OVERRIDES
    "Z_CUTOFF_CHOP_MR": {2: 3.2, 3: 3.8},
    # Volatility adjustments applied when GMM cluster == 2 (high vol)
    "HIGH_VOL_ADJUSTMENT": 0.3,
    # Live-only MR safety gates
    "MIN_CHOP_BARS_FOR_MR":     3,
    "MIN_CHOP_TRANSITION_PROB": 0.70,
    "MR_BB_LOWER_THRESHOLD":    0.35,  # max BB position for MR_BUY
    "MR_BB_UPPER_THRESHOLD":    0.65,  # min BB position for MR_SELL
    # Fallback fixed thresholds used when regime_stats are absent
    "FALLBACK_PROB_THRESHOLD":  0.65,
    "FALLBACK_SHORT_THRESHOLD": 0.35,
}


## Per-TF Z-Score cutoff overrides ────────────────────────────────────────────
# Applied in SignalEvaluator.__init__ after base config, before caller config.
# M5 uses higher cutoffs (noisier bars → only fire on genuinely extreme probs).
# H1/M15 use slightly lower MR cutoffs (fewer chop bars → need more triggered).
_TF_CUTOFF_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "H1": {
        "Z_CUTOFF_BULL":     2.5,
        "Z_CUTOFF_BEAR":    -2.5,
        "Z_CUTOFF_CHOP_MR": {2: 3.0, 3: 3.5},
        "MIN_CHOP_BARS_FOR_MR": 2,
    },
    "M15": {
        "Z_CUTOFF_BULL":     2.0,
        "Z_CUTOFF_BEAR":    -2.0,
        "Z_CUTOFF_CHOP_MR": {2: 3.0, 3: 3.5},
        "MIN_CHOP_BARS_FOR_MR": 3,
    },
    "M5": {
        "Z_CUTOFF_BULL":     2.8,
        "Z_CUTOFF_BEAR":    -2.8,
        "Z_CUTOFF_CHOP_MR": {2: 3.5, 3: 4.0},
        "HIGH_VOL_ADJUSTMENT": 0.4,
        "MIN_CHOP_BARS_FOR_MR": 4,
    },
}


class SignalEvaluator:
    """Regime-specific Z-Score calibrated signal evaluator.

    Args:
        regime_stats: Mapping of HMM state_id → ``{"mean": float, "std": float,
                      "count": int}``.  Compute on IS data via
                      :func:`src.engine_xgb.compute_regime_stats`.
        tf:           Timeframe string — applies TF-specific Z-Score cutoffs from
                      :data:`_TF_CUTOFF_OVERRIDES` (e.g. M5 uses tighter thresholds).
        config:       Optional overrides for :data:`_DEFAULT_CONFIG` (applied last,
                      highest priority).
    """

    def __init__(
        self,
        regime_stats: Dict[int, Dict[str, float]],
        tf: str = "H1",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.regime_stats = regime_stats or {}
        self.config: Dict[str, Any] = {**_DEFAULT_CONFIG}
        # Apply TF-specific overrides, then any caller-supplied config (highest priority)
        self.config.update(_TF_CUTOFF_OVERRIDES.get(tf.upper(), {}))
        if config:
            self.config.update(config)

    # ── Core helpers ────────────────────────────────────────────────────────

    def calculate_z_score(self, prob_buy: float, hmm_state: int) -> float:
        """Return Z-Score: how many σ the current probability is from the IS mean.

        Falls back to mean=0.50, std=0.15 when the state has no calibration data.
        """
        key = int(hmm_state)
        if key in self.regime_stats:
            stats = self.regime_stats[key]
            mean  = stats["mean"]
            std   = max(float(stats["std"]), 0.02)
        else:
            mean, std = 0.50, 0.15
            logger.warning(
                "[Z-CALC] No regime_stats for state %d — fallback mean=0.50 std=0.15",
                hmm_state,
            )
        z = (prob_buy - mean) / std
        logger.debug(
            "[Z-CALC] state=%d prob=%.4f mean=%.4f std=%.4f → z=%.3f",
            hmm_state, prob_buy, mean, std, z,
        )
        return float(z)

    def _vol_adjusted_cutoffs(
        self, gmm_cluster: int
    ) -> Tuple[float, float, Dict[int, float]]:
        """Return (bull_cutoff, bear_cutoff, chop_mr_cutoffs) adjusted for vol."""
        adj  = self.config["HIGH_VOL_ADJUSTMENT"] if gmm_cluster == 2 else 0.0
        bull = self.config["Z_CUTOFF_BULL"]  + adj
        bear = self.config["Z_CUTOFF_BEAR"]  - adj
        chop = {
            s: v + adj
            for s, v in self.config["Z_CUTOFF_CHOP_MR"].items()
        }
        return bull, bear, chop

    def _calculate_tiered_override(self, prob_buy: float) -> float:
        """Amount to reduce the Z-Score requirement based on XGBoost conviction.

        Stronger conviction (probability further from 0.50) → larger reduction.
        Only applied to Bull/Bear trend signals; MR (Chop) is unaffected.
        The effective cutoff is always clamped to a minimum of 1.0 by callers.

        Returns:
            float: Reduction to subtract from the Z cutoff (0.0 to 1.0).
        """
        extremity = abs(prob_buy - 0.50)
        if extremity >= 0.10:    # prob ≤ 0.40 or ≥ 0.60 — very strong conviction
            return 1.0
        elif extremity >= 0.07:  # prob ≤ 0.43 or ≥ 0.57 — strong
            return 0.5
        elif extremity >= 0.04:  # prob ≤ 0.46 or ≥ 0.54 — moderate
            return 0.25
        return 0.0

    # ── Backtester / Optimiser interface ────────────────────────────────────

    def evaluate_signal_fast(
        self,
        prob_buy:    float,
        hmm_state:   int,
        gmm_cluster: int = 1,
        use_tiered:  bool = False,
    ) -> Tuple[Optional[str], float]:
        """Fast, gate-free evaluation for vectorised backtesting.

        Args:
            use_tiered: When True, reduce Bull/Bear Z cutoff by conviction
                        strength (see :meth:`_calculate_tiered_override`).
                        MR (Chop) cutoffs are never modified.  The effective
                        cutoff floor is 1.0.

        Returns:
            ``(signal_or_None, z_score)`` — signal is one of
            ``'BUY' | 'SELL' | 'MR_BUY' | 'MR_SELL' | None``.
        """
        z = self.calculate_z_score(prob_buy, hmm_state)
        bull_cut, bear_cut, chop_cuts = self._vol_adjusted_cutoffs(gmm_cluster)

        if use_tiered and hmm_state in (0, 1):
            reduction = self._calculate_tiered_override(prob_buy)
            if hmm_state == 0:
                bull_cut = max(1.0, bull_cut - reduction)
            else:
                bear_cut = min(-1.0, bear_cut + reduction)

        sig: Optional[str] = None
        if hmm_state == 0:
            if z > bull_cut:
                sig = "BUY"
        elif hmm_state == 1:
            if z < bear_cut:
                sig = "SELL"
        elif hmm_state in (2, 3):
            cut = chop_cuts.get(hmm_state, 2.5)
            if z < -cut:
                sig = "MR_BUY"
            elif z > cut:
                sig = "MR_SELL"

        # Audit assertions — log ERROR if regime/signal mismatch (should never fire)
        if sig == "BUY"  and hmm_state != 0: logger.error("[BUG] BUY in non-Bull state %d",  hmm_state)
        if sig == "SELL" and hmm_state != 1: logger.error("[BUG] SELL in non-Bear state %d", hmm_state)
        if sig in ("MR_BUY", "MR_SELL") and hmm_state not in (2, 3):
            logger.error("[BUG] MR signal %s in non-Chop state %d", sig, hmm_state)

        return (sig, z)

    # ── Live-trading interface ───────────────────────────────────────────────

    def _calculate_confirmation_requirements(
        self,
        hmm_state:         int,
        z_score:           float,
        transition_prob:   float,
        gmm_cluster:       int,
        consec_bars:       int,
        just_entered_state: bool,
        exited_from_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Adaptive per-state confirmation requirements (bars, min-Z, etc.)."""
        abs_z    = abs(z_score)
        high_vol = gmm_cluster == 2
        _, _, chop_cuts = self._vol_adjusted_cutoffs(gmm_cluster)

        req: Dict[str, Any] = {
            "required_bars":     1,
            "min_z_score":       0.0,
            "early_entry_allowed": False,
            "blocked_reason":    None,
            "confidence_level":  "MEDIUM",
            "warning":           None,
        }

        # ── BULL (0): trend BUY ────────────────────────────────────────────
        if hmm_state == 0:
            bull_cut = self.config["Z_CUTOFF_BULL"] + (0.3 if high_vol else 0.0)
            if abs_z >= 3.5:   required_bars, conf = 1, "HIGH"
            elif abs_z >= 3.0: required_bars, conf = 1, "HIGH"
            elif abs_z >= 2.5: required_bars, conf = 2, "MEDIUM"
            else:              required_bars, conf = 3, "LOW"

            if just_entered_state:
                if exited_from_state == 1:
                    required_bars += 1; conf = "LOW"
                    req["warning"] = "Exited Bear — confirming Bull"
                elif exited_from_state in (2, 3):
                    if abs_z >= 3.0 and transition_prob >= 0.75:
                        required_bars = max(1, required_bars - 1)
                        req["early_entry_allowed"] = True
                        req["warning"] = "Chop breakout — accelerated"
                    else:
                        required_bars += 1

            if high_vol: required_bars += 1
            if transition_prob >= 0.80:   required_bars = max(1, required_bars - 1)
            elif transition_prob < 0.65:  required_bars += 1; conf = "LOW"

            req["required_bars"]    = min(4, max(1, required_bars))
            req["min_z_score"]      = bull_cut
            req["confidence_level"] = conf
            if consec_bars < req["required_bars"]:
                req["blocked_reason"] = (
                    f"Bull not established: {consec_bars}/{req['required_bars']} bars"
                )
            elif z_score < req["min_z_score"]:
                req["blocked_reason"] = (
                    f"Bull z={z_score:.2f} < {req['min_z_score']:.1f}"
                )
            return req

        # ── BEAR (1): trend SELL ───────────────────────────────────────────
        if hmm_state == 1:
            bear_cut = abs(self.config["Z_CUTOFF_BEAR"]) + (0.3 if high_vol else 0.0)
            if abs_z >= 3.5:   required_bars, conf = 1, "HIGH"
            elif abs_z >= 3.0: required_bars, conf = 1, "HIGH"
            elif abs_z >= 2.5: required_bars, conf = 2, "MEDIUM"
            else:              required_bars, conf = 3, "LOW"

            if just_entered_state:
                if exited_from_state == 0:
                    required_bars += 1; conf = "LOW"
                    req["warning"] = "Exited Bull — confirming Bear"
                elif exited_from_state in (2, 3):
                    if abs_z >= 3.0 and transition_prob >= 0.75:
                        required_bars = max(1, required_bars - 1)
                        req["early_entry_allowed"] = True
                        req["warning"] = "Chop breakdown — accelerated"
                    else:
                        required_bars += 1

            if high_vol: required_bars += 1
            if transition_prob >= 0.80:   required_bars = max(1, required_bars - 1)
            elif transition_prob < 0.65:  required_bars += 1; conf = "LOW"

            req["required_bars"]    = min(4, max(1, required_bars))
            req["min_z_score"]      = bear_cut
            req["confidence_level"] = conf
            if consec_bars < req["required_bars"]:
                req["blocked_reason"] = (
                    f"Bear not established: {consec_bars}/{req['required_bars']} bars"
                )
            elif z_score > -req["min_z_score"]:
                req["blocked_reason"] = (
                    f"Bear z={z_score:.2f} > -{req['min_z_score']:.1f}"
                )
            return req

        # ── CHOP_LOW (2): MR BUY ─────────────────────────────────────────
        if hmm_state == 2:
            base_z = chop_cuts.get(2, 2.2)
            if abs_z >= 3.5:   required_bars, conf = 1, "HIGH"
            elif abs_z >= 3.0: required_bars, conf = 1, "HIGH"
            elif abs_z >= 2.8: required_bars, conf = 2, "MEDIUM"
            else:              required_bars, conf = 3, "LOW"

            if just_entered_state:
                if exited_from_state == 1:
                    if abs_z >= 2.5:
                        required_bars = max(1, required_bars - 1)
                        req["warning"] = "Natural Bear→Chop_Low — accelerated"
                elif exited_from_state == 0:
                    required_bars += 2; conf = "LOW"
                    req["warning"] = "Suspicious Bull→Chop_Low — extra confirmation"

            if high_vol: required_bars += 1
            if transition_prob >= 0.80:   required_bars = max(1, required_bars - 1)
            elif transition_prob < 0.65:  required_bars += 1

            req["required_bars"]    = min(4, max(1, required_bars))
            req["min_z_score"]      = base_z
            req["confidence_level"] = conf
            if consec_bars < req["required_bars"]:
                req["blocked_reason"] = (
                    f"Chop_Low not stable: {consec_bars}/{req['required_bars']} bars"
                )
            elif abs_z < req["min_z_score"]:
                req["blocked_reason"] = (
                    f"|z|={abs_z:.2f} < {req['min_z_score']:.1f}"
                )
            return req

        # ── CHOP_HIGH (3): MR SELL ────────────────────────────────────────
        if hmm_state == 3:
            base_z = chop_cuts.get(3, 2.8)
            if abs_z >= 4.0:   required_bars, conf = 1, "HIGH"
            elif abs_z >= 3.5: required_bars, conf = 2, "MEDIUM"
            elif abs_z >= 3.0: required_bars, conf = 3, "MEDIUM"
            else:              required_bars, conf = 4, "LOW"

            if just_entered_state:
                if exited_from_state == 0:
                    if abs_z >= 3.0:
                        required_bars = max(1, required_bars - 1)
                        req["warning"] = "Natural Bull→Chop_High — accelerated"
                elif exited_from_state == 1:
                    required_bars += 2; conf = "LOW"
                    req["warning"] = "Suspicious Bear→Chop_High — extra confirmation"

            if high_vol: required_bars += 1
            if transition_prob >= 0.80:   required_bars = max(1, required_bars - 1)
            elif transition_prob < 0.65:  required_bars += 1

            req["required_bars"]    = min(5, max(1, required_bars))
            req["min_z_score"]      = base_z
            req["confidence_level"] = conf
            if consec_bars < req["required_bars"]:
                req["blocked_reason"] = (
                    f"Chop_High not stable: {consec_bars}/{req['required_bars']} bars"
                )
            elif abs_z < req["min_z_score"]:
                req["blocked_reason"] = (
                    f"|z|={abs_z:.2f} < {req['min_z_score']:.1f}"
                )
            return req

        req["blocked_reason"] = f"Unknown HMM state {hmm_state}"
        return req

    def evaluate_signal(
        self,
        prob_buy:        float,
        hmm_state:       int,
        gmm_cluster:     int,
        stability:       Dict[str, Any],
        bb_position:     float = 0.5,
        transition_prob: Optional[float] = None,
        use_tiered:      bool = False,
    ) -> Dict[str, Any]:
        """Full evaluation with all live safety gates.

        Args:
            prob_buy:        XGBoost BUY-class probability.
            hmm_state:       Current HMM regime (0=Bull, 1=Bear, 2=Chop_Low, 3=Chop_High).
            gmm_cluster:     Volatility cluster (0=low, 1=med, 2=high).
            stability:       Dict from :func:`build_stability` tracking consecutive
                             bars, regime changes, etc.
            bb_position:     Price position within Bollinger Bands (0=lower, 1=upper).
            transition_prob: HMM self-transition probability P(stay in current state).
            use_tiered:      When True, apply tiered Z-Score conviction override to
                             Bull/Bear trend signals.  MR (Chop) signals are unaffected.
                             The effective Z cutoff floor is 1.0.
        Returns:
            ``{"signal", "confidence", "reason", "gate_results", "confidence_level"}``
        """
        result: Dict[str, Any] = {
            "signal":           None,
            "confidence":       0.0,
            "reason":           "",
            "gate_results":     {},
            "confidence_level": "MEDIUM",
        }

        z = self.calculate_z_score(prob_buy, hmm_state)
        result["confidence"] = round(z, 3)
        abs_z    = abs(z)
        is_chop  = hmm_state in (2, 3)

        consec        = stability.get("consecutive_bars",   1)
        just_entered  = stability.get("just_entered_state", False)
        exited_from   = stability.get("exited_from_state",  None)
        t_prob        = transition_prob if transition_prob is not None else 0.70

        # ── Universal confirmation gate ────────────────────────────────────
        req = self._calculate_confirmation_requirements(
            hmm_state=hmm_state,
            z_score=z,
            transition_prob=t_prob,
            gmm_cluster=gmm_cluster,
            consec_bars=consec,
            just_entered_state=just_entered,
            exited_from_state=exited_from,
        )
        result["confidence_level"] = req["confidence_level"]

        # ── Tiered Z-Score override (Bull/Bear only) ───────────────────────
        # If use_tiered, reduce req["min_z_score"] based on XGBoost conviction.
        # Only clears a Z-based block — bar-count blocks are never overridden.
        if use_tiered and hmm_state in (0, 1):
            reduction = self._calculate_tiered_override(prob_buy)
            if reduction > 0:
                new_min_z = max(1.0, req["min_z_score"] - reduction)
                req["min_z_score"] = new_min_z
                if req["blocked_reason"]:
                    if hmm_state == 0 and z >= new_min_z:
                        req["blocked_reason"] = None
                    elif hmm_state == 1 and z <= -new_min_z:
                        req["blocked_reason"] = None

        if req["blocked_reason"]:
            result["reason"] = req["blocked_reason"]
            result["gate_results"]["confirmation_failed"] = True
            return result
        result["gate_results"]["confirmation_passed"] = True

        # ── Transition-probability gate (chop only) ────────────────────────
        if is_chop and t_prob < self.config["MIN_CHOP_TRANSITION_PROB"]:
            result["reason"] = (
                f"Chop unstable: P(stay)={t_prob:.2f} "
                f"< {self.config['MIN_CHOP_TRANSITION_PROB']}"
            )
            result["gate_results"]["transition_failed"] = True
            return result
        if is_chop:
            result["gate_results"]["transition_passed"] = True

        # ── Bollinger Band confluence gate (chop only) ─────────────────────
        _, _, chop_cuts = self._vol_adjusted_cutoffs(gmm_cluster)

        if is_chop:
            cut = chop_cuts.get(hmm_state, 2.5)
            if z < -cut:
                # Potential MR_BUY — price must be near lower band
                if bb_position > self.config["MR_BB_LOWER_THRESHOLD"]:
                    result["reason"] = (
                        f"MR_BUY blocked: BB {bb_position:.2f} "
                        f"> {self.config['MR_BB_LOWER_THRESHOLD']:.2f}"
                    )
                    result["gate_results"]["bb_failed"] = True
                    return result
                result["signal"] = "MR_BUY"
                result["gate_results"]["bb_passed"] = True
                result["reason"] = (
                    f"Chop_Low MR_BUY: |z|={abs_z:.2f}>{cut:.1f} "
                    f"({consec} bars, {req['confidence_level']})"
                )
            elif z > cut:
                # Potential MR_SELL — price must be near upper band
                if bb_position < self.config["MR_BB_UPPER_THRESHOLD"]:
                    result["reason"] = (
                        f"MR_SELL blocked: BB {bb_position:.2f} "
                        f"< {self.config['MR_BB_UPPER_THRESHOLD']:.2f}"
                    )
                    result["gate_results"]["bb_failed"] = True
                    return result
                result["signal"] = "MR_SELL"
                result["gate_results"]["bb_passed"] = True
                result["reason"] = (
                    f"Chop_High MR_SELL: |z|={abs_z:.2f}>{cut:.1f} "
                    f"({consec} bars, {req['confidence_level']})"
                )
            else:
                result["reason"] = (
                    f"Chop: |z|={abs_z:.2f} < {cut:.1f}"
                )

        # ── Trend states ───────────────────────────────────────────────────
        elif hmm_state == 0:
            result["signal"] = "BUY"
            result["reason"] = (
                f"Bull BUY: z={z:.2f}>{req['min_z_score']:.1f} "
                f"({consec} bars, {req['confidence_level']})"
            )
        elif hmm_state == 1:
            result["signal"] = "SELL"
            result["reason"] = (
                f"Bear SELL: z={z:.2f}<-{req['min_z_score']:.1f} "
                f"({consec} bars, {req['confidence_level']})"
            )

        if req.get("warning"):
            result["reason"] += f" [{req['warning']}]"

        # Audit assertions — log ERROR if regime/signal mismatch (should never fire)
        _sig = result["signal"]
        if _sig == "BUY"  and hmm_state != 0: logger.error("[BUG] evaluate_signal: BUY in state %d",  hmm_state)
        if _sig == "SELL" and hmm_state != 1: logger.error("[BUG] evaluate_signal: SELL in state %d", hmm_state)
        if _sig in ("MR_BUY", "MR_SELL") and hmm_state not in (2, 3):
            logger.error("[BUG] evaluate_signal: MR signal %s in state %d", _sig, hmm_state)

        return result
