"""
Regime-Conditional Expected Value (RCEV) Scorer.

Replaces Z-Score signal gating — calculates expected dollar profit for each
potential trade based on historically-calibrated trade outcomes in similar
market conditions.

Calibration happens once after IS training via :meth:`RCEVScorer.calibrate`
(called from ``main.py --mode train``).  The calibration is saved to
``models/rcev_{tf}_{broker}.pkl`` and loaded by the live bridge on startup.

If no calibration file exists (first boot before training), the live bridge
falls back to the existing Z-Score :meth:`~src.signal_evaluator.SignalEvaluator.evaluate_signal`
path — no sys.exit, no crash.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default RCEV thresholds per TF ────────────────────────────────────────────
# These are used as fallbacks; override via --rcev_threshold CLI arg.
RCEV_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "H1":  0.50,   # $0.50 minimum expected profit
    "M15": 0.35,   # $0.35 for intraday frequency
    "M5":  0.20,   # $0.20 for scalping volume
}

# ── Tiered threshold floors per TF ────────────────────────────────────────────
# Even at maximum conviction the threshold is never lowered below these values.
RCEV_TIERED_FLOORS: Dict[str, float] = {
    "H1":  0.15,
    "M15": 0.12,
    "M5":  0.08,
}


def get_rcev_path(tf: str, broker: str = "headway_cent") -> Path:
    """Return the RCEV calibration pkl path for this TF/broker."""
    return Path(f"models/rcev_{tf.upper()}_{broker}.pkl")


class RCEVScorer:
    """Regime-Conditional Expected Value scorer.

    Calculates expected dollar profit for each potential trade using:
    - Per-regime XGBoost probability outputs
    - Historical trade outcomes in similar conditions (probability + volatility buckets)
    - Current market conditions (session quality, ATR/spread efficiency, regime stability)
    - TCN confidence multiplier (scales condition quality)

    Attributes:
        broker:           Broker profile key (e.g. 'headway_cent').
        calibration_data: Per-regime bucketed expected-P&L maps (populated by calibrate()).
        session_quality:  Hour (UTC) → quality multiplier [0.3, 1.0].
    """

    # UTC hour → session quality multiplier
    SESSION_QUALITY: Dict[int, float] = {
        0:  0.6,  1:  0.5,  2:  0.4,  3:  0.3,  4:  0.3,  5:  0.4,
        6:  0.6,  7:  0.8,  8:  1.0,  9:  1.0,  10: 0.9,  11: 0.8,
        12: 0.9,  13: 1.0,  14: 1.0,  15: 0.9,  16: 0.8,
        17: 0.7,  18: 0.6,  19: 0.5,  20: 0.6,  21: 0.7,
        22: 0.8,  23: 0.7,
    }

    def __init__(self, broker: str = "headway_cent") -> None:
        self.broker = broker
        self.calibration_data: Dict[int, Dict[str, Any]] = {}
        self.session_quality: Dict[int, float] = dict(self.SESSION_QUALITY)

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, trades_df: pd.DataFrame) -> None:
        """Calibrate expected-P&L lookup tables from IS backtest trade records.

        Args:
            trades_df: DataFrame produced by ``vectorized_backtest(..., return_trades=True)``.
                       Required columns: ``regime``, ``prob``, ``volatility``, ``pnl``.
                       Optional: ``gmm_vol_cluster``.

        Populates ``self.calibration_data`` — call :meth:`save` afterwards.
        """
        if trades_df is None or trades_df.empty:
            logger.warning("[RCEV] calibrate(): trades_df is empty — calibration skipped.")
            return

        required = {"regime", "prob", "volatility", "pnl"}
        missing = required - set(trades_df.columns)
        if missing:
            logger.warning("[RCEV] calibrate(): missing columns %s — calibration skipped.", missing)
            return

        prob_bins = [0.0, 0.55, 0.65, 0.75, 0.85, 1.0]
        vol_bins  = [0.0, 0.001, 0.003, 0.006, 0.01, 1.0]

        regime_map = {0: "Bull", 1: "Bear", 2: "Chop"}

        for regime_id in [0, 1, 2]:
            regime_name = regime_map[regime_id]
            # Chop regime includes both Chop_Low(2) and Chop_High(3)
            regime_mask = (
                (trades_df["regime"] == 2) | (trades_df["regime"] == 3)
                if regime_id == 2
                else (trades_df["regime"] == regime_id)
            )
            regime_trades = trades_df[regime_mask].copy()

            if len(regime_trades) == 0:
                logger.warning("[RCEV] No IS trades for %s — using zero default.", regime_name)
                self.calibration_data[regime_id] = {
                    "prob_bins":         prob_bins,
                    "vol_bins":          vol_bins,
                    "calibration_map":   {},
                    "default_expected":  0.0,
                    "n_trades":          0,
                }
                continue

            calibration_map: Dict[tuple, Dict[str, Any]] = {}
            for i in range(len(prob_bins) - 1):
                for j in range(len(vol_bins) - 1):
                    bucket = regime_trades[
                        (regime_trades["prob"]       >= prob_bins[i]) &
                        (regime_trades["prob"]        < prob_bins[i + 1]) &
                        (regime_trades["volatility"] >= vol_bins[j]) &
                        (regime_trades["volatility"]  < vol_bins[j + 1])
                    ]
                    if len(bucket) >= 3:
                        calibration_map[(i, j)] = {
                            "expected_pnl": float(bucket["pnl"].mean()),
                            "win_rate":     float((bucket["pnl"] > 0).mean()),
                            "n_trades":     len(bucket),
                        }

            default_expected = float(regime_trades["pnl"].mean())
            self.calibration_data[regime_id] = {
                "prob_bins":         prob_bins,
                "vol_bins":          vol_bins,
                "calibration_map":   calibration_map,
                "default_expected":  default_expected,
                "n_trades":          len(regime_trades),
            }
            logger.info(
                "[RCEV] %s: %d IS trades | %d calibrated buckets | "
                "default_expected=$%.4f",
                regime_name, len(regime_trades), len(calibration_map), default_expected,
            )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(
        self,
        regime_id:        int,
        xgb_prob:         float,
        volatility:       float,
        spread:           float,
        atr:              float,
        hour:             int,
        tcn_multiplier:   float = 1.0,
        regime_stability: float = 0.70,
    ) -> Dict[str, Any]:
        """Calculate expected dollar profit for a potential trade.

        Args:
            regime_id:        HMM state mapped to 0=Bull, 1=Bear, 2=Chop.
            xgb_prob:         Per-regime XGBoost probability from
                              :func:`~src.engine_xgb.predict_regime_proba`.
            volatility:       Current ATR normalised value.
            spread:           Current spread in price-fraction units.
            atr:              Raw ATR value (same units as spread for efficiency ratio).
            hour:             UTC hour (0–23) for session quality lookup.
            tcn_multiplier:   TCN confidence multiplier (0.7–1.3).  Values > 1.0
                              lower the condition_multiplier, raising the effective
                              RCEV threshold (low confidence → harder to trade).
            regime_stability: P(stay in current state) from HMM transmat diagonal.

        Returns:
            Dict with keys: ``expected_pnl``, ``base_expected``, ``win_rate``,
            ``session_factor``, ``efficiency_factor``, ``condition_multiplier``,
            ``tcn_multiplier``, ``should_trade`` (always None — caller decides).
        """
        if regime_id not in self.calibration_data:
            return {
                "expected_pnl":       0.0,
                "base_expected":      0.0,
                "win_rate":           0.0,
                "session_factor":     0.0,
                "efficiency_factor":  0.0,
                "condition_multiplier": 0.0,
                "tcn_multiplier":     float(tcn_multiplier),
                "should_trade":       None,
                "reason":             "No calibration data",
            }

        calib = self.calibration_data[regime_id]
        prob_bins = calib["prob_bins"]
        vol_bins  = calib["vol_bins"]

        prob_idx = int(np.digitize(xgb_prob,  prob_bins) - 1)
        vol_idx  = int(np.digitize(volatility, vol_bins)  - 1)
        prob_idx = max(0, min(prob_idx, len(prob_bins) - 2))
        vol_idx  = max(0, min(vol_idx,  len(vol_bins)  - 2))

        bucket_key = (prob_idx, vol_idx)
        bucket = calib["calibration_map"].get(bucket_key)

        if bucket:
            base_expected = bucket["expected_pnl"]
            base_win_rate = bucket["win_rate"]
        else:
            base_expected = calib["default_expected"]
            base_win_rate = 0.50

        # Condition adjustments
        session_factor = self.session_quality.get(int(hour) % 24, 0.50)

        # ATR / spread efficiency ratio, normalised to [0.5, 1.0]
        efficiency_ratio  = min(atr / max(spread, 1e-6), 5.0) / 5.0
        efficiency_factor = 0.5 + efficiency_ratio * 0.5

        # TCN: higher multiplier = lower confidence = lower condition quality
        tcn_factor = 1.0 / max(float(tcn_multiplier), 0.70)

        condition_multiplier = (
            session_factor *
            efficiency_factor *
            float(regime_stability) *
            tcn_factor
        )

        expected_pnl  = base_expected * condition_multiplier
        adjusted_wr   = min(base_win_rate * condition_multiplier, 0.95)

        return {
            "expected_pnl":       round(expected_pnl,         4),
            "base_expected":      round(base_expected,         4),
            "win_rate":           round(adjusted_wr,           3),
            "session_factor":     round(session_factor,        3),
            "efficiency_factor":  round(efficiency_factor,     3),
            "condition_multiplier": round(condition_multiplier, 3),
            "tcn_multiplier":     round(float(tcn_multiplier), 3),
            "should_trade":       None,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: Path | str) -> None:
        """Persist calibration data to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as fh:
            pickle.dump(
                {
                    "calibration_data": self.calibration_data,
                    "session_quality":  self.session_quality,
                    "broker":           self.broker,
                },
                fh,
            )
        logger.info("[RCEV] Calibration saved: %s", filepath)

    def load(self, filepath: Path | str) -> "RCEVScorer":
        """Load calibration data from disk.  Returns self for chaining."""
        filepath = Path(filepath)
        with open(filepath, "rb") as fh:
            data = pickle.load(fh)
        self.calibration_data = data["calibration_data"]
        self.session_quality  = data["session_quality"]
        self.broker           = data["broker"]
        total_trades = sum(v.get("n_trades", 0) for v in self.calibration_data.values())
        logger.info(
            "[RCEV] Calibration loaded: %s  (regimes=%d  total_IS_trades=%d)",
            filepath, len(self.calibration_data), total_trades,
        )
        return self

    @classmethod
    def from_file(cls, filepath: Path | str, broker: str = "headway_cent") -> Optional["RCEVScorer"]:
        """Load and return an RCEVScorer, or None if the file does not exist."""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.info("[RCEV] Calibration file not found (%s) — Z-Score fallback active.", filepath)
            return None
        instance = cls(broker=broker)
        instance.load(filepath)
        return instance
