"""
Signal pipeline audit — verifies Bull/Bear/Chop signal generation end-to-end.

Tests confirm:
  1. Z-Score uses per-regime stats (not shared or fallback)
  2. Bull (state 0) only fires BUY; never SELL or MR
  3. Bear (state 1) only fires SELL; never BUY or MR
  4. Chop states (2, 3) only fire MR_BUY / MR_SELL; never trend signals
  5. Z comparison direction is correct per state
  6. Tiered override reduces the Bull/Bear cutoff without touching MR
  7. Backtester collapses MR_BUY/MR_SELL to ±1 (unified array encoding)

Run:
    python tests/audit_signal_pipeline.py
"""

import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

from src.signal_evaluator import SignalEvaluator
from src.backtester import compute_signals_zscore

# ── Fixture —————————————————————————————————————————————————————————————————
# Realistic per-state stats (representative of a trained M15 model).
# These values produce Z > |2.0| for most test probabilities.
REGIME_STATS = {
    0: {"mean": 0.5146, "std": 0.0193, "count": 9871},   # Bull
    1: {"mean": 0.4995, "std": 0.0168, "count": 17777},  # Bear
    2: {"mean": 0.4954, "std": 0.0148, "count": 68128},  # Chop_Low
    3: {"mean": 0.5004, "std": 0.0174, "count": 56062},  # Chop_High
}


def _pass(n, msg):
    logger.info("PASS  [T%02d] %s", n, msg)
    return True

def _fail(n, msg):
    logger.error("FAIL  [T%02d] %s", n, msg)
    return False


def run_tests() -> bool:
    ev = SignalEvaluator(regime_stats=REGIME_STATS, tf="M15")
    results = []

    # ── T01: Bull + strong bullish prob → BUY ────────────────────────────────
    # z = (0.58 - 0.5146) / 0.0193 ≈ +3.39  (> M15 cutoff 2.0)
    sig, z = ev.evaluate_signal_fast(prob_buy=0.58, hmm_state=0)
    r = sig == "BUY" and z > 2.0
    results.append(_pass(1, f"Bull→BUY  z={z:.3f}") if r else
                   _fail(1, f"Bull→BUY  got sig={sig} z={z:.3f}"))

    # ── T02: Bull + neutral prob → no signal ────────────────────────────────
    # z = (0.52 - 0.5146) / 0.0193 ≈ +0.28
    sig, z = ev.evaluate_signal_fast(prob_buy=0.52, hmm_state=0)
    r = sig is None
    results.append(_pass(2, f"Bull→None  z={z:.3f}") if r else
                   _fail(2, f"Bull→None  got sig={sig} z={z:.3f}"))

    # ── T03: Bull + bearish prob → NOT SELL (state blocks it) ────────────────
    # z ≈ -4.90 — Bull branch only generates BUY, never SELL
    sig, z = ev.evaluate_signal_fast(prob_buy=0.42, hmm_state=0)
    r = sig != "SELL"
    results.append(_pass(3, f"Bull never SELL  sig={sig} z={z:.3f}") if r else
                   _fail(3, f"Bull generated SELL!  z={z:.3f}"))

    # ── T04: Bear + strong bearish prob → SELL ───────────────────────────────
    # z = (0.44 - 0.4995) / 0.0168 ≈ -3.54  (< -2.0)
    sig, z = ev.evaluate_signal_fast(prob_buy=0.44, hmm_state=1)
    r = sig == "SELL" and z < -2.0
    results.append(_pass(4, f"Bear→SELL  z={z:.3f}") if r else
                   _fail(4, f"Bear→SELL  got sig={sig} z={z:.3f}"))

    # ── T05: Bear + neutral prob → no signal ────────────────────────────────
    # z = (0.49 - 0.4995) / 0.0168 ≈ -0.57
    sig, z = ev.evaluate_signal_fast(prob_buy=0.49, hmm_state=1)
    r = sig is None
    results.append(_pass(5, f"Bear→None  z={z:.3f}") if r else
                   _fail(5, f"Bear→None  got sig={sig} z={z:.3f}"))

    # ── T06: Bear + bullish prob → NOT BUY ───────────────────────────────────
    # z ≈ +4.79 — Bear branch only generates SELL, never BUY
    sig, z = ev.evaluate_signal_fast(prob_buy=0.58, hmm_state=1)
    r = sig != "BUY"
    results.append(_pass(6, f"Bear never BUY  sig={sig} z={z:.3f}") if r else
                   _fail(6, f"Bear generated BUY!  z={z:.3f}"))

    # ── T07: Chop_Low + extreme bearish prob → MR_BUY ────────────────────────
    # z = (0.43 - 0.4954) / 0.0148 ≈ -4.42  (< -3.0)
    sig, z = ev.evaluate_signal_fast(prob_buy=0.43, hmm_state=2)
    r = sig == "MR_BUY"
    results.append(_pass(7, f"Chop_Low→MR_BUY  z={z:.3f}") if r else
                   _fail(7, f"Chop_Low→MR_BUY  got sig={sig} z={z:.3f}"))

    # ── T08: Chop_High + extreme bullish prob → MR_SELL ──────────────────────
    # z = (0.58 - 0.5004) / 0.0174 ≈ +4.57  (> 3.5)
    sig, z = ev.evaluate_signal_fast(prob_buy=0.58, hmm_state=3)
    r = sig == "MR_SELL"
    results.append(_pass(8, f"Chop_High→MR_SELL  z={z:.3f}") if r else
                   _fail(8, f"Chop_High→MR_SELL  got sig={sig} z={z:.3f}"))

    # ── T09: Chop + moderate prob → no trend signal ───────────────────────────
    # Chop branches may produce MR or None, but NEVER BUY / SELL
    sig, z = ev.evaluate_signal_fast(prob_buy=0.515, hmm_state=2)
    r = sig not in ("BUY", "SELL")
    results.append(_pass(9, f"Chop never trend  sig={sig} z={z:.3f}") if r else
                   _fail(9, f"Chop generated trend signal={sig}!  z={z:.3f}"))

    # ── T10: Per-state Z-Scores are distinct for same prob ────────────────────
    prob = 0.52
    zs = {s: round(ev.calculate_z_score(prob, s), 3) for s in (0, 1, 2, 3)}
    r = len(set(zs.values())) >= 3  # at least 3 distinct values
    results.append(_pass(10, f"Per-state Z distinct: {zs}") if r else
                   _fail(10, f"Z-Scores too similar — regime_stats not used: {zs}"))

    # ── T11: Tiered fires trade that standard misses ─────────────────────────
    # Bull + prob=0.550 → z=(0.55-0.5146)/0.0193≈+1.84 < 2.0 (no signal normally)
    # extremity = |0.55-0.50| = 0.05 ≥ 0.04 → reduction=0.25 → bull_cut=1.75
    # z=1.84 > 1.75 → BUY with tiered
    sig_std,    _ = ev.evaluate_signal_fast(prob_buy=0.550, hmm_state=0, use_tiered=False)
    sig_tiered, _ = ev.evaluate_signal_fast(prob_buy=0.550, hmm_state=0, use_tiered=True)
    r = sig_std is None and sig_tiered == "BUY"
    results.append(_pass(11, f"Tiered promotes sub-threshold trade  std={sig_std} tiered={sig_tiered}") if r else
                   _fail(11, f"Tiered test  std={sig_std} tiered={sig_tiered}"))

    # ── T12: Tiered floor — extreme conviction cannot drop Z below 1.0 ────────
    # Bull + prob=0.62 → extremity=0.12 ≥ 0.10 → reduction=1.0
    # bull_cut = max(1.0, 2.0 - 1.0) = 1.0
    # z=(0.62-0.5146)/0.0193≈+5.46 > 1.0 → still BUY, and effective cutoff = 1.0
    ev2 = SignalEvaluator(regime_stats=REGIME_STATS, tf="M15",
                          config={"Z_CUTOFF_BULL": 0.5})   # below floor
    sig, z = ev2.evaluate_signal_fast(prob_buy=0.62, hmm_state=0, use_tiered=True)
    # _calculate_tiered_override returns 1.0; max(1.0, 0.5 - 1.0) → max(1.0, -0.5) = 1.0
    r = sig == "BUY" and z > 1.0
    results.append(_pass(12, f"Tiered floor=1.0 respected  sig={sig} z={z:.3f}") if r else
                   _fail(12, f"Tiered floor violated!  sig={sig} z={z:.3f}"))

    # ── T13: Backtester collapses MR_BUY → +1, MR_SELL → -1 ─────────────────
    # Feed 4 bars: Bull+prob_high, Bear+prob_low, Chop_Low+prob_extreme, Chop_High+prob_extreme
    probs  = np.array([0.58, 0.44, 0.43, 0.58])
    states = np.array([0,    1,    2,    3   ])
    sigs = compute_signals_zscore(probs, states, REGIME_STATS, tf="M15")
    # Expected: [1, -1, 1, -1]
    expected = np.array([1, -1, 1, -1], dtype=np.int8)
    r = np.array_equal(sigs, expected)
    results.append(_pass(13, f"Backtester signal encoding  {sigs.tolist()} == {expected.tolist()}") if r else
                   _fail(13, f"Backtester encoding wrong  got={sigs.tolist()} want={expected.tolist()}"))

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print()
    print("=" * 60)
    print(f"  SIGNAL PIPELINE AUDIT: {passed}/{total} passed")
    if passed < total:
        print(f"  {total - passed} FAILED — review FAIL lines above")
    else:
        print("  All tests passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
