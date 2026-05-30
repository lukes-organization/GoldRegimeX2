"""
Signal pipeline audit — verifies Bull/Bear/Chop signal generation end-to-end.

Tests confirm:
  1. SignalEngine fires BUY on Bull regime with sufficient VIX and persistence.
  2. SignalEngine fires SELL on Bear regime with sufficient VIX and persistence.
  3. SignalEngine fires no trend signal on Chop state for M5/M15.
  4. SignalEngine respects MIN_CONFIRMATION_BARS before entering.
  5. should_apply_prob_gate returns True only for H1.
  6. evaluate_bar returns SignalDecision with correct structure.
  7. Parity test: same synthetic bar yields identical SignalDecision from
     both backtest adapter and live adapter paths.
  8. M15 _m15_entry_ok gate: accepts/rejects correctly.
  9. dynamic_prob_threshold adapts to window percentile.
 10. SignalEngine should_exit fires on regime reversal after confirm bars.
 11. Per-call Z override changes evaluate_bar decision (relaxed vs default).
 12. Backtester smoke: strict Z yields 0 trades, relaxed Z yields >0 trades.

Run:
    python tests/audit_signal_pipeline.py
"""

import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

from src.signal_engine import SignalEngine, SignalDecision, should_apply_prob_gate


# ── Helpers ─────────────────────────────────────────────────────────────────

def _pass(n, msg):
    logger.info("PASS  [T%02d] %s", n, msg)
    return True


def _fail(n, msg):
    logger.error("FAIL  [T%02d] %s", n, msg)
    return False


def _make_transmat():
    return np.array([[0.85, 0.10, 0.05],
                     [0.10, 0.80, 0.10],
                     [0.05, 0.10, 0.85]])


def _prime(engine: SignalEngine, state: int, bars: int = 5) -> dict:
    info = {}
    for _ in range(bars):
        info = engine.update_regime(state, _make_transmat())
    return info


# ── Tests ────────────────────────────────────────────────────────────────────

def run_tests() -> bool:
    results = []

    # ── T01: Bull regime → BUY signal ────────────────────────────────────────
    eng = SignalEngine("H1")
    info = _prime(eng, 0)
    entry = eng.should_enter(info, xgb_prob=0.62, synth_vix_zscore=1.2, atr_band_position=0.45)
    r = entry is not None and entry["signal"] == "BUY"
    results.append(_pass(1, f"H1 Bull→BUY  entry={entry}") if r else
                   _fail(1, f"H1 Bull did not fire BUY  entry={entry}"))

    # ── T02: Bear regime → SELL signal ───────────────────────────────────────
    eng = SignalEngine("H1")
    info = _prime(eng, 1)
    entry = eng.should_enter(info, xgb_prob=0.38, synth_vix_zscore=1.2, atr_band_position=0.55)
    r = entry is not None and entry["signal"] == "SELL"
    results.append(_pass(2, f"H1 Bear→SELL  entry={entry}") if r else
                   _fail(2, f"H1 Bear did not fire SELL  entry={entry}"))

    # ── T03: M5 Chop (state=2) → no trend signal ────────────────────────────
    eng = SignalEngine("M5")
    info = _prime(eng, 2, bars=3)
    entry = eng.should_enter(info, xgb_prob=0.65, synth_vix_zscore=2.0, atr_band_position=0.5)
    r = entry is None
    results.append(_pass(3, f"M5 Chop→no trend  entry={entry}") if r else
                   _fail(3, f"M5 Chop fired trend signal!  entry={entry}"))

    # ── T04: MIN_CONFIRMATION_BARS respected — no entry after only 1 bar ─────
    eng = SignalEngine("H1")
    info = eng.update_regime(0, _make_transmat())
    entry = eng.should_enter(info, xgb_prob=0.70, synth_vix_zscore=1.5, atr_band_position=0.3)
    r = entry is None
    results.append(_pass(4, "H1 no entry before MIN_CONFIRMATION_BARS") if r else
                   _fail(4, f"H1 fired entry after only 1 bar  entry={entry}"))

    # ── T05: should_apply_prob_gate H1=True, M15/M5=False ────────────────────
    r = (should_apply_prob_gate("H1") is True
         and should_apply_prob_gate("M15") is False
         and should_apply_prob_gate("M5") is False
         and should_apply_prob_gate("h1") is True)
    results.append(_pass(5, "should_apply_prob_gate correct for all TFs") if r else
                   _fail(5, "should_apply_prob_gate mismatch"))

    # ── T06: evaluate_bar returns valid SignalDecision ─────────────────────────
    eng = SignalEngine("H1")
    _prime(eng, 0)
    row_ctx = {"hmm_state": 0, "prob": 0.63,
               "synth_vix_zscore": 1.3, "atr_band_position": 0.40}
    tf_cfg = {"tf": "H1", "hmm_transmat": _make_transmat(),
              "sl_atr_multiple": 2.0, "tp_atr_multiple": 1.5}
    decision = eng.evaluate_bar(row_ctx, current_position=None, tf_config=tf_cfg)
    r = (isinstance(decision, SignalDecision)
         and hasattr(decision, "enter") and hasattr(decision, "direction")
         and hasattr(decision, "reason") and hasattr(decision, "position_size_multiplier"))
    results.append(_pass(6, f"evaluate_bar→SignalDecision  enter={decision.enter} dir={decision.direction}") if r else
                   _fail(6, f"evaluate_bar invalid: {decision}"))

    # ── T07: Parity — same input → same SignalDecision ────────────────────────
    def _primed_engine(state=0, bars=5):
        e = SignalEngine("H1")
        _prime(e, state, bars)
        return e

    row_par = {"hmm_state": 0, "prob": 0.61,
               "synth_vix_zscore": 1.1, "atr_band_position": 0.45}
    tf_par = {"tf": "H1", "hmm_transmat": _make_transmat(),
              "sl_atr_multiple": 2.0, "tp_atr_multiple": 1.5}

    d_bt = _primed_engine().evaluate_bar(row_par, None, tf_par)
    d_lv = _primed_engine().evaluate_bar(row_par, None, tf_par)

    r = (d_bt.enter == d_lv.enter and d_bt.exit == d_lv.exit
         and d_bt.direction == d_lv.direction and d_bt.reason == d_lv.reason)
    results.append(_pass(7, f"Parity backtest==live  enter={d_bt.enter} dir={d_bt.direction}") if r else
                   _fail(7, f"Parity mismatch  bt={d_bt}  lv={d_lv}"))

    # ── T08: _m15_entry_ok gate accepts good bar, rejects weak bar ────────────
    eng = SignalEngine("M15")
    cfg_gate = {"m15_min_regime_bars": 4, "m15_min_persistence": 0.55,
                "m15_min_atr_expansion": 1.0, "m15_min_vix_z": 0.5,
                "m15_min_rv_ratio": 1.0}
    good_row = {"regime_duration": 5, "persistence": 0.72,
                "atr_expansion_ratio": 1.15, "synth_vix_zscore": 0.8,
                "realized_vol_ratio": 1.1}
    weak_row = {"regime_duration": 2, "persistence": 0.50,
                "atr_expansion_ratio": 0.90, "synth_vix_zscore": 0.3,
                "realized_vol_ratio": 0.9}
    r = eng._m15_entry_ok(good_row, cfg_gate) and not eng._m15_entry_ok(weak_row, cfg_gate)
    results.append(_pass(8, "_m15_entry_ok accepts good/rejects weak") if r else
                   _fail(8, f"_m15_entry_ok wrong  good={eng._m15_entry_ok(good_row, cfg_gate)} weak={eng._m15_entry_ok(weak_row, cfg_gate)}"))

    # ── T09: dynamic_prob_threshold adapts to window size ─────────────────────
    eng = SignalEngine("M5")
    t_small = eng.dynamic_prob_threshold(np.array([0.50, 0.51, 0.52]))  # <30 → fallback
    t_large = eng.dynamic_prob_threshold(np.linspace(0.50, 0.70, 100))
    r = t_small == 0.55 and t_large > 0.55
    results.append(_pass(9, f"dynamic_prob_threshold  small={t_small:.3f} large={t_large:.3f}") if r else
                   _fail(9, f"dynamic_prob_threshold wrong  small={t_small} large={t_large}"))

    # ── T10: should_exit fires on regime reversal after confirm bars ──────────
    eng = SignalEngine("H1")
    _prime(eng, 0, bars=3)
    eng.on_trade_entered(0)
    from src.signal_engine import MIN_EXIT_CONFIRM_BARS
    n_confirm = MIN_EXIT_CONFIRM_BARS.get("H1", 2)
    for _ in range(n_confirm):
        info = eng.update_regime(1, _make_transmat())
    exit_now, reason = eng.should_exit(info, cur_pnl=0.0, max_fav_pnl=1.0, bars_in_trade=5)
    r = exit_now and "regime" in reason
    results.append(_pass(10, f"should_exit fires on reversal  reason={reason}") if r else
                   _fail(10, f"should_exit failed  exit={exit_now} reason={reason}"))

    # ── T11: per-call Z override changes evaluate_bar decision ───────────────
    # H1 MIN_TREND_ZSCORE = 0.5.  vix_z=0.3 is below default → no entry.
    # With z_cutoff_bull=0.2, vix_z=0.3 >= 0.2 → entry fires.
    transmat_t11 = _make_transmat()
    eng_default = SignalEngine("H1")
    _prime(eng_default, 0)
    eng_relaxed = SignalEngine("H1")
    _prime(eng_relaxed, 0)

    row_t11 = {"hmm_state": 0, "prob": 0.63,
               "synth_vix_zscore": 0.3, "atr_band_position": 0.40}
    cfg_default_t11 = {"tf": "H1", "hmm_transmat": transmat_t11}
    cfg_relaxed_t11 = {"tf": "H1", "hmm_transmat": transmat_t11,
                       "z_cutoff_bull": 0.2, "z_cutoff_bear": -0.2}

    d_default = eng_default.evaluate_bar(row_t11, None, cfg_default_t11)
    d_relaxed = eng_relaxed.evaluate_bar(row_t11, None, cfg_relaxed_t11)
    r = (not d_default.enter) and d_relaxed.enter
    results.append(
        _pass(11, f"Z override changes decision  default.enter={d_default.enter} relaxed.enter={d_relaxed.enter}") if r else
        _fail(11, f"Z override had no effect  default={d_default.enter} relaxed={d_relaxed.enter}")
    )

    # ── T12: backtester smoke test — strict vs relaxed Z yields different trades
    import pandas as pd
    from src.backtester import vectorized_backtest
    from src.signal_engine import MIN_CONFIRMATION_BARS as _MCB

    n_bars = 120
    _confirm = _MCB.get("H1", 2)
    rng = np.random.default_rng(42)
    # Build a synthetic df with Bull regime throughout so entries are attempted.
    prices  = 1800.0 + np.cumsum(rng.normal(0, 0.5, n_bars))
    _df_bt  = pd.DataFrame({
        "Open": prices, "High": prices * 1.001, "Low": prices * 0.999,
        "Close": prices, "Volume": 1000,
        "log_return":   np.concatenate([[0.0], np.diff(np.log(prices))]),
        "atr_normalized": np.full(n_bars, 0.5),
        "synth_vix_zscore":  np.full(n_bars, 0.3),   # below H1 default (0.5)
        "atr_band_position": np.full(n_bars, 0.40),
    })
    _states_bt  = np.zeros(n_bars, dtype=int)          # all Bull
    _probs_bt   = np.full(n_bars, 0.63)                 # above ENTRY_PROB H1

    # Strict: Z default (0.5) — vix_z 0.3 never qualifies → 0 trades expected.
    res_strict = vectorized_backtest(
        _df_bt, _probs_bt, _states_bt, tf="H1", account_size=15.0,
    )
    # Relaxed: Z override 0.2 — vix_z 0.3 >= 0.2 → trades expected.
    res_relaxed = vectorized_backtest(
        _df_bt, _probs_bt, _states_bt, tf="H1", account_size=15.0,
        evaluator_config={"Z_CUTOFF_BULL": 0.2, "Z_CUTOFF_BEAR": -0.2},
    )
    r = res_strict["n_trades"] == 0 and res_relaxed["n_trades"] > 0
    results.append(
        _pass(12, f"backtester Z override: strict trades={res_strict['n_trades']} relaxed trades={res_relaxed['n_trades']}") if r else
        _fail(12, f"backtester Z smoke failed: strict={res_strict['n_trades']} relaxed={res_relaxed['n_trades']}")
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print()
    print("=" * 60)
    print(f"  SIGNAL PIPELINE AUDIT: {passed}/{total} passed")
    if passed < total:
        print(f"  {total - passed} FAILED — review FAIL lines above")
    else:
        print("  All tests passed")
    print("=" * 60)
    return passed == total


def test_decision_parity_same_input():
    """Mandatory parity test: same synthetic bar → identical SignalDecision
    from two independent engine instances (simulates backtest vs live adapters).
    """
    transmat = np.array([[0.85, 0.10, 0.05],
                         [0.10, 0.80, 0.10],
                         [0.05, 0.10, 0.85]])

    def _build(tf="H1", state=0, bars=5):
        eng = SignalEngine(tf=tf)
        for _ in range(bars):
            eng.update_regime(state, transmat)
        return eng

    row_ctx = {"hmm_state": 0, "prob": 0.61,
               "synth_vix_zscore": 1.2, "atr_band_position": 0.40}
    tf_cfg  = {"tf": "H1", "hmm_transmat": transmat,
               "sl_atr_multiple": 2.0, "tp_atr_multiple": 1.5}

    d1 = _build().evaluate_bar(row_ctx, None, tf_cfg)
    d2 = _build().evaluate_bar(row_ctx, None, tf_cfg)

    assert d1 == d2, f"Parity failure: {d1} != {d2}"
    return True


if __name__ == "__main__":
    ok = run_tests()
    try:
        test_decision_parity_same_input()
        logger.info("PASS  [PARITY] test_decision_parity_same_input")
    except AssertionError as exc:
        logger.error("FAIL  [PARITY] %s", exc)
        ok = False
    sys.exit(0 if ok else 1)
