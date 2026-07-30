"""Deterministic unit and invariance tests for adaptive_filter_lab.

Runs independently from Jupyter:

    python3 test_adaptive_filter_lab.py         # lightweight runner
    python3 -m pytest test_adaptive_filter_lab.py -q

No real market data, no numba, no notebook globals are required.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import adaptive_filter_lab as afl
from adaptive_filter_lab import (
    AdaptiveFilterConfig,
    add_adaptive_filter_features,
    all_true_mask,
    directional_probability_margin,
    ml_margin_mask,
    strategy_quality_mask,
    trend_pullback_quality_mask,
    volatility_expansion_quality_mask,
)

FILTER_COLUMNS = afl.FILTER_COLUMNS

TEST_CONFIG = AdaptiveFilterConfig(
    m5_window=200,
    m15_window=200,
    min_history_fraction=0.25,
)


def make_synthetic_frame(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic OHLCV + macro-ADX + ATR frame."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(
        "2021-01-01",
        periods=n,
        freq="5min",
    )
    close = 1800.0 + np.cumsum(rng.normal(0.0, 0.5, size=n))
    open_ = close + rng.normal(0.0, 0.2, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.3, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.3, size=n))
    volume = rng.integers(100, 5000, size=n).astype(float)
    atr14 = pd.Series(np.abs(rng.normal(1.0, 0.25, size=n)) + 0.1, index=index)
    atr100 = pd.Series(np.abs(rng.normal(1.0, 0.10, size=n)) + 0.5, index=index)
    m15_adx14 = pd.Series(
        np.clip(rng.normal(22.0, 8.0, size=n), 0.0, 60.0),
        index=index,
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr14": atr14.to_numpy(),
            "atr100": atr100.to_numpy(),
            "m15_adx14": m15_adx14.to_numpy(),
        },
        index=index,
    )


def make_mixed_regime_frame(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """Frame designed so trend and expansion masks differ.

    First half: controlled volatility (atr_ratio ~ 1.0), rising ADX.
    Second half: expanding volatility (atr_ratio > 1.1), large candle bodies.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="5min")
    half = n // 2
    atr14 = np.concatenate(
        [
            np.full(half, 1.0) + rng.normal(0.0, 0.02, size=half),
            np.full(n - half, 1.6) + rng.normal(0.0, 0.05, size=n - half),
        ]
    )
    atr100 = np.full(n, 1.0)
    adx = np.concatenate(
        [
            np.linspace(20.0, 45.0, half),
            np.linspace(45.0, 20.0, n - half),
        ]
    )
    close = 1800.0 + np.cumsum(rng.normal(0.0, 0.4, size=n))
    body = np.concatenate(
        [
            rng.normal(0.0, 0.05, size=half),
            np.sign(rng.normal(size=n - half)) * (atr14[half:] * 0.6),
        ]
    )
    open_ = close - body
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    volume = np.concatenate(
        [
            rng.integers(3000, 5000, size=half).astype(float),
            rng.integers(3500, 5000, size=n - half).astype(float),
        ]
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr14": atr14,
            "atr100": atr100,
            "m15_adx14": adx,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Causality: a future change must not modify prior filter features
# ---------------------------------------------------------------------------
def test_future_change_does_not_modify_past_features():
    original = make_synthetic_frame()
    modified = original.copy()
    modified.iloc[-1, modified.columns.get_loc("volume")] *= 1000
    left = add_adaptive_filter_features(original, "M5", TEST_CONFIG)
    right = add_adaptive_filter_features(modified, "M5", TEST_CONFIG)
    pd.testing.assert_frame_equal(
        left.iloc[:-1][FILTER_COLUMNS],
        right.iloc[:-1][FILTER_COLUMNS],
    )


# ---------------------------------------------------------------------------
# Disabled filter is identity (all True)
# ---------------------------------------------------------------------------
def test_disabled_filter_is_all_true():
    frame = add_adaptive_filter_features(
        make_synthetic_frame(), "M5", TEST_CONFIG
    )
    config = AdaptiveFilterConfig(enabled=False)
    mask = trend_pullback_quality_mask(frame, config)
    assert mask.all()
    mask_x = volatility_expansion_quality_mask(frame, config)
    assert mask_x.all()


# ---------------------------------------------------------------------------
# Mask monotonicity: strict accepts no more than loose
# ---------------------------------------------------------------------------
def test_trend_mask_monotonicity():
    frame = add_adaptive_filter_features(
        make_synthetic_frame(), "M5", TEST_CONFIG
    )
    loose = _apply_preset(frame, afl.TREND_FILTER_PRESETS["loose"], TEST_CONFIG)
    loose_mask = trend_pullback_quality_mask(frame, loose)
    strict = _apply_preset(frame, afl.TREND_FILTER_PRESETS["strict"], TEST_CONFIG)
    strict_mask = trend_pullback_quality_mask(frame, strict)
    moderate = _apply_preset(frame, afl.TREND_FILTER_PRESETS["moderate"], TEST_CONFIG)
    moderate_mask = trend_pullback_quality_mask(frame, moderate)
    balanced = _apply_preset(frame, afl.TREND_FILTER_PRESETS["balanced"], TEST_CONFIG)
    balanced_mask = trend_pullback_quality_mask(frame, balanced)
    assert strict_mask.sum() <= loose_mask.sum()
    # moderate sits between loose and balanced (looser than balanced, tighter than loose)
    assert loose_mask.sum() >= moderate_mask.sum() >= balanced_mask.sum() >= strict_mask.sum()


def test_expansion_mask_monotonicity():
    frame = add_adaptive_filter_features(
        make_mixed_regime_frame(), "M5", TEST_CONFIG
    )
    loose = _apply_preset(
        frame, afl.EXPANSION_FILTER_PRESETS["loose"], TEST_CONFIG, expansion=True
    )
    loose_mask = volatility_expansion_quality_mask(frame, loose)
    strict = _apply_preset(
        frame, afl.EXPANSION_FILTER_PRESETS["strict"], TEST_CONFIG, expansion=True
    )
    strict_mask = volatility_expansion_quality_mask(frame, strict)
    moderate = _apply_preset(
        frame, afl.EXPANSION_FILTER_PRESETS["moderate"], TEST_CONFIG, expansion=True
    )
    moderate_mask = volatility_expansion_quality_mask(frame, moderate)
    balanced = _apply_preset(
        frame, afl.EXPANSION_FILTER_PRESETS["balanced"], TEST_CONFIG, expansion=True
    )
    balanced_mask = volatility_expansion_quality_mask(frame, balanced)
    assert strict_mask.sum() <= loose_mask.sum()
    assert loose_mask.sum() >= moderate_mask.sum() >= balanced_mask.sum() >= strict_mask.sum()


def _apply_preset(frame, preset, base, expansion=False):
    """Rebuild a config keeping the test window but preset thresholds."""
    from dataclasses import replace

    if expansion:
        return replace(
            base,
            expansion_min_volume_percentile=preset.expansion_min_volume_percentile,
            expansion_min_adx_percentile=preset.expansion_min_adx_percentile,
            expansion_min_atr_ratio=preset.expansion_min_atr_ratio,
            expansion_max_atr_ratio=preset.expansion_max_atr_ratio,
            expansion_min_body_atr_ratio=preset.expansion_min_body_atr_ratio,
        )
    return replace(
        base,
        min_volume_percentile=preset.min_volume_percentile,
        min_adx_percentile=preset.min_adx_percentile,
        min_atr_ratio=preset.min_atr_ratio,
        max_atr_ratio=preset.max_atr_ratio,
        min_adx_change_3=preset.min_adx_change_3,
    )


# ---------------------------------------------------------------------------
# Index preservation
# ---------------------------------------------------------------------------
def test_index_preservation():
    frame = add_adaptive_filter_features(
        make_synthetic_frame(), "M5", TEST_CONFIG
    )
    mask = trend_pullback_quality_mask(frame, TEST_CONFIG)
    assert mask.index.equals(frame.index)


# ---------------------------------------------------------------------------
# No missing boolean output
# ---------------------------------------------------------------------------
def test_boolean_no_nan_output():
    frame = add_adaptive_filter_features(
        make_synthetic_frame(), "M5", TEST_CONFIG
    )
    for mask in (
        trend_pullback_quality_mask(frame, TEST_CONFIG),
        volatility_expansion_quality_mask(frame, TEST_CONFIG),
        strategy_quality_mask(frame, "trend_pullback", TEST_CONFIG),
    ):
        assert mask.dtype == bool
        assert not mask.isna().any()


# ---------------------------------------------------------------------------
# Strategy separation: trend != expansion on a mixed-regime dataset
# ---------------------------------------------------------------------------
def test_strategy_separation():
    frame = add_adaptive_filter_features(
        make_mixed_regime_frame(), "M5", TEST_CONFIG
    )
    trend = trend_pullback_quality_mask(frame, TEST_CONFIG)
    expansion = volatility_expansion_quality_mask(frame, TEST_CONFIG)
    assert not trend.equals(expansion)


# ---------------------------------------------------------------------------
# All-day requirement: adaptive functions embed no clock/session logic
# ---------------------------------------------------------------------------
def test_adaptive_functions_have_no_clock_conditions():
    forbidden = ("index.hour", "session", "london", "new york", "asia")
    funcs = [
        afl.add_adaptive_filter_features,
        afl.rolling_percentile,
        afl.volume_mask,
        afl.atr_band_mask,
        afl.adx_quality_mask,
        afl.candle_quality_mask,
        afl.trend_pullback_quality_mask,
        afl.volatility_expansion_quality_mask,
        afl.strategy_quality_mask,
    ]
    for func in funcs:
        source = inspect.getsource(func).lower()
        for token in forbidden:
            assert token not in source, (
                f"{func.__name__} must not reference '{token}'"
            )


# ---------------------------------------------------------------------------
# ML-margin direction: long uses up prob, short uses down prob
# ---------------------------------------------------------------------------
def test_ml_margin_direction():
    signals = np.array([1, -1, 0], dtype=np.int8)
    # columns: [down, flat, up]
    probs = np.array(
        [
            [0.10, 0.20, 0.70],  # long -> should select up (0.70)
            [0.65, 0.20, 0.15],  # short -> should select down (0.65)
            [0.33, 0.34, 0.33],  # flat -> NaN
        ]
    )
    frame = directional_probability_margin(signals, probs)
    assert frame["selected_probability"].iloc[0] == 0.70
    assert frame["selected_probability"].iloc[1] == 0.65
    assert np.isnan(frame["selected_probability"].iloc[2])
    # margin = selected - max(competing)
    assert abs(frame["directional_margin"].iloc[0] - (0.70 - 0.20)) < 1e-9
    assert abs(frame["directional_margin"].iloc[1] - (0.65 - 0.20)) < 1e-9
    mask = ml_margin_mask(signals, probs, minimum_probability=0.5, minimum_margin=0.1)
    assert bool(mask[0]) and bool(mask[1]) and not bool(mask[2])


# ---------------------------------------------------------------------------
# Rolling percentile is causal (equivalent recompute on a prefix matches)
# ---------------------------------------------------------------------------
def test_rolling_percentile_prefix_stability():
    frame = make_synthetic_frame(n=400)
    full = add_adaptive_filter_features(frame, "M5", TEST_CONFIG)
    prefix = add_adaptive_filter_features(frame.iloc[:300], "M5", TEST_CONFIG)
    pd.testing.assert_series_equal(
        full["filter_adx_percentile"].iloc[:300],
        prefix["filter_adx_percentile"],
    )


# ---------------------------------------------------------------------------
# build_experiment_mask identities
# ---------------------------------------------------------------------------
def test_experiment_mask_identities():
    frame = add_adaptive_filter_features(
        make_synthetic_frame(), "M5", TEST_CONFIG
    )
    session_mask = pd.Series(
        (frame.index.hour >= 13) & (frame.index.hour < 16),
        index=frame.index,
    )
    adaptive = strategy_quality_mask(frame, "trend_pullback", TEST_CONFIG)
    vol = afl.volume_mask(frame, TEST_CONFIG.min_volume_percentile)
    atrm = afl.atr_band_mask(frame, TEST_CONFIG.min_atr_ratio, TEST_CONFIG.max_atr_ratio)
    adxm = afl.adx_quality_mask(frame, TEST_CONFIG.min_adx_percentile, TEST_CONFIG.min_adx_change_3)
    baseline = afl.build_experiment_mask(
        frame, "baseline_all_day", session_mask, adaptive, vol, atrm, adxm
    )
    assert baseline.all()
    session_only = afl.build_experiment_mask(
        frame, "session_only", session_mask, adaptive, vol, atrm, adxm
    )
    assert session_only.equals(session_mask.fillna(False))
    adaptive_only = afl.build_experiment_mask(
        frame, "adaptive_quality", session_mask, adaptive, vol, atrm, adxm
    )
    assert adaptive_only.equals(adaptive.fillna(False))


# ---------------------------------------------------------------------------
# Chronological split honours the ~80/20 intent
# ---------------------------------------------------------------------------
def test_chronological_split_ratio():
    frame = make_synthetic_frame(n=1000)
    train, oos, split_time = afl.chronological_split(frame, holdout_fraction=0.20)
    assert len(train) == 800
    assert len(oos) == 200
    assert train.index.max() == split_time
    assert oos.index.min() > split_time


def _collect_tests():
    return [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]


def main() -> int:
    failures = []
    tests = _collect_tests()
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"{len(tests) - len(failures)}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
