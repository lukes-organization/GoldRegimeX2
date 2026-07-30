"""Adaptive, clock-free market-quality filter lab (reusable calculations).

This module contains ONLY pure, side-effect-free candidate-eligibility
calculations for the adaptive-filter experiment. It performs:

  * No model training.
  * No notebook globals.
  * No file writing.
  * No lookahead (all rolling statistics are causal / trailing-only).
  * No fixed clock-hour / calendar conditions of any kind.

The market-quality masks answer a single question: given only the current and
prior bars, does market activity, volatility and trend quality meet the
configured conditions? The clock-based comparison mask is computed in the
notebook and only *combined* here via ``build_experiment_mask``.

All masks return boolean pandas Series aligned to the input index with no NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdaptiveFilterConfig:
    enabled: bool = True
    # Approximately 20 days of bars.
    m5_window: int = 288 * 20
    m15_window: int = 96 * 20
    min_history_fraction: float = 0.25
    # Trend-pullback defaults.
    min_volume_percentile: float = 0.50
    min_adx_percentile: float = 0.60
    min_atr_ratio: float = 0.85
    max_atr_ratio: float = 1.35
    min_adx_change_3: float = 0.0
    # Volatility-expansion defaults.
    expansion_min_volume_percentile: float = 0.60
    expansion_min_adx_percentile: float = 0.50
    expansion_min_atr_ratio: float = 1.10
    expansion_max_atr_ratio: float = 2.00
    expansion_min_body_atr_ratio: float = 0.20


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------
def _resolve_column(
    df: pd.DataFrame,
    candidates: Tuple[str, ...],
    semantic_name: str,
) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(
        f"Unable to locate {semantic_name}. "
        f"Tried columns: {candidates}"
    )


# ---------------------------------------------------------------------------
# Causal rolling percentile
# ---------------------------------------------------------------------------
def rolling_percentile(
    series: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Trailing-only percentile rank in [0, 1].

    Uses only the current and prior bars inside a trailing window. It must
    never use centered windows, backward fills or full-sample percentiles.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric.rolling(
        window=window,
        min_periods=min_periods,
    ).rank(
        method="average",
        pct=True,
    )
    return result


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
FILTER_COLUMNS = [
    "filter_atr_ratio",
    "filter_adx_percentile",
    "filter_volume_percentile",
    "filter_adx_change_3",
    "filter_adx_rising",
    "filter_body_atr_ratio",
]


def add_adaptive_filter_features(
    df: pd.DataFrame,
    timeframe: str,
    config: AdaptiveFilterConfig,
) -> pd.DataFrame:
    out = df.copy()
    timeframe = timeframe.upper()
    if timeframe == "M5":
        window = int(config.m5_window)
    elif timeframe == "M15":
        window = int(config.m15_window)
    else:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )
    min_periods = max(
        100,
        int(window * config.min_history_fraction),
    )
    adx_col = _resolve_column(
        out,
        ("m15_adx14", "macro_adx14"),
        "macro ADX",
    )
    volume_col = _resolve_column(
        out,
        ("volume", "Volume"),
        "tick volume",
    )
    open_col = _resolve_column(
        out,
        ("open", "Open"),
        "open price",
    )
    close_col = _resolve_column(
        out,
        ("close", "Close"),
        "close price",
    )
    for required in ("atr14", "atr100"):
        if required not in out.columns:
            raise KeyError(
                f"Required filter feature missing: {required}"
            )
    atr14 = pd.to_numeric(
        out["atr14"],
        errors="coerce",
    )
    atr100 = pd.to_numeric(
        out["atr100"],
        errors="coerce",
    ).replace(0.0, np.nan)
    adx = pd.to_numeric(
        out[adx_col],
        errors="coerce",
    )
    volume = pd.to_numeric(
        out[volume_col],
        errors="coerce",
    )
    open_price = pd.to_numeric(
        out[open_col],
        errors="coerce",
    )
    close_price = pd.to_numeric(
        out[close_col],
        errors="coerce",
    )
    out["filter_atr_ratio"] = atr14 / atr100
    out["filter_adx_percentile"] = rolling_percentile(
        adx,
        window=window,
        min_periods=min_periods,
    )
    out["filter_volume_percentile"] = rolling_percentile(
        volume,
        window=window,
        min_periods=min_periods,
    )
    out["filter_adx_change_3"] = adx.diff(3)
    out["filter_adx_rising"] = (
        out["filter_adx_change_3"] > 0.0
    )
    out["filter_body_atr_ratio"] = (
        (close_price - open_price).abs()
        / atr14.replace(0.0, np.nan)
    )
    return out


# ---------------------------------------------------------------------------
# Component masks (independent, so attribution stays possible)
# ---------------------------------------------------------------------------
def all_true_mask(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )


def volume_mask(
    df: pd.DataFrame,
    minimum_percentile: float,
) -> pd.Series:
    return (
        df["filter_volume_percentile"]
        .ge(float(minimum_percentile))
        .fillna(False)
    )


def atr_band_mask(
    df: pd.DataFrame,
    minimum_ratio: float,
    maximum_ratio: float,
) -> pd.Series:
    return (
        df["filter_atr_ratio"]
        .between(
            float(minimum_ratio),
            float(maximum_ratio),
            inclusive="both",
        )
        .fillna(False)
    )


def adx_quality_mask(
    df: pd.DataFrame,
    minimum_percentile: float,
    minimum_change_3: float = 0.0,
) -> pd.Series:
    return (
        df["filter_adx_percentile"]
        .ge(float(minimum_percentile))
        & df["filter_adx_change_3"]
        .gt(float(minimum_change_3))
    ).fillna(False)


def candle_quality_mask(
    df: pd.DataFrame,
    minimum_body_atr_ratio: float,
) -> pd.Series:
    return (
        df["filter_body_atr_ratio"]
        .ge(float(minimum_body_atr_ratio))
        .fillna(False)
    )


# ---------------------------------------------------------------------------
# Strategy-specific composite masks
# ---------------------------------------------------------------------------
def trend_pullback_quality_mask(
    df: pd.DataFrame,
    config: AdaptiveFilterConfig,
) -> pd.Series:
    if not config.enabled:
        return all_true_mask(df)
    return (
        volume_mask(
            df,
            config.min_volume_percentile,
        )
        & atr_band_mask(
            df,
            config.min_atr_ratio,
            config.max_atr_ratio,
        )
        & adx_quality_mask(
            df,
            config.min_adx_percentile,
            config.min_adx_change_3,
        )
    ).fillna(False)


def volatility_expansion_quality_mask(
    df: pd.DataFrame,
    config: AdaptiveFilterConfig,
) -> pd.Series:
    if not config.enabled:
        return all_true_mask(df)
    return (
        volume_mask(
            df,
            config.expansion_min_volume_percentile,
        )
        & atr_band_mask(
            df,
            config.expansion_min_atr_ratio,
            config.expansion_max_atr_ratio,
        )
        & adx_quality_mask(
            df,
            config.expansion_min_adx_percentile,
            minimum_change_3=-np.inf,
        )
        & candle_quality_mask(
            df,
            config.expansion_min_body_atr_ratio,
        )
    ).fillna(False)


def strategy_quality_mask(
    df: pd.DataFrame,
    strategy_name: str,
    config: AdaptiveFilterConfig,
) -> pd.Series:
    if strategy_name == "trend_pullback":
        return trend_pullback_quality_mask(
            df,
            config,
        )
    if strategy_name == "volatility_expansion":
        return volatility_expansion_quality_mask(
            df,
            config,
        )
    raise ValueError(
        f"Unsupported strategy: {strategy_name}"
    )


# ---------------------------------------------------------------------------
# Experiment selection (combiner only; receives precomputed masks)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    "baseline_all_day",
    "session_only",
    "volume_only",
    "atr_only",
    "adx_only",
    "volume_atr",
    "adaptive_quality",
    "session_plus_adaptive",
]


def build_experiment_mask(
    df: pd.DataFrame,
    experiment_name: str,
    session_mask: pd.Series,
    adaptive_mask: pd.Series,
    volume_only_mask: pd.Series,
    atr_only_mask: pd.Series,
    adx_only_mask: pd.Series,
) -> pd.Series:
    all_day = all_true_mask(df)
    masks = {
        "baseline_all_day": all_day,
        "session_only": session_mask,
        "volume_only": volume_only_mask,
        "atr_only": atr_only_mask,
        "adx_only": adx_only_mask,
        "volume_atr": (
            volume_only_mask
            & atr_only_mask
        ),
        "adaptive_quality": adaptive_mask,
        "session_plus_adaptive": (
            session_mask
            & adaptive_mask
        ),
    }
    if experiment_name not in masks:
        raise KeyError(
            f"Unknown experiment: {experiment_name}"
        )
    result = masks[experiment_name]
    if not result.index.equals(df.index):
        raise AssertionError(
            "Experiment mask index is not aligned."
        )
    return result.fillna(False)


# ---------------------------------------------------------------------------
# Candidate funnel
# ---------------------------------------------------------------------------
def candidate_funnel(
    base_signals: pd.Series,
    session_mask: pd.Series,
    volume_mask_value: pd.Series,
    atr_mask_value: pd.Series,
    adx_mask_value: pd.Series,
    adaptive_mask: pd.Series,
    final_mask: pd.Series,
) -> Dict[str, int]:
    base = base_signals.ne(0)
    return {
        "base_routed_candidates": int(base.sum()),
        "after_session": int(
            (base & session_mask).sum()
        ),
        "after_volume": int(
            (base & volume_mask_value).sum()
        ),
        "after_atr": int(
            (base & atr_mask_value).sum()
        ),
        "after_adx": int(
            (base & adx_mask_value).sum()
        ),
        "after_adaptive_composite": int(
            (base & adaptive_mask).sum()
        ),
        "final_candidates": int(
            (base & final_mask).sum()
        ),
    }


# ---------------------------------------------------------------------------
# Hourly candidate reporting
# ---------------------------------------------------------------------------
def hourly_candidate_report(
    base_signals: pd.Series,
    filtered_signals: pd.Series,
) -> pd.DataFrame:
    if not base_signals.index.equals(
        filtered_signals.index
    ):
        raise AssertionError(
            "Signal indexes are not aligned."
        )
    frame = pd.DataFrame(
        {
            "hour": base_signals.index.hour,
            "base_candidate": (
                base_signals.ne(0).astype(int)
            ),
            "accepted_candidate": (
                filtered_signals.ne(0).astype(int)
            ),
        },
        index=base_signals.index,
    )
    report = (
        frame.groupby("hour")
        .agg(
            base_candidates=(
                "base_candidate",
                "sum",
            ),
            accepted_candidates=(
                "accepted_candidate",
                "sum",
            ),
        )
        .reset_index()
    )
    report["acceptance_rate"] = (
        report["accepted_candidates"]
        / report["base_candidates"].replace(
            0,
            np.nan,
        )
    )
    return report


# ---------------------------------------------------------------------------
# Hourly trade performance and concentration
# ---------------------------------------------------------------------------
def hourly_trade_report(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "entry_hour",
                "trades",
                "net_pnl_cents",
                "mean_pnl_cents",
                "win_rate",
            ]
        )
    required = {
        "entry_time",
        "pnl_cents",
    }
    missing = required.difference(
        trades.columns
    )
    if missing:
        raise KeyError(
            f"Trade log missing columns: {missing}"
        )
    out = trades.copy()
    out["entry_hour"] = pd.to_datetime(
        out["entry_time"]
    ).dt.hour
    report = (
        out.groupby("entry_hour")
        .agg(
            trades=("pnl_cents", "size"),
            net_pnl_cents=("pnl_cents", "sum"),
            mean_pnl_cents=("pnl_cents", "mean"),
            win_rate=(
                "pnl_cents",
                lambda values: float(
                    (values > 0).mean()
                ),
            ),
        )
        .reset_index()
    )
    return report


def hourly_concentration_metrics(
    trades: pd.DataFrame,
) -> dict:
    if trades.empty:
        return {
            "dominant_hour_trade_share": np.nan,
            "top_three_hours_trade_share": np.nan,
            "dominant_hour_pnl_share": np.nan,
        }
    hourly = hourly_trade_report(trades)
    total_trades = max(
        int(hourly["trades"].sum()),
        1,
    )
    total_positive_pnl = float(
        hourly["net_pnl_cents"]
        .clip(lower=0.0)
        .sum()
    )
    dominant_hour_trade_share = float(
        hourly["trades"].max()
        / total_trades
    )
    top_three_hours_trade_share = float(
        hourly.nlargest(
            3,
            "trades",
        )["trades"].sum()
        / total_trades
    )
    if total_positive_pnl > 0.0:
        dominant_hour_pnl_share = float(
            hourly["net_pnl_cents"]
            .clip(lower=0.0)
            .max()
            / total_positive_pnl
        )
    else:
        dominant_hour_pnl_share = np.nan
    return {
        "dominant_hour_trade_share": (
            dominant_hour_trade_share
        ),
        "top_three_hours_trade_share": (
            top_three_hours_trade_share
        ),
        "dominant_hour_pnl_share": (
            dominant_hour_pnl_share
        ),
    }


# ---------------------------------------------------------------------------
# Chronological IS / OOS split
# ---------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame,
    holdout_fraction: float = 0.20,
):
    """Chronological train / out-of-sample split.

    NOTE (deviation from the written spec): the spec text used
    ``int(len(ordered) - (1.0 - holdout_fraction))`` which collapses the
    holdout to a single row. The documented intent is an 80/20 split
    (``holdout_fraction=0.20``) with MIN_OOS_TRADES of 50 (M15) / 150 (M5),
    which is impossible with one OOS row. This uses the multiplicative form
    ``int(len(ordered) * (1.0 - holdout_fraction))`` to honour that intent.
    """
    ordered = df.sort_index().copy()
    split_position = int(
        len(ordered)
        * (1.0 - holdout_fraction)
    )
    split_position = max(
        1,
        min(
            split_position,
            len(ordered) - 1,
        ),
    )
    train = ordered.iloc[
        :split_position
    ].copy()
    oos = ordered.iloc[
        split_position:
    ].copy()
    split_time = train.index[-1]
    return train, oos, split_time


# ---------------------------------------------------------------------------
# Explorer-compatible ML directional-margin diagnostic (for later use)
# ---------------------------------------------------------------------------
def directional_probability_margin(
    raw_signals: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    signals = np.asarray(
        raw_signals,
        dtype=np.int8,
    )
    probs = np.asarray(
        probabilities,
        dtype=float,
    )
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(
            "Expected probability matrix with "
            "columns [down, flat, up]."
        )
    if len(signals) != len(probs):
        raise ValueError(
            "Signal and probability lengths differ."
        )
    prob_down = probs[:, 0]
    prob_flat = probs[:, 1]
    prob_up = probs[:, 2]
    selected_probability = np.where(
        signals == 1,
        prob_up,
        np.where(
            signals == -1,
            prob_down,
            np.nan,
        ),
    )
    competing_probability = np.where(
        signals == 1,
        np.maximum(
            prob_flat,
            prob_down,
        ),
        np.where(
            signals == -1,
            np.maximum(
                prob_flat,
                prob_up,
            ),
            np.nan,
        ),
    )
    margin = (
        selected_probability
        - competing_probability
    )
    return pd.DataFrame(
        {
            "selected_probability": (
                selected_probability
            ),
            "competing_probability": (
                competing_probability
            ),
            "directional_margin": margin,
        }
    )


def ml_margin_mask(
    raw_signals: np.ndarray,
    probabilities: np.ndarray,
    minimum_probability: float,
    minimum_margin: float,
) -> np.ndarray:
    margin_frame = directional_probability_margin(
        raw_signals,
        probabilities,
    )
    return (
        np.asarray(raw_signals) != 0
    ) & (
        margin_frame[
            "selected_probability"
        ].to_numpy()
        >= float(minimum_probability)
    ) & (
        margin_frame[
            "directional_margin"
        ].to_numpy()
        >= float(minimum_margin)
    )


ML_MARGIN_GRID = {
    "minimum_probability": [
        0.35,
        0.40,
        0.45,
        0.50,
    ],
    "minimum_margin": [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
    ],
}


# ---------------------------------------------------------------------------
# Initial filter presets (broad exploration; not final production values)
# ---------------------------------------------------------------------------
TREND_FILTER_PRESETS = {
    "loose": AdaptiveFilterConfig(
        min_volume_percentile=0.40,
        min_adx_percentile=0.50,
        min_atr_ratio=0.75,
        max_atr_ratio=1.50,
        min_adx_change_3=-np.inf,
    ),
    "moderate": AdaptiveFilterConfig(
        # Tuned midway between loose and balanced. The rising-ADX gate (min_adx_change_3)
        # is the single biggest retention killer, so it stays disabled here (like loose)
        # while the percentile/ATR band thresholds tighten partway toward balanced.
        min_volume_percentile=0.45,
        min_adx_percentile=0.55,
        min_atr_ratio=0.80,
        max_atr_ratio=1.42,
        min_adx_change_3=-np.inf,
    ),
    "balanced": AdaptiveFilterConfig(
        min_volume_percentile=0.50,
        min_adx_percentile=0.60,
        min_atr_ratio=0.85,
        max_atr_ratio=1.35,
        min_adx_change_3=0.0,
    ),
    "strict": AdaptiveFilterConfig(
        min_volume_percentile=0.65,
        min_adx_percentile=0.70,
        min_atr_ratio=0.90,
        max_atr_ratio=1.25,
        min_adx_change_3=0.0,
    ),
}

EXPANSION_FILTER_PRESETS = {
    "loose": AdaptiveFilterConfig(
        expansion_min_volume_percentile=0.45,
        expansion_min_adx_percentile=0.40,
        expansion_min_atr_ratio=1.00,
        expansion_max_atr_ratio=2.50,
        expansion_min_body_atr_ratio=0.10,
    ),
    "moderate": AdaptiveFilterConfig(
        # Midway between loose and balanced for the volatility-expansion strategy.
        expansion_min_volume_percentile=0.52,
        expansion_min_adx_percentile=0.45,
        expansion_min_atr_ratio=1.05,
        expansion_max_atr_ratio=2.25,
        expansion_min_body_atr_ratio=0.15,
    ),
    "balanced": AdaptiveFilterConfig(
        expansion_min_volume_percentile=0.60,
        expansion_min_adx_percentile=0.50,
        expansion_min_atr_ratio=1.10,
        expansion_max_atr_ratio=2.00,
        expansion_min_body_atr_ratio=0.20,
    ),
    "strict": AdaptiveFilterConfig(
        expansion_min_volume_percentile=0.70,
        expansion_min_adx_percentile=0.60,
        expansion_min_atr_ratio=1.20,
        expansion_max_atr_ratio=1.80,
        expansion_min_body_atr_ratio=0.30,
    ),
}


# ---------------------------------------------------------------------------
# Selection thresholds (diagnostics, not automatic production approval)
# ---------------------------------------------------------------------------
MAX_ACCEPTABLE_DD = 30.0

MIN_OOS_TRADES = {
    "M15": 50,
    "M5": 150,
}

# Tuning targets for IN-SAMPLE preset selection (IS/CPCV; avoids OOS peeking).
# A preset is a viable candidate only if it retains enough signals AND its in-sample
# drawdown stays within tolerance of the session-only baseline. The out-of-sample
# result is then REPORTED on the selected preset; OOS is never used to choose it.
MIN_IS_RETENTION = 0.25
DD_SESSION_TOLERANCE = 1.10

RESULT_COLUMNS = [
    "timeframe",
    "strategy_name",
    "exit_model",
    "filter_preset",
    "experiment_name",
    "segment",
    "split_time",
    "row_count",
    "base_candidate_count",
    "accepted_candidate_count",
    "candidate_acceptance_rate",
    "trade_count",
    "net_profit",
    "profit_per_trade",
    "profit_factor",
    "win_rate",
    "sharpe",
    "sortino",
    "max_drawdown",
    "ending_balance",
    "dominant_hour_trade_share",
    "top_three_hours_trade_share",
    "dominant_hour_pnl_share",
]
