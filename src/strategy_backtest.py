"""src/strategy_backtest.py -- single source of truth for scoring + backtesting.

This module is the *only* bridge between the trained notebook engine
(``src.grid_search_plateau``) and the rest of the application (train / validate
/ live).  It loads the exported live bundle produced by ``--mode optimize``
(``pipeline_verification_bundle/models/goldregimex_live_model.pkl``) and runs
inference + backtests through the SAME code path the notebook uses
(``grid_search_plateau.evaluate_ml_model``), so:

    the model that is optimized  ==  the model that is validated
                                 ==  the model that is traded.

The legacy ML stack (processor / engine_hmm / engine_xgb / backtester /
signal_engine) has been retired -- see CONSOLIDATION.md.

Canonical engine contract (from grid_search_plateau):
    feat  = build_features(panel, tf)          # adds regime_code, atr14, Close, ...
    probs = model.predict_proba_raw(feat)      # [:,0]=P(down)  [:,2]=P(up)
    trades, metrics = run_ml_filtered_backtest(tf, feat, probs, base_params, threshold)
Signals are TREND-gated (regime_code == 1) and confidence-filtered on the
grid-selected xgb_threshold.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__)

# Repo-root anchored path to the exported live bundle.
_REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_MODEL_PKL = _REPO_ROOT / "pipeline_verification_bundle" / "models" / "goldregimex_live_model.pkl"


def load_live_bundle(path: Path | str = LIVE_MODEL_PKL) -> dict:
    """Load the exported live bundle (models + thresholds + base_params).

    Raises FileNotFoundError with actionable guidance if optimize hasn't run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Live model bundle not found at {path}. "
            "Run  python main.py --mode optimize  first."
        )
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    return bundle


def bundle_params_for(tf: str, bundle: Optional[dict] = None):
    """Return (model, base_params, threshold) for a timeframe from the bundle.

    base_params already carries ``exit_model`` (matching the notebook export).
    ``threshold`` is the plateau-selected xgb_threshold for the timeframe.
    """
    tf = str(tf).upper()
    if bundle is None:
        bundle = load_live_bundle()
    models = bundle.get("models", {}) or {}
    thresholds = bundle.get("thresholds", {}) or {}
    base = bundle.get("base_params", {}) or {}
    if tf not in models:
        raise KeyError(f"Timeframe {tf} not in live bundle models: {list(models)}")
    if tf not in thresholds:
        raise KeyError(f"Timeframe {tf} not in live bundle thresholds: {list(thresholds)}")
    model = models[tf]
    base_params = dict(base.get(tf) or {})
    threshold = float(thresholds[tf])
    return model, base_params, threshold


def _build_feat(tf: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load the exec panel (or use ``df``) and build the engine feature frame.

    Features are always computed on the full supplied history so rolling
    windows (EMA200, ATR, triple-barrier, ...) are correct; callers may slice
    the RESULT afterwards.
    """
    from src import grid_search_plateau as engine
    panel = df if df is not None else engine.load_panel(tf)
    return engine.build_features(panel, tf)


def build_scored_frame(tf: str, df: Optional[pd.DataFrame] = None,
                       bundle: Optional[dict] = None) -> pd.DataFrame:
    """Return the feature frame with model probabilities + filtered signal.

    Adds columns: prob_down, prob_flat, prob_up, raw_signal, signal.
    Signal semantics mirror ``run_ml_filtered_backtest`` exactly:
      * raw_signal from TrendPullbackStrategy, zeroed outside TREND (regime_code==1)
      * signal keeps a long only when prob_up >= threshold, a short only when
        prob_down >= threshold, else 0.
    """
    from src import grid_search_plateau as engine
    model, base_params, threshold = bundle_params_for(tf, bundle)
    feat = _build_feat(tf, df)

    probs = np.asarray(model.predict_proba_raw(feat))
    feat = feat.copy()
    feat["prob_down"] = probs[:, 0]
    feat["prob_flat"] = probs[:, 1] if probs.shape[1] > 2 else 0.0
    feat["prob_up"] = probs[:, 2] if probs.shape[1] > 2 else probs[:, -1]

    raw = engine.TrendPullbackStrategy().generate_signals(feat, base_params)
    trend_mask = feat["regime_code"].to_numpy() == 1
    raw = raw.where(pd.Series(trend_mask, index=feat.index), 0)
    raw_arr = raw.to_numpy(dtype=np.int8)

    filtered = raw_arr.copy()
    filtered[(raw_arr == 1) & (feat["prob_up"].to_numpy() < threshold)] = 0
    filtered[(raw_arr == -1) & (feat["prob_down"].to_numpy() < threshold)] = 0

    feat["raw_signal"] = raw_arr
    feat["signal"] = filtered
    return feat


def backtest_tf(tf: str, df: Optional[pd.DataFrame] = None,
                bundle: Optional[dict] = None, tail: Optional[int] = None) -> dict:
    """Backtest a timeframe through the notebook engine and return metrics.

    Mirrors ``grid_search_plateau.evaluate_ml_model``.  ``df`` may supply an
    alternative panel (e.g. freshly synced data); otherwise ``load_panel`` is
    used.  ``tail`` slices the FEATURE frame to the most recent N rows AFTER
    feature construction (used by the validator for a recent-window Sharpe).

    Returned dict keys come straight from ``compute_metrics``:
      profit_factor, sharpe, sortino, calmar, max_drawdown (PERCENT),
      expectancy, win_rate (0-1), trade_count, net_profit, profit_per_trade,
      net_return_pct.
    """
    from src import grid_search_plateau as engine
    model, base_params, threshold = bundle_params_for(tf, bundle)
    feat = _build_feat(tf, df)
    if tail and len(feat) > int(tail):
        feat = feat.iloc[-int(tail):]

    probs = np.asarray(model.predict_proba_raw(feat))
    _trades, metrics = engine.run_ml_filtered_backtest(
        tf.upper(), feat, probs, base_params, float(threshold)
    )
    return dict(metrics)


def latest_signal(tf: str, df: Optional[pd.DataFrame] = None,
                  bundle: Optional[dict] = None) -> dict:
    """Return the most recent bar's trading decision for live execution.

    The returned dict is engine-consistent with the backtest and carries
    everything the live executor needs to size an order identically to how the
    backtest fills it:

        signal       -1 / 0 / +1   (SELL / no-trade / BUY)
        prob_up      float          P(up) for the last bar
        prob_down    float          P(down) for the last bar
        threshold    float          grid-selected xgb_threshold
        regime_code  int            HMM regime of the last bar (1 == TREND)
        atr          float          atr14 of the last bar (SL/TP sizing base)
        close        float          last close price
        base_params  dict           strategy params incl. atr_stop/atr_target/
                                     leg_a_atr_target/exit_model
        timestamp    Timestamp      index of the last bar

    SL/TP for live must be sized as  base_params['atr_stop'] * atr  (stop) and
    base_params['atr_target'] * atr  (target) so live == backtest.
    """
    _model, base_params, threshold = bundle_params_for(tf, bundle)
    feat = build_scored_frame(tf, df=df, bundle=bundle)
    last = feat.iloc[-1]
    return {
        "signal": int(last["signal"]),
        "prob_up": float(last["prob_up"]),
        "prob_down": float(last["prob_down"]),
        "threshold": float(threshold),
        "regime_code": int(last["regime_code"]),
        "atr": float(last["atr14"]) if "atr14" in feat.columns else float("nan"),
        "close": float(last["Close"]) if "Close" in feat.columns else float("nan"),
        "base_params": dict(base_params),
        "timestamp": feat.index[-1],
    }
