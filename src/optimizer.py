"""src/optimizer.py -- Grid-Sensitivity Plateau optimizer (Optuna removed).

Backward-compatible surface for the modules that still `from src.optimizer
import ...` (main.py, validator.py, mt5_trader.py). All optimization now runs
through src.grid_search_plateau, which sweeps ONLY the XGB confidence threshold
(`xgb_threshold`) via CPCV + purge/embargo and selects a plateau center. No
Optuna, no study DB, no Kalman-parameter search.

Key compatibility notes:
- `get_best_params(balance, broker, tf)` reads the selected threshold + base
  params straight from the exported live model pickle. It no longer returns
  `obs_cov` / `trans_cov` / `persistence_threshold`; callers already fall back
  to their own defaults for those (per the migration decision).
- `run_optimization(...)` delegates to the grid-plateau engine (which trains
  both timeframes and exports the live pickle) and returns a small Study-like
  object exposing `.best_value` / `.best_params` for legacy callers.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

# --- constants main.py imports at module scope ---
CPCV_N_BLOCKS = 4
CPCV_K_TEST = 1
_N_PATHS = math.comb(CPCV_N_BLOCKS, CPCV_K_TEST)
WFO_PARAMS = {"n_blocks": 4, "k_val_blocks": 1, "embargo_bars": 12}
WFO_PARAMS_FAST = dict(WFO_PARAMS)

_LIVE_PKL = (
    Path(__file__).resolve().parents[1]
    / "pipeline_verification_bundle"
    / "models"
    / "goldregimex_live_model.pkl"
)


class _StudyShim:
    """Minimal stand-in for an Optuna Study consumed by main.cmd_optimize."""

    def __init__(self, best_value, best_params):
        self.best_value = float(best_value)
        self.best_params = dict(best_params)


def resolve_n_states(tf, params=None):
    """Canonical regime contract: exactly 3 states for every timeframe."""
    return 3


def _score_result(result, *args, **kwargs):
    """Plateau score extractor (replaces the Optuna composite score)."""
    if isinstance(result, dict):
        return float(
            result.get("mean_sharpe", result.get("stability_adjusted_sharpe", 0.0)) or 0.0
        )
    return float(getattr(result, "score", 0.0) or 0.0)


def get_best_params(balance: float = 15.0, broker: str = "standard", tf: str = "H1") -> dict:
    """Read the plateau-selected threshold + base params from the live pickle.

    Self-contained (only needs pickle + pathlib) so validator/mt5_trader can
    call it without importing the heavy ML engine. Kalman covariances are NOT
    returned; callers fall back to their own defaults.
    """
    tf = str(tf).upper()
    if not _LIVE_PKL.exists():
        raise FileNotFoundError(
            "Live model bundle not found at %s. Run  python main.py --mode optimize  first."
            % _LIVE_PKL
        )
    with open(_LIVE_PKL, "rb") as fh:
        bundle = pickle.load(fh)
    thresholds = bundle.get("thresholds", {}) or {}
    base = bundle.get("base_params", {}) or {}
    if tf not in thresholds:
        raise KeyError("Timeframe %s not in live bundle thresholds: %s" % (tf, list(thresholds)))
    params = dict(base.get(tf) or {})
    params["xgb_threshold"] = float(thresholds[tf])
    return params


def extract_consensus_params(*args, **kwargs) -> dict:
    """Consensus == the single plateau-selected threshold for the timeframe."""
    tf = kwargs.get("tf") or (args[0] if args else "M5")
    broker = kwargs.get("broker", "standard")
    balance = kwargs.get("balance", 15.0)
    return get_best_params(balance=balance, broker=broker, tf=tf)


def run_optimization(
    df=None,
    tf=None,
    broker: str = "standard",
    account_size: float = 15.0,
    n_trials=None,
    wfo_mode: str = "standard",
    n_jobs=None,
    **kwargs,
):
    """Run the grid-plateau engine (both timeframes) and export the live pickle.

    `df` / `tf` / `n_trials` / `wfo_mode` are accepted for signature
    compatibility. The engine loads its own panels and trains M15 + M5 in one
    pass, so a per-timeframe `df` is not required.
    """
    from src import grid_search_plateau as engine

    out = engine.run_optimization(n_jobs=n_jobs)
    pipeline = out.get("pipeline", {}) or {}
    key = str(tf or getattr(engine, "EXEC_TF", "M5")).upper()
    p = pipeline.get(key)
    threshold = getattr(p, "threshold", None)
    plateau = getattr(p, "plateau", {}) or {}
    best_value = float(plateau.get("selection_score", 0.0) or 0.0)
    return _StudyShim(best_value, {"xgb_threshold": threshold})


def run_optimization_stage1(*args, **kwargs):
    """Staged Optuna optimization was removed; single-pass grid-plateau only."""
    raise NotImplementedError(
        "Staged optimization was removed with Optuna. Use run_optimization()."
    )


def run_wfa(
    df=None,
    tf: str = "M5",
    broker: str = "standard",
    account_size: float = 15.0,
    wfo_mode: str = "standard",
    n_jobs=None,
    **kwargs,
):
    """Walk-forward robustness via a single coarse CPCV grid over the IS window.

    Returns the dict shape main.cmd_wfa expects: n_windows, n_valid_windows,
    window_scores, std_sharpe, median_trades, wfo_score.
    """
    import numpy as np
    from src import grid_search_plateau as engine

    tf = str(tf).upper()
    if df is None:
        df = engine.load_panel(tf)
    is_df, _oos, _split = engine.split_dataset(df, engine.HOLDOUT_FRAC)
    grid = engine.build_coarse_grid()
    res = engine.run_grid_parallel(
        exec_df=is_df,
        grid=grid,
        n_blocks=WFO_PARAMS["n_blocks"],
        k_val_blocks=WFO_PARAMS["k_val_blocks"],
        embargo_bars=WFO_PARAMS["embargo_bars"],
        exec_tf=tf,
        n_jobs=n_jobs or getattr(engine, "N_JOBS", 1),
    )
    scores = [float(x) for x in res.get("mean_sharpe", []).tolist()] if hasattr(res, "get") else []
    if not scores and "mean_sharpe" in getattr(res, "columns", []):
        scores = [float(x) for x in res["mean_sharpe"].tolist()]
    valid = [s for s in scores if s > 0]
    std_sharpe = 0.0
    if "variance_sharpe" in getattr(res, "columns", []):
        std_sharpe = float(np.sqrt(res["variance_sharpe"].clip(lower=0)).mean())
    median_trades = 0
    if "median_trades" in getattr(res, "columns", []):
        median_trades = int(res["median_trades"].median())
    return {
        "n_windows": len(scores),
        "n_valid_windows": len(valid),
        "window_scores": scores,
        "std_sharpe": std_sharpe,
        "median_trades": median_trades,
        "wfo_score": float(np.median(scores)) if scores else 0.0,
    }
