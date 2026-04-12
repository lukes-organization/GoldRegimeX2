"""Model Validation Gatekeeper.

Loads recently synced MT5 data, applies the full feature pipeline, runs live
inference through the saved HMM + XGBoost models, and computes an annualised
Sharpe Ratio over the recent window.  The result is classified as:

    pass  (Sharpe >= 0.8) — model is stable; proceed to live.
    warn  (Sharpe 0.5–0.8) — borderline; proceed with extreme caution.
    fail  (Sharpe < 0.5)  — model drift detected; retune before going live.

NOTE: This validation Sharpe is computed on the most recent N months of live
market data.  It is NOT equivalent to the OOS Sharpe from training, which uses
an 80/20 chronological split of the full 10-year dataset.  Both metrics matter,
but they answer different questions.
"""

import time as _time

import numpy as np
import pandas as pd
from pathlib import Path

from src.logger import setup_logger
from src.processor import (
    TF_CONFIG,
    USDCHF_MASTER_PATH,
    load_usdchf_data,
    map_usdchf_to_bars,
    compute_log_returns,
    kalman_smooth,
    compute_volatility,
    compute_rsi,
    compute_atr,
)
from src.engine_hmm import load_model as load_hmm, predict_states, get_model_path as hmm_model_path, MODEL_PATH as HMM_GENERIC_PATH
from src.engine_xgb import load_xgb_ensemble, prepare_features, get_predictions_ensemble, get_ensemble_path, ENSEMBLE_PKL_PATH as XGB_GENERIC_PATH
from src.backtester import vectorized_backtest

logger = setup_logger(__name__)

SYNC_DATA_PATH        = Path("data/processed/mt5_sync_data.csv")
SHARPE_PASS_THRESHOLD = 0.8
SHARPE_WARN_THRESHOLD = 0.5
# TF-aware minimum: H1 naturally fires ~10 trades/quarter, M15 ~15, M5 ~30+
MIN_TRADES_WARNING_BY_TF: dict = {"M5": 30, "M15": 15, "H1": 10}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_features(
    df: pd.DataFrame,
    tf: str,
    obs_cov: float,
    trans_cov: float,
) -> pd.DataFrame:
    """Apply the same feature steps as process_pipeline() to an arbitrary frame.

    This function mirrors processor.process_pipeline() exactly.  If any
    standalone function in processor.py changes its behaviour, this must be
    updated to match — both must produce identical column values for the same
    input data.
    """
    cfg = TF_CONFIG[tf.upper()]
    obs_cov   = obs_cov   if obs_cov   is not None else cfg["obs_cov_default"]
    trans_cov = trans_cov if trans_cov is not None else cfg["trans_cov_default"]

    df = df.copy()
    df["log_return"]     = compute_log_returns(df["Close"])
    df["kalman_return"]  = kalman_smooth(df["log_return"].values, obs_cov, trans_cov)
    df["volatility"]     = compute_volatility(df["log_return"])
    df["rsi"]            = compute_rsi(df["Close"])
    df["rsi_slope"]      = df["rsi"].diff()
    df["atr_normalized"] = compute_atr(df)

    # Mirror process_pipeline: add USDCHF if the master file exists
    usdchf_df = load_usdchf_data(USDCHF_MASTER_PATH)
    if usdchf_df is not None:
        df["usdchf_log_return"] = map_usdchf_to_bars(df.index, usdchf_df)

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Primary entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    sync_data_path: Path = SYNC_DATA_PATH,
    tf: str = "H1",
    broker: str = "headway_cent",
    account_size: float = 15.0,
    obs_cov: float = None,
    trans_cov: float = None,
) -> dict:
    """Validate the saved models against recent synced MT5 data.

    Returns a dict:
        sharpe   (float)  — annualised Sharpe over the validation window
        n_trades (int)    — number of trades fired by the model
        win_rate (float)  — fraction of winning trades
        max_dd   (float)  — maximum drawdown (0–1)
        status   (str)    — 'pass', 'warn', or 'fail'
        message  (str)    — human-readable gate decision

    Raises:
        FileNotFoundError  if the sync CSV or model files are missing.
    """
    sync_data_path = Path(sync_data_path)
    if not sync_data_path.exists():
        raise FileNotFoundError(
            f"Sync data not found at {sync_data_path}. "
            "Run  python main.py --mode sync_validate  first."
        )

    # Attempt to load Optuna best params; fall back to TF defaults
    try:
        from src.optimizer import get_best_params
        params   = get_best_params(balance=account_size, broker=broker, tf=tf)
        obs_cov   = params.get("obs_cov")   or obs_cov
        trans_cov = params.get("trans_cov") or trans_cov
    except Exception:
        params = {}

    # Load and featurise data
    df = pd.read_csv(sync_data_path, index_col="Date", parse_dates=True)
    logger.info("Loaded %d bars from %s for validation", len(df), sync_data_path)
    df = _apply_features(df, tf, obs_cov, trans_cov)

    # Load models — prefer broker+TF specific file; fall back to generic
    hmm_path = hmm_model_path(tf, broker)
    if not hmm_path.exists():
        hmm_path = HMM_GENERIC_PATH
        if hmm_path.exists():
            logger.warning(
                "No TF-specific HMM model found for %s; falling back to %s. "
                "Run --mode train --tf %s to create a dedicated model.", tf, hmm_path, tf
            )
    try:
        model_hmm = load_hmm(hmm_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"HMM model not found at {hmm_path}. "
            "Run  python main.py --mode train  first."
        )

    xgb_path = get_ensemble_path(tf, broker)
    if not xgb_path.exists():
        xgb_path = XGB_GENERIC_PATH
        if xgb_path.exists():
            logger.warning(
                "No broker+TF XGB ensemble found for %s/%s; falling back to %s. "
                "Run --mode train --tf %s --broker %s to create a dedicated model.", tf, broker, xgb_path, tf, broker
            )
    try:
        models_xgb, thresholds_xgb, _ = load_xgb_ensemble(xgb_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "XGB ensemble not found at models/xgb_ensemble.pkl. "
            "Run  python main.py --mode train  first."
        )

    # Inference
    states                = predict_states(model_hmm, df)
    X, _, df_aligned      = prepare_features(df, states)
    states_aligned        = states[df.index.isin(df_aligned.index)]
    _, probabilities      = get_predictions_ensemble(models_xgb, thresholds_xgb, X)

    # Backtest the full synced window — no IS/OOS split
    result = vectorized_backtest(
        df_aligned, probabilities, states_aligned,
        split_idx=None,
        account_size=account_size,
        broker=broker,
        tf=tf,
        prob_threshold=params.get("prob_threshold"),
        short_threshold=params.get("short_threshold"),
    )

    sharpe   = result["sharpe_ratio"]
    n_trades = result["n_trades"]
    win_rate = result["win_rate"]
    max_dd   = result["max_drawdown"]

    min_trades_warn = MIN_TRADES_WARNING_BY_TF.get(tf.upper(), 15)
    if n_trades < min_trades_warn:
        logger.warning(
            "Only %d trades in the validation window — Sharpe estimate may be "
            "unreliable.  Consider using a longer --period (e.g. '6m').",
            n_trades,
        )

    # Gate decision
    if sharpe >= SHARPE_PASS_THRESHOLD:
        status  = "pass"
        message = (
            f"Recent-Period Sharpe {sharpe:.3f} >= {SHARPE_PASS_THRESHOLD}. "
            "Model is stable — safe to go live."
        )
    elif sharpe >= SHARPE_WARN_THRESHOLD:
        status  = "warn"
        message = (
            f"Recent-Period Sharpe {sharpe:.3f} is borderline "
            f"({SHARPE_WARN_THRESHOLD}–{SHARPE_PASS_THRESHOLD}). "
            "Proceed with reduced position size or wait for a clearer regime."
        )
    else:
        status  = "fail"
        message = (
            f"Recent-Period Sharpe {sharpe:.3f} < {SHARPE_WARN_THRESHOLD}. "
            "Market drift detected — DO NOT go live.  "
            "Run --mode optimize then --mode train to retune the model."
        )

    logger.info(
        "Validation [%s]: status=%s  sharpe=%.3f  trades=%d  wr=%.1f%%  dd=%.1f%%",
        tf, status, sharpe, n_trades, win_rate * 100, max_dd * 100,
    )
    if status in ("warn", "fail"):
        logger.warning("VALIDATION %s: %s", status.upper(), message)

    return {
        "sharpe":   sharpe,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "max_dd":   max_dd,
        "status":   status,
        "message":  message,
    }


def check_model_age(tf: str = "H1", broker: str = "headway_cent") -> float:
    """Return the age of the saved ensemble model in days, based on file mtime.

    Checks the broker+TF specific ensemble pkl first; falls back to the generic
    path.  Returns ``float('inf')`` if no model file exists anywhere — the caller
    should treat this as an infinitely stale model.
    """
    from src.engine_xgb import get_ensemble_path, ENSEMBLE_PKL_PATH as _GENERIC

    path = get_ensemble_path(tf, broker)
    if not path.exists():
        path = _GENERIC
    if not path.exists():
        return float("inf")

    age_sec = _time.time() - path.stat().st_mtime
    return age_sec / 86_400   # convert seconds → days
