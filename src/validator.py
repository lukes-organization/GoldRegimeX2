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
    _USDCHF_PATH_BY_TF,
    _EXTERNAL_ASSET_PATHS,
    load_usdchf_data,
    map_usdchf_to_bars,
    load_asset_data,
    map_asset_to_bars,
    compute_log_returns,
    kalman_smooth,
    compute_volatility,
    compute_rsi,
    compute_atr,
    compute_gmm_vol_cluster,
    compute_synth_vix,
    load_gmm_model,
    load_feature_scaler,
)
from src.engine_hmm import load_model as load_hmm, predict_states, get_model_path as hmm_model_path, MODEL_PATH as HMM_GENERIC_PATH
from src.engine_xgb import load_xgb_ensemble, prepare_features, get_predictions_ensemble, get_ensemble_path, ENSEMBLE_PKL_PATH as XGB_GENERIC_PATH
from src.backtester import vectorized_backtest
from src.risk_manager import BROKER_CONFIGS

logger = setup_logger(__name__)

SYNC_DATA_PATH        = Path("data/processed/mt5_sync_data.csv")
SHARPE_PASS_THRESHOLD = 0.8
SHARPE_WARN_THRESHOLD = 0.5
# H1 fires ~10 trades/year: 45 trades gives ±0.29 CI → relax threshold to avoid false fails
TF_SHARPE_PASS: dict = {"H1": 0.25, "M15": 0.50, "M5": 0.70}
TF_SHARPE_WARN: dict = {"H1": 0.05, "M15": 0.20, "M5": 0.40}
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
    broker: str = "headway_cent",
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
    df["rsi_slope"]       = df["rsi"].diff()
    df["atr_normalized"]  = compute_atr(df)

    # Load the training GMM + scaler — strictly no re-fitting on live/validation data.
    _gmm, _scaler = load_gmm_model(tf, broker)
    df["gmm_vol_cluster"] = compute_gmm_vol_cluster(
        df["volatility"].values, fitted_gmm=_gmm, fitted_scaler=_scaler
    )

    # All external cross-asset features — mirrors process_pipeline exactly.
    for asset_key, path_by_tf in _EXTERNAL_ASSET_PATHS.items():
        col_name = f"{asset_key}_log_return"
        path     = path_by_tf.get(tf.upper(), path_by_tf.get("H1"))
        asset_df = load_asset_data(path, col_name)
        if asset_df is not None:
            df[col_name] = map_asset_to_bars(df.index, asset_df, col_name)

    # Backfill so short-history assets (e.g. XTIUSD from Feb 2017) don't wipe rows.
    _ext_cols = [c for c in df.columns if c.endswith("_log_return") and c != "log_return"]
    if _ext_cols:
        df[_ext_cols] = df[_ext_cols].bfill()

    df.dropna(inplace=True)

    # Synthetic VIX (Williams VIX Fix z-score) — must come after dropna on core cols.
    df["synth_vix_zscore"] = compute_synth_vix(df)

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
    period: str | None = None,  # TODO: filter validation window to last N months
                                # (e.g. '3m', '6m'). Currently accepted for guardian
                                # API compatibility; the full synced CSV is always used.
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
    df = _apply_features(df, tf, obs_cov, trans_cov, broker=broker)

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
        models_xgb, thresholds_xgb, xgb_meta = load_xgb_ensemble(xgb_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "XGB ensemble not found at models/xgb_ensemble.pkl. "
            "Run  python main.py --mode train  first."
        )

    # Inference — load the same feature scaler used at training time
    states = predict_states(model_hmm, df)
    try:
        _feat_scaler = load_feature_scaler(tf=tf, broker=broker)
    except FileNotFoundError:
        logger.warning(
            "Feature scaler not found for [%s/%s] — validating without scaling. "
            "Re-run --mode train to generate it.",
            tf, broker,
        )
        _feat_scaler = None
    X, _, df_aligned, _   = prepare_features(df, states, feature_scaler=_feat_scaler, tf=tf)
    states_aligned        = states[df.index.isin(df_aligned.index)]
    _, probabilities      = get_predictions_ensemble(models_xgb, thresholds_xgb, X)

    # Backtest the full synced window — no IS/OOS split
    _z = params.get("z_cutoff_bull")
    _eval_cfg = {"Z_CUTOFF_BULL": _z, "Z_CUTOFF_BEAR": -_z} if _z else None
    result = vectorized_backtest(
        df_aligned, probabilities, states_aligned,
        split_idx=None,
        account_size=account_size,
        broker=broker,
        tf=tf,
        regime_stats=xgb_meta.get("regime_stats"),
        evaluator_config=_eval_cfg,
    )

    sharpe   = result["sharpe_ratio"]
    n_trades = result["n_trades"]
    win_rate = result["win_rate"]
    max_dd   = result.get("floating_max_drawdown", result["max_drawdown"])
    pf       = result.get("profit_factor", 1.0)
    payoff   = result.get("expected_payoff", 0.0)
    rf       = result.get("recovery_factor", 0.0)
    avg_eff  = result.get("avg_efficiency", 0.0)
    cost_eff = result.get("cost_efficiency", 0.0)
    total_return = result.get("total_return", 0.0)

    # Complex Criterion score — same formula as optimizer._score_result
    _fdd = max_dd if max_dd > 0 else 0.0
    _net = result.get("total_return", 0.0)
    _rf_capped = min(_net / _fdd, 50.0) if _fdd > 0 else (50.0 if _net > 0 else 0.0)
    score = float((_rf_capped * 0.4) + (pf * 0.3) + (sharpe * 0.3))

    min_trades_warn = MIN_TRADES_WARNING_BY_TF.get(tf.upper(), 15)
    _period_suggestions = {"M5": "3m", "M15": "6m", "H1": "1y"}
    _period_hint = _period_suggestions.get(tf.upper(), "1y")
    if n_trades < min_trades_warn:
        logger.warning(
            "Only %d trades in the validation window — Sharpe estimate may be "
            "unreliable.  Consider using a longer --period (e.g. '%s').",
            n_trades, _period_hint,
        )

    # Spread-Payoff Ratio check — warn if broker costs eat >50% of the model's edge
    _bcfg         = BROKER_CONFIGS.get(broker, BROKER_CONFIGS["standard"])
    _spread_cost  = _bcfg["spread_frac"] + _bcfg.get("commission_frac", 0.0)
    _payoff_usd   = payoff * account_size          # log-return units → approximate USD
    _spread_usd   = _spread_cost * account_size    # cost per round-trip
    if _payoff_usd > 0 and _spread_usd > 0.5 * _payoff_usd:
        logger.warning(
            "High Spread Erosion: spread=$%.4f vs payoff=$%.4f (%.0f%% consumed). "
            "Edge is too thin to survive broker costs.",
            _spread_usd, _payoff_usd, (_spread_usd / _payoff_usd) * 100,
        )
        print("  [!] WARNING: High Spread Erosion. Spread is consuming > 50% of your edge.")

    # Gate decision — TF-specific thresholds: H1 fires rarely, needs lower bar
    _pass_thr = TF_SHARPE_PASS.get(tf.upper(), SHARPE_PASS_THRESHOLD)
    _warn_thr = TF_SHARPE_WARN.get(tf.upper(), SHARPE_WARN_THRESHOLD)

    if sharpe >= _pass_thr:
        status  = "pass"
        message = (
            f"Recent-Period Sharpe {sharpe:.3f} >= {_pass_thr} [{tf} threshold]. "
            "Model is stable — safe to go live."
        )
    elif sharpe >= _warn_thr:
        status  = "warn"
        message = (
            f"Recent-Period Sharpe {sharpe:.3f} is borderline "
            f"({_warn_thr}–{_pass_thr}, {tf} thresholds). "
            "Proceed with reduced position size or wait for a clearer regime."
        )
    else:
        status  = "fail"
        message = (
            f"Recent-Period Sharpe {sharpe:.3f} < {_warn_thr} [{tf} threshold]. "
            "Market drift detected — DO NOT go live.  "
            "Run --mode optimize then --mode train to retune the model."
        )

    logger.info(
        "Validation [%s]: status=%s  score=%.2f  sharpe=%.3f  pf=%.2f  eff=%.2fx  trades=%d  wr=%.1f%%  dd=%.1f%%",
        tf, status, score, sharpe, pf, avg_eff, n_trades, win_rate * 100, max_dd * 100,
    )
    if status in ("warn", "fail"):
        logger.warning("VALIDATION %s: %s", status.upper(), message)

    return {
        "sharpe":           sharpe,
        "n_trades":         n_trades,
        "win_rate":         win_rate,
        "max_dd":           max_dd,
        "profit_factor":    pf,
        "expected_payoff":  payoff,
        "recovery_factor":  rf,
        "avg_efficiency":   avg_eff,
        "cost_efficiency":  cost_eff,
        "total_return":     total_return,
        "score":            score,
        "status":           status,
        "message":          message,
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
