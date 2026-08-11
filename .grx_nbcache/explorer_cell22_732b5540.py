# =====================================================================
# EXPORT DEPLOYED MODEL ARTIFACT  ->  pipeline_verification_bundle/models
# =====================================================================
import pickle
from pathlib import Path

# --- Locate the bundle folder (repo_root resolved in the first cell) ---
_BUNDLE_DIR = Path(_repo_root) / "pipeline_verification_bundle"
_MODEL_DIR = _BUNDLE_DIR / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LIVE_MODEL_PKL = _MODEL_DIR / "goldregimex_live_model.pkl"

# --- Sanity: models must have been trained (run the training cell first) ---
_missing = [tf for tf in TIMEFRAMES if getattr(pipeline.get(tf), "model", None) is None]
if _missing:
    raise RuntimeError(
        "No trained model for %s. Run the training / IS-OOS validation cell "
        "first so pipeline[tf].model is populated." % _missing
    )

# --- Execution settings taken directly from THIS notebook -----------------
#     (these are what the live simulation will trade with -- lot size, etc.)
live_settings = {
    "position_a": float(POSITION_A),                       # leg A lot size
    "position_b": float(POSITION_B),                       # leg B lot size
    "initial_balance_cents": float(INITIAL_BALANCE_CENTS),
    "pip_size_price": float(PIP_SIZE_PRICE),
    "pip_value_cents_per_1lot": float(PIP_VALUE_CENTS_PER_1LOT),
    "slippage_pips": float(SLIPPAGE_PIPS),
    "commission_cents_per_trade": float(COMMISSION_CENTS_PER_TRADE),
    "spread_cap_points": float(SPREAD_CAP_POINTS),
    "lot_cycle_small": list(LOT_CYCLE_SMALL),
    "max_positions_per_cycle": int(MAX_POSITIONS_PER_CYCLE),
    "profit_scale_threshold_cents": float(PROFIT_SCALE_THRESHOLD_CENTS),
    "lot_cycle_scaled_options": list(LOT_CYCLE_SCALED_OPTIONS),
    "live_enable_lot_scaling": bool(LIVE_ENABLE_LOT_SCALING),
    "live_scaled_lot": float(LIVE_SCALED_LOT),
}

# --- Per-timeframe strategy parameters (from the Strategy Tester bridge) ---
live_base_params = {}
for tf in TIMEFRAMES:
    _bp = dict(ML_TARGET_PARAMS[tf]["parameters"])
    _bp["exit_model"] = ML_TARGET_PARAMS[tf]["exit_model"]
    live_base_params[tf] = _bp

live_bundle = {
    "schema": "goldregimex_live_v1",
    "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "random_state": int(RANDOM_STATE),
    "timeframes": list(TIMEFRAMES),
    "settings": live_settings,
    "models": {tf: pipeline[tf].model for tf in TIMEFRAMES},
    "thresholds": {tf: float(pipeline[tf].threshold) for tf in TIMEFRAMES},
    "base_params": live_base_params,
    "split_time": {tf: str(getattr(pipeline.get(tf), "split_time", None)) for tf in TIMEFRAMES},
}

with open(LIVE_MODEL_PKL, "wb") as _fh:
    pickle.dump(live_bundle, _fh, protocol=pickle.HIGHEST_PROTOCOL)

_size_kb = LIVE_MODEL_PKL.stat().st_size / 1024.0
print("Saved live model artifact:")
print("   path : %s" % LIVE_MODEL_PKL)
print("   size : %.1f KB" % _size_kb)
print("   TFs  : %s" % ", ".join(TIMEFRAMES))
for tf in TIMEFRAMES:
    print("   [%s] threshold=%.4f  lots=(A %.2f / B %.2f)  exit=%s" % (
        tf, live_bundle["thresholds"][tf], live_settings["position_a"],
        live_settings["position_b"], live_base_params[tf].get("exit_model", "fixed_tp")))