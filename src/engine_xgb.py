import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import onnx
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from src.logger import setup_logger

logger = setup_logger(__name__)

ONNX_PATH        = Path("models/xgb_model.onnx")
XGB_PKL_PATH     = Path("models/xgb_model.pkl")
ENSEMBLE_PKL_PATH = Path("models/xgb_ensemble.pkl")

# Continuous features that are StandardScaler-normalized before XGBoost.
# Discrete/categorical columns (hmm_state, gmm_vol_cluster) are excluded.
# LSTM context columns (lstm_ctx_0..3) are tanh-activated [-1,1] -- no scaling.
# Staleness counters ({asset}_staleness) are integer counts -- no scaling needed.
_CONTINUOUS_COLS = [
    "rsi_slope", "atr_normalized", "prev_log_return",
    "usdchf_log_return", "xagusd_log_return", "xtiusd_log_return",
    "us500_log_return", "usdjpy_log_return", "synth_vix_zscore",
    "atr_band_position",
    # Phase 3 specialization features (present after add_specialized_features)
    "atr_expansion_ratio", "realized_vol_ratio", "atr_slope",
    "minutes_from_london_open", "minutes_from_ny_open",
]

# All optional external-asset feature columns (log returns + synth_vix + cyclic time)
# US500 and USDJPY removed -- consistently 0.0 importance across H1 trials.
_EXTERNAL_ASSETS = [
    "usdchf_log_return", "xagusd_log_return", "xtiusd_log_return",
    "synth_vix_zscore", "atr_band_position",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    # Phase 3 specialization features
    "atr_expansion_ratio", "realized_vol_ratio", "atr_slope",
    "minutes_from_london_open", "minutes_from_ny_open",
]

# LSTM context feature columns added when a trained LSTM context model is present.
LSTM_CONTEXT_COLS = [f"lstm_ctx_{i}" for i in range(4)]


def get_ensemble_path(tf: str, broker: str = "headway_cent") -> Path:
    """Return the TF+broker-specific XGB ensemble path.

    Example: get_ensemble_path("H1", "headway_cent") -> models/xgb_ensemble_H1_headway_cent.pkl
    Falls back to the generic models/xgb_ensemble_H1.pkl (then ENSEMBLE_PKL_PATH) if absent.
    """
    return Path(f"models/xgb_ensemble_{tf.upper()}_{broker}.pkl")


def load_meta_model(path: Path):
    """Load an optional meta-label XGBoost model from *path* (Phase 4).

    Returns None when the artifact is absent so callers can fall back to
    primary-only mode without crashing.  The meta model gates entries with
    a second-stage confidence filter on top of the primary direction model.

    Args:
        path: Path to the joblib-serialised meta-label model artifact.

    Returns:
        Loaded model object, or None if the file does not exist.
    """
    if not path.exists():
        return None
    return joblib.load(path)

# Base features always present; gmm_vol_cluster and usdchf_log_return are
# added conditionally when the columns are present in the processed DataFrame.
# Feature order (canonical): hmm_state | gmm_vol_cluster | rsi_slope |
#                             atr_normalized | prev_log_return | usdchf_log_return
FEATURE_COLS    = ["hmm_state", "gmm_vol_cluster", "rsi_slope", "atr_normalized", "prev_log_return"]
GMM_FEATURE     = "gmm_vol_cluster"
USDCHF_FEATURE  = "usdchf_log_return"
DXY_FEATURE     = USDCHF_FEATURE   # legacy alias -- kept so old imports don't crash

# Volatility bucket labels (ATR tertiles: low / med / high)
VOL_BUCKETS = ["low", "med", "high"]


def _validate_state_feature_schema(df: "pd.DataFrame", tf: str) -> None:
    """Raise if the HMM state feature columns violate the canonical 3-state contract.

    H1  : must have integer 'hmm_state'; 'state_3' is forbidden.
    M15/M5: must have exactly 'state_0', 'state_1', 'state_2'; 'state_3' is forbidden.

    Called immediately after OHE / feature construction in prepare_features.
    """
    tfu = tf.upper()
    if tfu == "H1":
        if "hmm_state" not in df.columns:
            raise ValueError(
                "H1 feature schema error: 'hmm_state' column is missing. "
                "Expected integer column with values in {0,1,2}."
            )
        if "state_3" in df.columns:
            raise ValueError(
                "H1 feature schema error: unexpected 'state_3' column detected. "
                "4-state HMM output has leaked into features. Re-run with n_states=3."
            )
        return
    # M15 / M5
    required = {"state_0", "state_1", "state_2"}
    missing  = sorted(c for c in required if c not in df.columns)
    if missing:
        raise ValueError(
            f"{tfu} feature schema error: missing OHE columns {missing}. "
            f"Expected state_0, state_1, state_2 from 3-state HMM output."
        )
    if "state_3" in df.columns:
        raise ValueError(
            f"{tfu} feature schema error: unexpected 'state_3' column detected. "
            f"4-state HMM output has leaked into features. Re-run with n_states=3."
        )

# Timeframe-specific IS/OOS split ratios.
# H1 has a smaller dataset (~125K bars but 21 years of hourly data) -- 70/30 gives
# a more realistic OOS window without starving the scaler fit.
# M15/M5 have larger bar counts per year so a wider OOS window (35%) is feasible.
TF_TRAIN_RATIO = {"H1": 0.70, "M15": 0.65, "M5": 0.65}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the feature column list for this DataFrame.

    Starts from FEATURE_COLS (base set including gmm_vol_cluster), then:
    - For M5/M15 (after OHE): replaces hmm_state with state_0, state_1, state_2.
    - Drops gmm_vol_cluster if not present (old parquet without GMM).
    - Appends any external asset log returns and synth_vix that are present
      and >50% non-null (graceful degradation when masters are absent).
    """
    cols = []
    for c in FEATURE_COLS:
        if c == "hmm_state":
            # OHE path (M5/M15): state_* columns replace the integer hmm_state.
            # H1 fall-through: hmm_state column still present as integer.
            if "state_0" in df.columns:
                for sc in ("state_0", "state_1", "state_2"):
                    if sc in df.columns:
                        cols.append(sc)
            elif c in df.columns:
                cols.append(c)
        elif c == GMM_FEATURE:
            if c in df.columns and df[c].notna().mean() > 0.5:
                cols.append(c)
        else:
            cols.append(c)
    # External asset log returns + synth_vix (ordered consistently)
    for c in _EXTERNAL_ASSETS:
        if c in df.columns and df[c].notna().mean() > 0.5:
            cols.append(c)
    return cols


# -- Triple-Barrier Config (per spec: pt/sl as ATR multiples, vb as bar horizon) -
# pt (profit target): ATR multiples for upper barrier
# sl (stop loss):     ATR multiples for lower barrier
# vb (vertical bar):  max look-forward bars before time-barrier fires
# Wider pt + tighter sl = regime-specific asymmetric risk/reward.
TB_CONFIG = {
    "H1":  {"pt": 2.5, "sl": 1.0, "vb": 48},
    "M15": {"pt": 2.0, "sl": 1.0, "vb": 24},
    "M5":  {"pt": 1.5, "sl": 1.0, "vb": 12},
}
# Internal alias for _compute_triple_barrier_labels (keeps existing call sites)
_TBM_CONFIG = {
    tf: {"tp_mult": v["pt"], "sl_mult": v["sl"], "horizon": v["vb"]}
    for tf, v in TB_CONFIG.items()
}


def _compute_triple_barrier_labels(
    closes: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
) -> np.ndarray:
    """Compute Triple Barrier Method labels (L?pez de Prado, AFML Ch. 3).

    For each bar i, scans the next `horizon` bars to determine which barrier
    is touched first:
        +1  upper barrier hit (entry_close * (1 + tp_mult * atr_pct)) -- profit
        -1  lower barrier hit (entry_close * (1 - sl_mult * atr_pct)) -- loss
         0  time barrier (neither barrier touched in horizon bars) -- ambiguous

    Args:
        closes:   Close price array (raw, not log-transformed).
        atr:      ATR array in price units (raw ATR, not normalised by price).
        tp_mult:  Upper barrier as multiple of ATR.
        sl_mult:  Lower barrier as multiple of ATR (positive value -> subtract).
        horizon:  Max look-forward bars before time barrier fires.

    Returns:
        labels: int8 array of {-1, 0, +1} aligned with closes.
                The last `horizon` bars are set to 0 because there are
                insufficient future bars to evaluate barriers.
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        entry = closes[i]
        if entry <= 0 or np.isnan(entry):
            continue
        atr_frac = atr[i] / entry if entry > 0 else 0.0
        upper = entry * (1.0 + tp_mult * atr_frac)
        lower = entry * (1.0 - sl_mult * atr_frac)

        for j in range(1, horizon + 1):
            future_close = closes[i + j]
            if future_close >= upper:
                labels[i] = 1
                break
            elif future_close <= lower:
                labels[i] = -1
                break
        # If neither barrier hit: labels[i] stays 0 (time barrier -- ambiguous)

    return labels


def prepare_features(df: pd.DataFrame, hmm_states: np.ndarray, feature_scaler=None, tf: str = "H1"):
    """Build the XGBoost feature matrix from a featurised DataFrame.

    Continuous features (RSI slope, ATR, returns) are scaled to zero-mean /
    unit-variance using a StandardScaler fitted on the IS portion of the data
    so the model always sees a normalised distribution regardless of absolute
    price levels.  The IS fraction is TF-specific (see TF_TRAIN_RATIO).

    Args:
        df:             Featurised DataFrame from process_pipeline / _apply_features.
        hmm_states:     HMM state array aligned with df.
        feature_scaler: Pre-fitted StandardScaler to reuse (inference / validation).
        tf:             Timeframe string -- drives the IS split ratio for scaler fitting.
                        When None a new scaler is fitted on IS data (training mode).

    Returns:
        X             (pd.DataFrame)     -- scaled feature matrix
        y             (pd.Series)        -- binary target (next-bar direction)
        df_aligned    (pd.DataFrame)     -- df rows aligned to X
        scaler        (StandardScaler)   -- fitted scaler (same object as input when
                                           feature_scaler is provided)
    """
    df = df.copy()
    df["hmm_state"] = hmm_states

    # -- One-hot encode HMM states for M5/M15 (causal rolling median) ---------
    # Use a backward-looking window so no future state information leaks into
    # training.  The integer hmm_state is replaced by state_0/1/2 indicator
    # columns that XGBoost treats as unordered categories, preventing false
    # ordinal ranking (state 2 > state 1 > state 0).
    if tf.upper() in ("M5", "M15"):
        df["hmm_state_smoothed"] = (
            df["hmm_state"]
            .rolling(window=2, min_periods=1)
            .median()
            .astype(int)
        )
        df.drop("hmm_state", axis=1, inplace=True)
        df = pd.get_dummies(df, columns=["hmm_state_smoothed"], prefix="state", dtype=int)
        # Ensure all 3 columns exist even when a state never appears in this window
        for _i in range(3):
            _col = f"state_{_i}"
            if _col not in df.columns:
                df[_col] = 0

    # Validate HMM state feature schema against canonical 3-state contract.
    _validate_state_feature_schema(df, tf)

    df["prev_log_return"] = df["log_return"].shift(1)

    # TF-specific target: label 1 when the *next N bars* cumulative return
    # exceeds a minimum profit margin.
    #
    # Horizon rationale:
    #   H1 : 6 bars =  6 hours -- regime moves are well-defined at this scale.
    #   M15: 12 bars = 3 hours -- 6-bar window too noisy (90-min window near
    #        random walk); 12 bars averages over more regime structure.
    #   M5 : 18 bars = 1.5 hours -- same logic applied to finer granularity.
    #
    # Minimum profit margin rationale:
    #   Near-zero forward returns are statistically indistinguishable from noise
    #   (bid-ask spread + slippage).  A small TF-specific margin moves the
    #   decision boundary away from the near-flat zone, reducing ambiguous
    #   labels that carry zero predictive information.
    #   H1 : 0.0000 (spread is negligible relative to 6-hour moves)
    #   M15: 0.0002 (~2 pips gold, ~= half the typical gold spread)
    #   M5 : 0.0001 (~1 pip gold)
    tf_up  = (tf or "H1").upper()
    _tbm   = _TBM_CONFIG.get(tf_up, _TBM_CONFIG["H1"])

    # Compute raw ATR in price units for barrier placement.
    # Uses 14-bar ATR on absolute price moves (not the normalised feature).
    _raw_atr = (
        df["Close"]
        .pct_change()
        .abs()
        .rolling(14, min_periods=1)
        .mean()
        .bfill()
        .values
        * df["Close"].values        # convert fraction -> price units
    )

    _tbm_labels = _compute_triple_barrier_labels(
        closes  = df["Close"].values,
        atr     = _raw_atr,
        tp_mult = _tbm["tp_mult"],
        sl_mult = _tbm["sl_mult"],
        horizon = _tbm["horizon"],
    )

    # Map +1/-1 -> 1/0 for binary classification; exclude ambiguous (0) labels
    y = pd.Series(
        np.where(_tbm_labels == 1, 1, np.where(_tbm_labels == -1, 0, np.nan)),
        index=df.index,
        name="target",
    )

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid].copy()
    y = y[valid]
    df_aligned = df.loc[X.index]

    # Scale continuous features so XGBoost sees the 10-year mean/std distribution
    cont_cols = [c for c in _CONTINUOUS_COLS if c in X.columns]
    if feature_scaler is None:
        # Training: fit only on IS portion to prevent future-data leakage.
        # Split ratio is TF-specific -- H1: 70%, M15/M5: 65%.
        scaler = StandardScaler()
        split_ratio = TF_TRAIN_RATIO.get(tf.upper(), 0.70)
        split_idx = int(len(X) * split_ratio)
        scaler.fit(X.iloc[:split_idx][cont_cols])
    else:
        scaler = feature_scaler
    X[cont_cols] = scaler.transform(X[cont_cols])

    logger.info("Features prepared: %d samples, %d features: %s", len(X), len(feature_cols), feature_cols)
    return X, y, df_aligned, scaler


def train_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    subsample: float = 0.8,
    min_child_weight: int = 5,
    gamma: float = 1.0,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 1.0,
    train_ratio: float = 0.8,
    training_context: str = "global",
):
    split_idx = int(len(X) * train_ratio)
    has_holdout = split_idx < len(X)
    X_train = X.iloc[:split_idx] if has_holdout else X
    y_train = y.iloc[:split_idx] if has_holdout else y
    X_test  = X.iloc[split_idx:] if has_holdout else None
    y_test  = y.iloc[split_idx:] if has_holdout else None

    # Adaptive min_child_weight cap: prevents zero-split (null) models when the
    # bucket training set is small.  At max_depth d, a full binary tree has 2^d
    # leaves; each needs min_child_weight samples.  Capping at rows / (2^d)
    # guarantees at least one valid split path even under heavy regularisation.
    es_preview = int(len(X_train) * 0.85) if not has_holdout else len(X_train)
    effective_train = max(es_preview, 30)
    max_safe_mcw = max(3, effective_train // (2 ** max_depth))
    if min_child_weight > max_safe_mcw:
        logger.debug(
            "min_child_weight capped %d->%d (effective_train=%d, max_depth=%d)",
            min_child_weight, max_safe_mcw, effective_train, max_depth,
        )
        min_child_weight = max_safe_mcw

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    if has_holdout:
        # Explicit holdout: use it for early stopping and accuracy reporting
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        X_eval, y_eval = X_test, y_test
    else:
        # Full CPCV mode (train_ratio=1.0): without a holdout XGBoost builds all
        # n_estimators trees -> 90%+ train accuracy (severe overfitting).
        # Fix: carve the last 15% of the stitched CPCV training path as an internal
        # eval set for early stopping.  This halts tree growth before memorisation
        # without leaking future OOS data into the scaler or feature distribution.
        es_idx = int(len(X_train) * 0.85)
        X_t, y_t = X_train.iloc[:es_idx], y_train.iloc[:es_idx]
        X_v, y_v = X_train.iloc[es_idx:], y_train.iloc[es_idx:]
        model.set_params(early_stopping_rounds=50)
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        X_eval, y_eval = X_v, y_v

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_eval, model.predict(X_eval))
    importance = dict(zip(list(X.columns), model.feature_importances_))
    _hmm_imp    = importance.get("hmm_state", None)
    _hmm_unique = X["hmm_state"].nunique() if "hmm_state" in X.columns else None
    if _hmm_imp == 0.0:
        if training_context == "regime_split" or _hmm_unique == 1:
            # Expected: in regime-split training the hmm_state column is
            # constant within the subset (all rows share the same regime),
            # so XGBoost correctly assigns it zero importance.
            logger.info(
                "hmm_state importance=0 is expected in regime-split training "
                "(state is constant within the regime subset)."
            )
        else:
            logger.warning(
                "hmm_state importance=0 in global training: regime signal may be "
                "ineffective. Check regularization (reg_alpha/gamma) and n_states."
            )

    logger.info("XGB Train Acc: %.4f | Test Acc: %.4f", train_acc, test_acc)
    logger.info("Feature importance: %s", importance)

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "feature_importance": importance,
        "split_idx": split_idx,
    }
    return model, metrics


def get_predictions(model: xgb.XGBClassifier, X: pd.DataFrame):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities


def save_xgb(model: xgb.XGBClassifier, metrics: dict = None, path: Path = XGB_PKL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics or {}}, path)
    logger.info("XGB model saved to %s", path)


def load_xgb(path: Path = XGB_PKL_PATH):
    data = joblib.load(path)
    if isinstance(data, dict) and "model" in data:
        model = data["model"]
        metrics = data.get("metrics", {})
    else:
        # Backwards compat: old saves stored just the model
        model = data
        metrics = {
            "feature_importance": dict(zip(FEATURE_COLS, model.feature_importances_)),
        }
    return model, metrics


def _strip_zipmap(onnx_model):
    """Remove ZipMap node from ONNX graph, exposing the raw float probability tensor.

    onnxmltools converts XGBoost probabilities as ZipMap (sequence of maps).
    MT5's OnnxRun expects a plain float32 tensor.  This surgery replaces the
    ZipMap output with the raw tensor it is wrapping, making the model fully
    compatible with MT5's OnnxSetOutputShape / OnnxRun API.
    """
    import onnx as _onnx

    graph = onnx_model.graph

    float_tensor_name  = None
    zipmap_output_name = None
    nodes_to_keep      = []

    for node in graph.node:
        if node.op_type == "ZipMap":
            float_tensor_name  = node.input[0]
            zipmap_output_name = node.output[0]
        else:
            nodes_to_keep.append(node)

    if float_tensor_name is None:
        return onnx_model  # no ZipMap present -- nothing to do

    new_outputs = []
    for output in graph.output:
        if output.name == zipmap_output_name:
            new_outputs.append(
                _onnx.helper.make_tensor_value_info(
                    float_tensor_name,
                    _onnx.TensorProto.FLOAT,
                    None,   # shape inferred at runtime
                )
            )
        else:
            new_outputs.append(output)

    new_graph = _onnx.helper.make_graph(
        nodes_to_keep,
        graph.name,
        list(graph.input),
        new_outputs,
        list(graph.initializer),
    )
    new_model = _onnx.helper.make_model(
        new_graph, opset_imports=onnx_model.opset_import
    )
    new_model.ir_version = onnx_model.ir_version
    return new_model


def export_onnx(model: xgb.XGBClassifier, n_features: int = 4, path: Path = ONNX_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)

    # onnxmltools requires feature names in 'f%d' pattern
    # Clone the model's booster with generic feature names
    import copy
    model_copy = copy.deepcopy(model)
    model_copy.get_booster().feature_names = [f"f{i}" for i in range(n_features)]

    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model_copy, initial_types=initial_type)
    onnx_model = _strip_zipmap(onnx_model)   # expose float tensor output for MT5

    onnx.save_model(onnx_model, str(path))
    onnx.checker.check_model(onnx_model)

    inputs    = [i.name for i in onnx_model.graph.input]
    outputs   = [o.name for o in onnx_model.graph.output]
    n_classes = model.n_classes_
    logger.info(
        "ONNX exported to %s | inputs: %s | outputs: %s | n_classes=%d",
        path, inputs, outputs, n_classes,
    )
    print(f"\n  ONNX export OK -- n_classes={n_classes}. "
          f"Set NStates={n_classes} in the MT5 EA inputs.\n")


# -----------------------------------------------------------------------------
# Three-model volatility-regime ensemble
# -----------------------------------------------------------------------------

def compute_vol_thresholds(atr_normalized: pd.Series) -> tuple[float, float]:
    """Compute ATR 33rd and 66th percentile thresholds from the supplied series.

    Call this on **IS data only** to avoid look-ahead bias.  The returned
    thresholds are stored with the ensemble and reused at inference time.
    """
    p33 = float(np.nanpercentile(atr_normalized.values, 33))
    p66 = float(np.nanpercentile(atr_normalized.values, 66))
    return p33, p66


def assign_vol_bucket(atr_values: np.ndarray, p33: float, p66: float) -> np.ndarray:
    """Assign each bar to a volatility bucket string: ``'low'``, ``'med'``, or ``'high'``."""
    return np.where(atr_values <= p33, "low",
           np.where(atr_values <= p66, "med", "high"))


def train_xgb_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,
    **xgb_kwargs,
) -> tuple[dict, tuple[float, float], dict]:
    """Train three XGBoost classifiers on Low / Med / High ATR volatility subsets.

    When train_ratio=1.0 (used by CPCV, which owns the IS/OOS split), the
    entire X is used as training data and the thresholds are computed on the
    full set.  No internal OOS evaluation is performed in this mode -- all
    validation is handled externally by the CPCV loop.

    Args:
        X:           Feature DataFrame (output of ``prepare_features``).
        y:           Target Series.
        train_ratio: Fraction of data used for IS training (default 0.8).
                     Pass 1.0 when CPCV owns the split.
        **xgb_kwargs: Forwarded to ``train_xgb`` for each bucket model.

    Returns:
        models:      ``{"low": model, "med": model, "high": model}``
        thresholds:  ``(p33, p66)`` ATR tertile boundaries from IS data.
        metrics:     Aggregate metrics dict including ``split_idx``,
                     ``vol_thresholds``, and ``feature_cols``.
    """
    split_idx = int(len(X) * train_ratio)
    # When train_ratio=1.0 (CPCV mode), split_idx == len(X) and X_is == X.
    # This is correct -- the caller owns the IS/OOS split.
    X_is = X.iloc[:split_idx] if split_idx < len(X) else X
    y_is = y.iloc[:split_idx] if split_idx < len(X) else y

    # Thresholds from IS only -- no look-ahead into OOS bars
    p33, p66 = compute_vol_thresholds(X_is["atr_normalized"])
    buckets_is = assign_vol_bucket(X_is["atr_normalized"].values, p33, p66)

    # gmm_vol_cluster is redundant within each vol-bucket: routing on atr_normalized
    # already encodes the volatility regime, so gmm_vol_cluster is near-constant
    # inside every bucket (e.g. all 0s in "low", all 1s in "med") and XGBoost
    # assigns it zero importance.  Dropping it keeps colsample_bytree efficient.
    bucket_feature_cols = [c for c in list(X.columns) if c != GMM_FEATURE]

    models = {}
    bucket_sizes = {}
    for bucket in VOL_BUCKETS:
        mask = (buckets_is == bucket)
        X_b = X_is[mask][bucket_feature_cols]
        y_b = y_is[mask]
        bucket_sizes[bucket] = int(mask.sum())

        if len(X_b) < 100:
            # Too few samples in this bucket -- fall back to all IS data
            logger.warning(
                "Vol bucket '%s' has only %d IS samples -- falling back to full IS data.",
                bucket, len(X_b),
            )
            X_b, y_b = X_is[bucket_feature_cols], y_is

        model, _ = train_xgb(X_b, y_b, train_ratio=train_ratio, **xgb_kwargs)
        models[bucket] = model
        logger.info("Trained vol-bucket '%s': %d samples.", bucket, len(X_b))

    # Feature importance from the med bucket (most representative)
    fi = {}
    if "med" in models:
        try:
            fi = dict(zip(bucket_feature_cols, models["med"].feature_importances_))
        except Exception:
            pass

    metrics = {
        "split_idx":        split_idx if split_idx < len(X) else None,
        "vol_thresholds":   (p33, p66),
        "feature_cols":     list(X.columns),
        "bucket_sizes":     bucket_sizes,
        "feature_importance": fi,
        "train_accuracy":   0.0,   # not meaningful for ensemble (per-bucket varies)
        "test_accuracy":    0.0,
    }
    logger.info(
        "Ensemble trained: buckets=%s  thresholds=(%.5f, %.5f)  features=%s",
        bucket_sizes, p33, p66, list(X.columns),
    )
    return models, (p33, p66), metrics


def get_predictions_ensemble(
    models: dict,
    thresholds: tuple[float, float],
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Route each bar to its vol-bucket model and return unified predictions.

    Args:
        models:     ``{"low": model, "med": model, "high": model}``
        thresholds: ``(p33, p66)`` from training.
        X:          Feature DataFrame -- must contain ``atr_normalized``.

    Returns:
        predictions:  int array of class labels.
        probabilities: float array of class-1 probabilities.
    """
    p33, p66 = thresholds
    buckets = assign_vol_bucket(X["atr_normalized"].values, p33, p66)

    predictions   = np.zeros(len(X), dtype=int)
    probabilities = np.full(len(X), 0.5, dtype=float)  # default 0.5 for bad rows

    # NaN/Inf guard: rows with corrupted features default to 0.5 (no signal)
    bad_mask = np.any(np.isnan(X.values) | np.isinf(X.values), axis=1)
    if bad_mask.any():
        logger.warning(
            "get_predictions_ensemble: %d row(s) contain NaN/Inf -- "
            "defaulting to prob=0.5 for those rows. "
            "Check feature pipeline for upstream errors.",
            int(bad_mask.sum()),
        )

    clean_mask = ~bad_mask
    for bucket in VOL_BUCKETS:
        mask = (buckets == bucket) & clean_mask
        if not mask.any():
            continue
        model = models[bucket]
        # Select only the columns the bucket model was trained on
        # (gmm_vol_cluster is excluded -- see train_xgb_ensemble)
        bucket_cols = list(model.feature_names_in_)
        X_b = X[mask][bucket_cols]
        predictions[mask]   = model.predict(X_b)
        probabilities[mask] = model.predict_proba(X_b)[:, 1]

    return predictions, probabilities


def compute_regime_stats(
    models: dict,
    thresholds: tuple[float, float],
    X_is: pd.DataFrame,
    hmm_states_is: np.ndarray,
) -> dict:
    """Compute per-HMM-state probability statistics from In-Sample data.

    These statistics calibrate the :class:`~src.signal_evaluator.SignalEvaluator`
    Z-Score thresholds.  Call on IS data only (no look-ahead into OOS bars).

    Args:
        models:       ``{"low": xgb, "med": xgb, "high": xgb}`` trained ensemble.
        thresholds:   ``(p33, p66)`` ATR percentile from IS data.
        X_is:         IS feature DataFrame (same index slice as used for training).
        hmm_states_is: HMM state labels aligned with ``X_is``.

    Returns:
        Mapping of ``state_id -> {"mean": float, "std": float, "count": int}``.
    """
    _, probs = get_predictions_ensemble(models, thresholds, X_is)

    regime_stats: dict = {}
    for state in sorted(np.unique(hmm_states_is)):
        mask     = hmm_states_is == state
        n        = int(mask.sum())
        state_p  = probs[mask]
        if n >= 30:
            mean = float(np.mean(state_p))
            std  = float(max(np.std(state_p), 0.010))
        else:
            logger.warning(
                "State %d: only %d IS samples -- using fallback stats (mean=0.50, std=0.15)",
                state, n,
            )
            mean, std = 0.50, 0.15
        regime_stats[int(state)] = {"mean": mean, "std": std, "count": n}
        logger.info(
            "  Regime stats  state=%d  mean=%.4f  std=%.4f  n=%d", state, mean, std, n
        )
    return regime_stats


# -- Regime-Specific Training (Phase C/D rebuild) --------------------------------

def train_regime_models(
    X,
    y,
    states,
    tf='H1',
    **xgb_kwargs
):
    # Train separate XGBoost models for TREND and VOLATILITY_SHOCK regimes.
    # MEAN_REVERSION bars are excluded (no-trade policy per STATE_POLICY).
    from src.engine_hmm import CANONICAL_REGIME_ID, REGIME_TREND, REGIME_SHOCK

    TREND_ID = CANONICAL_REGIME_ID[REGIME_TREND]   # 0
    SHOCK_ID = CANONICAL_REGIME_ID[REGIME_SHOCK]   # 2
    _MIN_SAMPLES = 50

    states_arr = __import__('numpy').asarray(states)
    if len(states_arr) != len(X):
        logger.warning(
            'train_regime_models: states length %d != X length %d; truncating.',
            len(states_arr), len(X)
        )
        _n = min(len(states_arr), len(X))
        states_arr = states_arr[:_n]
        X = X.iloc[:_n]
        y = y.iloc[:_n]

    results = {
        'trend_model': None, 'shock_model': None,
        'trend_n': 0,        'shock_n': 0,
        'trend_metrics': {}, 'shock_metrics': {},
    }

    # TREND model
    trend_mask = (states_arr == TREND_ID)
    results['trend_n'] = int(trend_mask.sum())
    if results['trend_n'] >= _MIN_SAMPLES:
        try:
            tm, tm_metrics = train_xgb(X[trend_mask], y[trend_mask], train_ratio=1.0, training_context="regime_split", **xgb_kwargs)
            results['trend_model']   = tm
            results['trend_metrics'] = tm_metrics
            logger.info('[%s] TREND model: n=%d  train_acc=%.4f',
                        tf, results['trend_n'], tm_metrics.get('train_accuracy', 0.0))
        except Exception as exc:
            logger.warning('[%s] TREND model training failed: %s', tf, exc)
    else:
        logger.warning('[%s] TREND model skipped: only %d samples (min %d)',
                       tf, results['trend_n'], _MIN_SAMPLES)

    # SHOCK model
    shock_mask = (states_arr == SHOCK_ID)
    results['shock_n'] = int(shock_mask.sum())
    if results['shock_n'] >= _MIN_SAMPLES:
        try:
            sm, sm_metrics = train_xgb(X[shock_mask], y[shock_mask], train_ratio=1.0, training_context="regime_split", **xgb_kwargs)
            results['shock_model']   = sm
            results['shock_metrics'] = sm_metrics
            logger.info('[%s] SHOCK model: n=%d  train_acc=%.4f',
                        tf, results['shock_n'], sm_metrics.get('train_accuracy', 0.0))
        except Exception as exc:
            logger.warning('[%s] SHOCK model training failed: %s', tf, exc)
    else:
        logger.warning('[%s] SHOCK model skipped: only %d samples (min %d)',
                       tf, results['shock_n'], _MIN_SAMPLES)

    return results


def get_regime_predictions(X, states, trend_model, shock_model, fallback_prob=0.50):
    # Merge per-regime predictions. TREND->trend_model, SHOCK->shock_model, MR->0.5
    import numpy as _np
    from src.engine_hmm import CANONICAL_REGIME_ID, REGIME_TREND, REGIME_SHOCK

    TREND_ID = CANONICAL_REGIME_ID[REGIME_TREND]
    SHOCK_ID = CANONICAL_REGIME_ID[REGIME_SHOCK]

    probs = _np.full(len(X), fallback_prob, dtype=_np.float64)
    states_arr = _np.asarray(states)

    trend_mask = (states_arr == TREND_ID)
    if trend_model is not None and trend_mask.any():
        probs[trend_mask] = trend_model.predict_proba(X[trend_mask])[:, 1]

    shock_mask = (states_arr == SHOCK_ID)
    if shock_model is not None and shock_mask.any():
        probs[shock_mask] = shock_model.predict_proba(X[shock_mask])[:, 1]

    return probs


def predict_by_regime(regime_label: str, row_features, models: dict) -> float | None:
    """Route a single-row probability request to the matching regime model.

    Returns None when no model is available for the requested regime.
    """
    if regime_label == "TREND" and models.get("TREND") is not None:
        return float(models["TREND"].predict_proba(row_features)[0, 1])
    if regime_label == "VOLATILITY_SHOCK" and models.get("VOLATILITY_SHOCK") is not None:
        return float(models["VOLATILITY_SHOCK"].predict_proba(row_features)[0, 1])
    return None


def regime_first_probability(
    features_df,
    hmm_state: int,
    tf: str,
    broker: str = "headway_cent",
    trend_model=None,
    shock_model=None,
    ensemble_models: dict | None = None,
    ensemble_thresholds: tuple[float, float] | None = None,
    fallback_prob: float = 0.50,
) -> tuple[float, str]:
    """Return single-row probability using regime models first, then ensemble fallback.

    Routing policy:
    - TREND / VOLATILITY_SHOCK: use matching regime model when available.
    - MEAN_REVERSION: return neutral fallback_prob.
    - If a tradeable regime model is unavailable, fall back to ensemble.
    """
    from src.engine_hmm import STATE_NAMES_3

    regime_label = STATE_NAMES_3.get(int(hmm_state), "MEAN_REVERSION")

    if trend_model is None and shock_model is None:
        trend_model, shock_model = load_regime_models(tf=tf, broker=broker)
    regime_models = {
        "TREND": trend_model,
        "VOLATILITY_SHOCK": shock_model,
    }

    prob = predict_by_regime(regime_label, features_df, regime_models)
    if prob is not None:
        return float(prob), "regime_model"

    if regime_label == "MEAN_REVERSION":
        return float(fallback_prob), "mr_neutral"

    if ensemble_models is None or ensemble_thresholds is None:
        xgb_path = get_ensemble_path(tf, broker)
        if not xgb_path.exists():
            xgb_path = ENSEMBLE_PKL_PATH
        if not xgb_path.exists():
            raise FileNotFoundError(
                f"No ensemble model found for fallback at {get_ensemble_path(tf, broker)} or {ENSEMBLE_PKL_PATH}"
            )
        ensemble_models, ensemble_thresholds, _ = load_xgb_ensemble(xgb_path)

    _, probs = get_predictions_ensemble(ensemble_models, ensemble_thresholds, features_df)
    return float(probs[0]), "ensemble_fallback"


def regime_first_probabilities(
    X,
    states,
    tf: str,
    broker: str = "headway_cent",
    trend_model=None,
    shock_model=None,
    ensemble_models: dict | None = None,
    ensemble_thresholds: tuple[float, float] | None = None,
    fallback_prob: float = 0.50,
) -> tuple[np.ndarray, str]:
    """Return batch probabilities with regime models preferred over ensemble.

    If at least one regime model exists, probabilities are generated via
    get_regime_predictions (MR and missing model paths stay neutral at fallback_prob).
    If no regime models exist, falls back to ensemble probabilities.
    """
    if trend_model is None and shock_model is None:
        trend_model, shock_model = load_regime_models(tf=tf, broker=broker)

    if trend_model is not None or shock_model is not None:
        probs = get_regime_predictions(
            X=X,
            states=states,
            trend_model=trend_model,
            shock_model=shock_model,
            fallback_prob=fallback_prob,
        )
        return probs, "regime_models"

    if ensemble_models is None or ensemble_thresholds is None:
        xgb_path = get_ensemble_path(tf, broker)
        if not xgb_path.exists():
            xgb_path = ENSEMBLE_PKL_PATH
        if not xgb_path.exists():
            raise FileNotFoundError(
                f"No regime models or ensemble fallback found for {tf}/{broker}."
            )
        ensemble_models, ensemble_thresholds, _ = load_xgb_ensemble(xgb_path)

    _, probs = get_predictions_ensemble(ensemble_models, ensemble_thresholds, X)
    return probs, "ensemble_fallback"


def save_regime_models(regime_result, tf, broker='headway_cent'):
    # Persist trend_model and shock_model to models/ directory.
    for key in ('trend_model', 'shock_model'):
        model = regime_result.get(key)
        if model is not None:
            path = Path('models/%s_%s_%s.pkl' % (key, tf.upper(), broker))
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
            logger.info('Saved %s -> %s', key, path)


def load_regime_models(tf, broker='headway_cent'):
    # Load trend_model and shock_model from models/ directory. Returns (trend, shock).
    models = []
    for key in ('trend_model', 'shock_model'):
        path = Path('models/%s_%s_%s.pkl' % (key, tf.upper(), broker))
        if path.exists():
            m = joblib.load(path)
            logger.info('Loaded %s <- %s', key, path)
        else:
            m = None
            logger.warning('%s not found at %s', key, path)
        models.append(m)
    return tuple(models)


def save_xgb_ensemble(
    models: dict,
    thresholds: tuple[float, float],
    metrics: dict,
    path: Path = ENSEMBLE_PKL_PATH,
) -> None:
    """Persist the ensemble (3 models + thresholds + metadata) to a single pkl.

    ``metrics`` may include a ``"regime_stats"`` key (added by callers after
    :func:`compute_regime_stats`) which is stored transparently alongside the
    model weights and loaded back by :func:`load_xgb_ensemble`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "thresholds": thresholds, "metrics": metrics}, path)
    logger.info("Ensemble saved to %s", path)


def load_xgb_ensemble(
    path: Path = ENSEMBLE_PKL_PATH,
) -> tuple[dict, tuple[float, float], dict]:
    """Load the ensemble pkl and return ``(models, thresholds, metrics)``."""
    data = joblib.load(path)
    return data["models"], data["thresholds"], data["metrics"]


def export_onnx_ensemble(
    models: dict,
    n_features: int,
    base_dir: Path = Path("models"),
) -> dict[str, Path]:
    """Export all three vol-bucket models to individual ONNX files.

    Output filenames:
        ``models/xgb_model_vol_low.onnx``
        ``models/xgb_model_vol_med.onnx``
        ``models/xgb_model_vol_high.onnx``

    Returns a dict mapping bucket name -> ONNX path.
    """
    base_dir = Path(base_dir)
    paths = {}
    for bucket, model in models.items():
        path = base_dir / f"xgb_model_vol_{bucket}.onnx"
        export_onnx(model, n_features=n_features, path=path)
        paths[bucket] = path
    return paths


# -----------------------------------------------------------------------------
# Per-regime XGBoost classifiers
# -----------------------------------------------------------------------------

def train_regime_classifiers(
    X: pd.DataFrame,
    hmm_states: np.ndarray,
    train_ratio: float = 0.70,
    **xgb_kwargs,
) -> dict:
    """Train 3 binary XGBoost classifiers for regime-conditional probability.

    Each classifier predicts whether the NEXT bar will be in regime r:
    - Classifier 0 (Bull):  y=1 if next_state == 0
    - Classifier 1 (Bear):  y=1 if next_state == 1
    - Classifier 2 (Chop):  y=1 if next_state IN {2, 3}  (4-state models)
                                 or next_state == 2         (3-state models)

    The three raw "stay" probabilities are later normalized to a distribution
    via :func:`predict_regime_proba`.

    Args:
        X:           Feature DataFrame (output of ``prepare_features``).
        hmm_states:  HMM state labels aligned row-for-row with X.
        train_ratio: IS fraction for fitting (default 0.70 = H1 ratio).
        **xgb_kwargs: Forwarded to XGBClassifier (max_depth, learning_rate...).

    Returns:
        ``{0: xgb_model, 1: xgb_model, 2: xgb_model}``
    """
    states = np.asarray(hmm_states, dtype=int)
    n_unique = len(np.unique(states))
    next_states = np.roll(states, -1)   # next-bar state; last entry is invalid

    # Drop last row -- no valid next_state
    X_fit = X.iloc[:-1]
    next_s = next_states[:-1]

    split_idx = int(len(X_fit) * train_ratio)
    X_is = X_fit.iloc[:split_idx]
    ns_is = next_s[:split_idx]

    default_xgb = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    default_xgb.update(xgb_kwargs)

    regime_models: dict = {}
    regime_names = {0: "Bull", 1: "Bear", 2: "Chop"}

    for regime_id, regime_name in regime_names.items():
        if regime_id == 2:
            if n_unique == 4:
                y_is = ((ns_is == 2) | (ns_is == 3)).astype(int)
            else:
                y_is = (ns_is == 2).astype(int)
        else:
            y_is = (ns_is == regime_id).astype(int)

        pos_ratio = y_is.mean()
        if pos_ratio <= 0 or pos_ratio >= 1:
            logger.warning(
                "Regime classifier [%s]: degenerate class ratio=%.3f -- using scale_pos_weight=1",
                regime_name, pos_ratio,
            )
            scale_pos_weight = 1.0
        else:
            scale_pos_weight = (1.0 - pos_ratio) / pos_ratio

        params = {**default_xgb, "scale_pos_weight": scale_pos_weight}
        model = xgb.XGBClassifier(**params)
        model.fit(X_is, y_is)

        # Quick validation accuracy on held-out OOS fraction
        if split_idx < len(X_fit):
            X_oos = X_fit.iloc[split_idx:]
            ns_oos = next_s[split_idx:]
            if regime_id == 2:
                y_oos = ((ns_oos == 2) | (ns_oos == 3)).astype(int) if n_unique == 4 else (ns_oos == 2).astype(int)
            else:
                y_oos = (ns_oos == regime_id).astype(int)
            baseline = max(float(y_oos.mean()), 1.0 - float(y_oos.mean()))
            val_acc = float(accuracy_score(y_oos, model.predict(X_oos)))
            logger.info(
                "  Regime classifier [%s]: val_acc=%.3f  baseline=%.3f  improvement=%.1f%%",
                regime_name, val_acc, baseline, (val_acc / baseline - 1) * 100,
            )

        regime_models[regime_id] = model

    logger.info("Per-regime XGBoost classifiers trained (n_unique_states=%d)", n_unique)
    return regime_models


def predict_regime_proba(
    regime_models: dict | None,
    X_row,
) -> dict:
    """Return normalised Bull/Bear/Chop probability distribution for one bar.

    Args:
        regime_models: ``{0: xgb, 1: xgb, 2: xgb}`` from :func:`train_regime_classifiers`,
                       or ``None`` for a uniform fallback.
        X_row:         Single-row feature input -- DataFrame row, Series, or 1-D array.

    Returns:
        ``{'Bull': float, 'Bear': float, 'Chop': float}``  (values sum to 1.0)
    """
    if not regime_models:
        return {"Bull": 0.333, "Bear": 0.333, "Chop": 0.333}

    # Reshape to 2-D for predict_proba
    if hasattr(X_row, "values"):
        arr = X_row.values.reshape(1, -1)
    else:
        arr = np.asarray(X_row, dtype=float).reshape(1, -1)

    raw: dict = {}
    for regime_id, model in regime_models.items():
        proba = model.predict_proba(arr)[0]
        raw[regime_id] = float(proba[1]) if len(proba) > 1 else float(proba[0])

    total = sum(raw.values())
    if total <= 0:
        return {"Bull": 0.333, "Bear": 0.333, "Chop": 0.333}

    return {
        "Bull": raw[0] / total,
        "Bear": raw[1] / total,
        "Chop": raw[2] / total,
    }


def save_regime_classifiers(regime_models: dict, base_path: Path) -> None:
    """Save per-regime classifiers alongside the ensemble pkl.

    File: ``{base_path stem}_regime_classifiers.pkl``
    """
    base_path = Path(base_path)
    out_path = base_path.parent / (base_path.stem + "_regime_classifiers.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(regime_models, out_path)
    logger.info("Regime classifiers saved to %s", out_path)


def load_regime_classifiers(base_path: Path) -> dict | None:
    """Load per-regime classifiers if they exist, else return None."""
    base_path = Path(base_path)
    pkl_path = base_path.parent / (base_path.stem + "_regime_classifiers.pkl")
    if not pkl_path.exists():
        return None
    regime_models = joblib.load(pkl_path)
    logger.info("Regime classifiers loaded from %s", pkl_path)
    return regime_models
