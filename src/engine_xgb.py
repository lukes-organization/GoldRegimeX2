import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import onnx
from pathlib import Path
from sklearn.metrics import accuracy_score
from src.logger import setup_logger

logger = setup_logger(__name__)

ONNX_PATH        = Path("models/xgb_model.onnx")
XGB_PKL_PATH     = Path("models/xgb_model.pkl")
ENSEMBLE_PKL_PATH = Path("models/xgb_ensemble.pkl")

# Base features always present; dxy_log_return is added when DXY data is available.
FEATURE_COLS     = ["hmm_state", "rsi_slope", "atr_normalized", "prev_log_return"]
DXY_FEATURE      = "dxy_log_return"

# Volatility bucket labels (ATR tertiles: low / med / high)
VOL_BUCKETS = ["low", "med", "high"]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the feature column list for this DataFrame.

    Includes ``dxy_log_return`` only when the column is present and has enough
    non-null values to be useful (>50% coverage).
    """
    cols = list(FEATURE_COLS)
    if DXY_FEATURE in df.columns and df[DXY_FEATURE].notna().mean() > 0.5:
        cols.append(DXY_FEATURE)
    return cols


def prepare_features(df: pd.DataFrame, hmm_states: np.ndarray):
    df = df.copy()
    df["hmm_state"] = hmm_states
    df["prev_log_return"] = df["log_return"].shift(1)
    y = (df["log_return"].shift(-1) > 0).astype(int).rename("target")

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]
    df_aligned = df.loc[X.index]

    logger.info("Features prepared: %d samples, %d features: %s", len(X), len(feature_cols), feature_cols)
    return X, y, df_aligned


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
    colsample_bytree: float = 0.8,
    train_ratio: float = 0.8,
):
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    importance = dict(zip(list(X.columns), model.feature_importances_))

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
        return onnx_model  # no ZipMap present — nothing to do

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
    print(f"\n  ONNX export OK — n_classes={n_classes}. "
          f"Set NStates={n_classes} in the MT5 EA inputs.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Three-model volatility-regime ensemble
# ─────────────────────────────────────────────────────────────────────────────

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

    Data is split temporally into IS (first ``train_ratio`` fraction) and OOS.
    ATR percentile thresholds are computed on IS data only to prevent
    look-ahead bias.  Each bucket model is trained on its IS-subset bars.

    Args:
        X:           Feature DataFrame (output of ``prepare_features``).
        y:           Target Series.
        train_ratio: Fraction of data used for IS training (default 0.8).
        **xgb_kwargs: Forwarded to ``train_xgb`` for each bucket model.

    Returns:
        models:      ``{"low": model, "med": model, "high": model}``
        thresholds:  ``(p33, p66)`` ATR tertile boundaries from IS data.
        metrics:     Aggregate metrics dict including ``split_idx``,
                     ``vol_thresholds``, and ``feature_cols``.
    """
    split_idx = int(len(X) * train_ratio)
    X_is = X.iloc[:split_idx]
    y_is = y.iloc[:split_idx]

    # Thresholds from IS only — no look-ahead into OOS bars
    p33, p66 = compute_vol_thresholds(X_is["atr_normalized"])
    buckets_is = assign_vol_bucket(X_is["atr_normalized"].values, p33, p66)

    models = {}
    bucket_sizes = {}
    for bucket in VOL_BUCKETS:
        mask = (buckets_is == bucket)
        X_b = X_is[mask]
        y_b = y_is[mask]
        bucket_sizes[bucket] = int(mask.sum())

        if len(X_b) < 100:
            # Too few samples in this bucket — fall back to all IS data
            logger.warning(
                "Vol bucket '%s' has only %d IS samples — falling back to full IS data.",
                bucket, len(X_b),
            )
            X_b, y_b = X_is, y_is

        model, _ = train_xgb(X_b, y_b, **xgb_kwargs)
        models[bucket] = model
        logger.info("Trained vol-bucket '%s': %d samples.", bucket, len(X_b))

    # Feature importance from the med bucket (most representative)
    fi = {}
    if "med" in models:
        try:
            fi = dict(zip(list(X.columns), models["med"].feature_importances_))
        except Exception:
            pass

    metrics = {
        "split_idx":        split_idx,
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
        X:          Feature DataFrame — must contain ``atr_normalized``.

    Returns:
        predictions:  int array of class labels.
        probabilities: float array of class-1 probabilities.
    """
    p33, p66 = thresholds
    buckets = assign_vol_bucket(X["atr_normalized"].values, p33, p66)

    predictions   = np.zeros(len(X), dtype=int)
    probabilities = np.zeros(len(X), dtype=float)

    for bucket in VOL_BUCKETS:
        mask = (buckets == bucket)
        if not mask.any():
            continue
        model = models[bucket]
        X_b   = X[mask]
        predictions[mask]   = model.predict(X_b)
        probabilities[mask] = model.predict_proba(X_b)[:, 1]

    return predictions, probabilities


def save_xgb_ensemble(
    models: dict,
    thresholds: tuple[float, float],
    metrics: dict,
    path: Path = ENSEMBLE_PKL_PATH,
) -> None:
    """Persist the ensemble (3 models + thresholds + metadata) to a single pkl."""
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

    Returns a dict mapping bucket name → ONNX path.
    """
    base_dir = Path(base_dir)
    paths = {}
    for bucket, model in models.items():
        path = base_dir / f"xgb_model_vol_{bucket}.onnx"
        export_onnx(model, n_features=n_features, path=path)
        paths[bucket] = path
    return paths
