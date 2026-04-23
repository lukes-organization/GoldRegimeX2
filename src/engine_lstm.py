"""LSTM Context Layer — self-supervised sequence model that adds 4 context features.

The model is a **regressor** that predicts the next bar's continuous log return.
A 4-neuron tanh bottleneck ("context_layer") is extracted and injected into the
XGBoost feature matrix as lstm_ctx_0..3.

Architecture:
    Input(100, n_feats) → LSTM(128)+Dropout → LSTM(64)+Dropout
    → Dense(32,relu)+Dropout → Dense(4,tanh,"context_layer") → Dense(1,linear)

Path convention:
    models/lstm_context_{TF}_{broker}.keras      — weights + architecture
    models/lstm_context_{TF}_{broker}.norm.json  — normalisation + target scaler

Usage:
    from src.engine_lstm import LSTMContextModel, load_lstm_model, get_lstm_path

    model = LSTMContextModel()
    model.fit(df, epochs=100)
    model.save(get_lstm_path("H1", "headway_cent"))

    model = load_lstm_model("H1", "headway_cent")
    ctx = model.predict_context(window_df_or_array)   # → np.ndarray shape (4,)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.logger import setup_logger

logger = setup_logger(__name__)

LSTM_WINDOW      = 100   # bars of look-back (≈4 days H1, ≈1 day M15)
LSTM_N_CONTEXT   = 4     # bottleneck neurons exported as XGBoost features
LSTM_CONTEXT_COLS = [f"lstm_ctx_{i}" for i in range(LSTM_N_CONTEXT)]

# Base input features always expected in the processed DataFrame.
# Derived features (rsi_normalized, volume_ratio, bb_position) are computed
# on-the-fly by _add_derived_features().  hmm_state / gmm_vol_cluster are
# optional — included when present (caller must add hmm_state to df before
# calling fit() if it should be used; gmm_vol_cluster is already in the parquet).
_BASE_FEATS   = ["log_return", "volatility", "rsi_normalized",
                 "atr_normalized", "volume_ratio", "bb_position"]
_OPTIONAL_FEATS = ["gmm_vol_cluster", "hmm_state"]

# Kept for backward-compat imports from processor / mt5_trader
LSTM_INPUT_FEATS = _BASE_FEATS   # updated at runtime by _get_active_features()


def get_lstm_path(tf: str, broker: str = "headway_cent") -> Path:
    return Path(f"models/lstm_context_{tf.upper()}_{broker}.keras")


def _norm_path(keras_path: Path) -> Path:
    return keras_path.with_suffix(".norm.json")


# ── Derived feature helpers ────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LSTM-specific derived columns on a copy of ``df``.

    Adds: rsi_normalized, volume_ratio, bb_position
    Requires columns present from process_pipeline: rsi, Volume, Close.
    """
    df = df.copy()

    if "rsi" in df.columns:
        df["rsi_normalized"] = df["rsi"] / 100.0

    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20, min_periods=1).mean()
        df["volume_ratio"] = df["Volume"] / vol_ma.replace(0, np.nan).fillna(1)

    if "Close" in df.columns:
        ma     = df["Close"].rolling(20, min_periods=1).mean()
        std    = df["Close"].rolling(20, min_periods=1).std().fillna(0)
        bb_rng = (4 * std).replace(0, np.nan)
        df["bb_position"] = ((df["Close"] - (ma - 2 * std)) / bb_rng).clip(0, 1).fillna(0.5)

    return df


def _get_active_features(df: pd.DataFrame) -> list[str]:
    """Return the feature columns actually present in ``df``."""
    candidates = _BASE_FEATS + _OPTIONAL_FEATS
    return [c for c in candidates if c in df.columns]


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMContextModel:
    """Self-supervised LSTM context extractor (regressor variant).

    Predicts the next bar's scaled log return (MSE loss).
    The context_layer bottleneck is the feature product exported to XGBoost.
    """

    def __init__(self, sequence_length: int = LSTM_WINDOW, context_dim: int = LSTM_N_CONTEXT):
        self.sequence_length = sequence_length
        self.context_dim     = context_dim
        self._model          = None
        self._extractor      = None
        # Feature normalisation (per-column means / stds)
        self._means: np.ndarray | None = None
        self._stds:  np.ndarray | None = None
        # Target scaler (StandardScaler fit on training targets)
        self._target_mean: float = 0.0
        self._target_std:  float = 1.0
        # Feature columns used at training time
        self._feature_cols: list[str] = []

    # ── Build ── ──────────────────────────────────────────────────────────────

    def _build_model(self, n_feats: int):
        from keras import Input
        from keras.layers import LSTM, Dense, Dropout
        from keras.models import Model
        from keras.optimizers import Adam

        inp = Input(shape=(self.sequence_length, n_feats), name="price_sequence")
        x   = LSTM(128, return_sequences=True, name="lstm_1")(inp)
        x   = Dropout(0.3, name="dropout_1")(x)
        x   = LSTM(64, return_sequences=False, name="lstm_2")(x)
        x   = Dropout(0.3, name="dropout_2")(x)
        x   = Dense(32, activation="relu", name="dense_1")(x)
        x   = Dropout(0.2, name="dropout_3")(x)
        ctx = Dense(self.context_dim, activation="tanh", name="context_layer")(x)
        out = Dense(1, activation="linear", name="output")(ctx)

        model = Model(inputs=inp, outputs=out, name="lstm_context")
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def _attach_extractor(self):
        import keras
        self._extractor = keras.Model(
            inputs=self._model.input,
            outputs=self._model.get_layer("context_layer").output,
            name="context_extractor",
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.1,
    ) -> None:
        """Train on a featurised DataFrame produced by process_pipeline().

        hmm_state and gmm_vol_cluster are used when present in ``df``.
        Derived features (rsi_normalized, volume_ratio, bb_position) are
        computed automatically from base columns (rsi, Volume, Close).
        """
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        df_prep = _add_derived_features(df)
        feat_cols = _get_active_features(df_prep)
        self._feature_cols = feat_cols

        X_seq, y_seq, means, stds, t_mean, t_std = _build_sequences(
            df_prep, feat_cols, self.sequence_length
        )
        if len(X_seq) < self.sequence_length * 2:
            raise ValueError(
                f"Insufficient data: {len(X_seq)} sequences (need "
                f"≥ {self.sequence_length * 2})."
            )

        self._means       = means
        self._stds        = stds
        self._target_mean = t_mean
        self._target_std  = t_std

        self._model = self._build_model(len(feat_cols))
        self._attach_extractor()

        logger.info(
            "LSTM training (regressor): %d sequences  features=%s  "
            "epochs=%d  batch=%d  val=%.0f%%",
            len(X_seq), feat_cols, epochs, batch_size, validation_split * 100,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                min_delta=0.0001,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        self._model.fit(
            X_seq,
            y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        if self._means is None or self._stds is None:
            return arr.astype(np.float32)
        return ((arr - self._means) / self._stds).astype(np.float32)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_context(self, window) -> np.ndarray:
        """Return the (4,) context vector for a window of bars.

        Args:
            window: DataFrame or ndarray of shape (N, n_feats) or (N,).
                    Must contain (or be aligned to) the columns used at training.
                    Padded with zeros at the front when N < sequence_length.

        Returns:
            np.ndarray shape (4,) in [-1, 1].  Zeros on error (graceful degradation).
        """
        if self._extractor is None:
            return np.zeros(LSTM_N_CONTEXT, dtype=np.float32)
        try:
            if isinstance(window, pd.DataFrame):
                window = _add_derived_features(window)
                cols   = [c for c in self._feature_cols if c in window.columns]
                arr    = window[cols].values.astype(np.float64)
                # Fill any missing trained columns with zeros
                if len(cols) < len(self._feature_cols):
                    full = np.zeros(
                        (arr.shape[0], len(self._feature_cols)), dtype=np.float64
                    )
                    for i, c in enumerate(self._feature_cols):
                        if c in cols:
                            full[:, i] = arr[:, cols.index(c)]
                    arr = full
            else:
                arr = np.array(window, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = self._normalise(arr)

            if arr.shape[0] < self.sequence_length:
                pad = np.zeros(
                    (self.sequence_length - arr.shape[0], arr.shape[1]),
                    dtype=np.float32,
                )
                arr = np.vstack([pad, arr])
            else:
                arr = arr[-self.sequence_length:]

            return self._extractor.predict(arr[np.newaxis], verbose=0)[0]
        except Exception as exc:
            logger.debug("LSTM predict_context error: %s", exc)
            return np.zeros(LSTM_N_CONTEXT, dtype=np.float32)

    def predict_context_batch(self, windows: np.ndarray) -> np.ndarray:
        """Batch predict context vectors.

        Args:
            windows: Already-normalised array shape (N, sequence_length, n_feats).
        """
        if self._extractor is None:
            return np.zeros((len(windows), LSTM_N_CONTEXT), dtype=np.float32)
        try:
            return self._extractor.predict(windows, verbose=0, batch_size=512)
        except Exception as exc:
            logger.debug("LSTM predict_context_batch error: %s", exc)
            return np.zeros((len(windows), LSTM_N_CONTEXT), dtype=np.float32)

    def log_context_quality(self, df: pd.DataFrame, n_samples: int = 1000) -> None:
        """Log context vector variance over the last ``n_samples`` bars.

        Collapsed context (all std < 0.1) means the model didn't learn useful
        patterns and lstm_ctx_* features will not benefit XGBoost.
        """
        if self._extractor is None:
            logger.warning("LSTM context quality check: model not loaded.")
            return

        df_prep   = _add_derived_features(df)
        arr       = df_prep[self._feature_cols].values.astype(np.float64)
        arr       = np.nan_to_num(arr, nan=0.0)
        arr       = self._normalise(arr)
        end       = len(arr)
        start     = max(self.sequence_length, end - n_samples)

        contexts  = []
        for i in range(start, end):
            w = arr[i - self.sequence_length: i]
            p = self._extractor.predict(w[np.newaxis], verbose=0)[0]
            contexts.append(p)

        ctxs = np.array(contexts)
        logger.info("Context vector quality check over last %d bars:", len(ctxs))
        any_collapsed = False
        for i in range(self.context_dim):
            col = ctxs[:, i]
            logger.info(
                "  ctx_%d: mean=%+.4f  std=%.4f  range=[%+.4f, %+.4f]",
                i, col.mean(), col.std(), col.min(), col.max(),
            )
            if col.std() < 0.1:
                any_collapsed = True
        if any_collapsed:
            logger.warning(
                "One or more context dimensions have std < 0.1 — the context "
                "layer may be collapsed.  XGBoost will likely ignore these "
                "features.  Consider retraining with more epochs or checking "
                "that input features have sufficient variance."
            )
        else:
            logger.info("Context quality OK — all dimensions show meaningful variance.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Model not built or trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        norm = {
            "means":        (self._means.tolist()  if self._means  is not None else None),
            "stds":         (self._stds.tolist()   if self._stds   is not None else None),
            "target_mean":  float(self._target_mean),
            "target_std":   float(self._target_std),
            "feature_cols": self._feature_cols,
            "sequence_length": self.sequence_length,
            "context_dim":     self.context_dim,
        }
        _norm_path(path).write_text(json.dumps(norm))
        logger.info("LSTM context model saved: %s  (+ .norm.json)", path)

    def load(self, path: Path) -> "LSTMContextModel":
        import keras
        path = Path(path)
        self._model = keras.models.load_model(str(path))
        self._attach_extractor()

        np_path = _norm_path(path)
        if np_path.exists():
            norm = json.loads(np_path.read_text())
            if norm.get("means") is not None:
                self._means = np.array(norm["means"], dtype=np.float32)
                self._stds  = np.array(norm["stds"],  dtype=np.float32)
            self._target_mean    = float(norm.get("target_mean", 0.0))
            self._target_std     = float(norm.get("target_std",  1.0))
            self._feature_cols   = norm.get("feature_cols", _BASE_FEATS)
            self.sequence_length = int(norm.get("sequence_length", LSTM_WINDOW))
            self.context_dim     = int(norm.get("context_dim",     LSTM_N_CONTEXT))
        else:
            logger.warning(
                "LSTM norm stats not found at %s — inference scaling may be wrong. "
                "Re-train with --mode train_lstm.", np_path,
            )
        logger.info(
            "LSTM context model loaded: %s  features=%s", path, self._feature_cols
        )
        return self


# ── Sequence builder ──────────────────────────────────────────────────────────

def _build_sequences(
    df: pd.DataFrame,
    feat_cols: list[str],
    sequence_length: int,
) -> tuple:
    """Build (X_seq, y_seq, means, stds, target_mean, target_std).

    Target is the next bar's log return (continuous), scaled by its IS std so
    the MSE loss is on a standardised scale.  Sequence inputs are normalised
    per-column by global mean/std.

    Returns:
        X_seq        (N, sequence_length, n_feats)  float32  normalised
        y_seq        (N,)  scaled next-bar log return  float32
        means        (n_feats,)  feature column means
        stds         (n_feats,)  feature column stds  (floor 1e-9)
        target_mean  float
        target_std   float
    """
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"_build_sequences: missing columns {missing}")

    arr = df[feat_cols].values.astype(np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    means = arr.mean(axis=0).astype(np.float32)
    stds  = arr.std(axis=0).astype(np.float32)
    stds  = np.where(stds > 1e-9, stds, np.float32(1.0))
    arr   = ((arr - means) / stds).astype(np.float32)

    # Continuous target: next-bar log return scaled by its std
    targets = df["log_return"].shift(-1).values.astype(np.float64)
    targets = np.nan_to_num(targets, nan=0.0)

    n_unique = len(np.unique(targets[targets != 0.0][:500]))
    logger.info(
        "LSTM target unique sample values: %d  (should be >> 2 for regression)", n_unique
    )
    if n_unique <= 2:
        logger.error(
            "Target appears binary — check that log_return is continuous in the DataFrame!"
        )

    t_mean = float(np.nanmean(targets))
    t_std  = float(np.nanstd(targets))
    if t_std < 1e-9:
        t_std = 1.0
    targets = ((targets - t_mean) / t_std).astype(np.float32)

    n    = len(arr)
    seqs = []
    ys   = []
    for i in range(sequence_length, n - 1):
        seqs.append(arr[i - sequence_length: i])
        ys.append(targets[i + 1])

    return (
        np.array(seqs,  dtype=np.float32),
        np.array(ys,    dtype=np.float32),
        means, stds,
        t_mean, t_std,
    )


# ── Pipeline integration ───────────────────────────────────────────────────────

def add_lstm_context(df: pd.DataFrame, model: "LSTMContextModel | None") -> None:
    """Add lstm_ctx_0..3 columns to ``df`` in-place.

    No-op (columns NOT added) when ``model`` is None — get_feature_cols() will
    then exclude them and XGBoost trains without LSTM context.
    """
    if model is None or model._extractor is None:
        logger.debug("LSTM context: no model — skipping context columns.")
        return

    df_prep   = _add_derived_features(df)
    feat_cols = model._feature_cols or _get_active_features(df_prep)
    missing = [c for c in feat_cols if c not in df_prep.columns]
    if missing:
        logger.warning(
            "LSTM context: missing columns %s — skipping context columns.", missing
        )
        return

    n   = len(df_prep)
    arr = df_prep[feat_cols].values.astype(np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = model._normalise(arr)

    sl      = model.sequence_length
    windows = np.zeros((n, sl, len(feat_cols)), dtype=np.float32)
    for i in range(n):
        start = max(0, i - sl)
        chunk = arr[start: i]
        if len(chunk) > 0:
            windows[i, sl - len(chunk):] = chunk

    ctx = model.predict_context_batch(windows)  # (n, 4)

    for j, col in enumerate(LSTM_CONTEXT_COLS):
        df[col] = ctx[:, j]

    logger.info(
        "LSTM context: %d context vectors computed (window=%d  features=%s).",
        n, sl, feat_cols,
    )


def load_lstm_model(
    tf: str, broker: str = "headway_cent"
) -> "LSTMContextModel | None":
    """Load a saved LSTM context model, or return None if not found."""
    path = get_lstm_path(tf, broker)
    if not path.exists():
        logger.debug("LSTM context model not found at %s.", path)
        return None
    try:
        return LSTMContextModel().load(path)
    except Exception as exc:
        logger.warning(
            "Failed to load LSTM model from %s: %s — "
            "continuing without LSTM context features.", path, exc,
        )
        return None
