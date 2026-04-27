"""Temporal Convolutional Network — dynamic Z-Score threshold adjustment.

Replaces the LSTM ensemble architecture.  Instead of classifying regime state,
the TCN directly scores how *confident* the current signal is based on the last
100 bars of market context.  Confidence is expressed as a multiplier [0.7, 1.3]
that scales the Z-Score cutoff in the live bridge:

    effective_cutoff = base_z_cutoff × confidence_multiplier

    multiplier < 1.0  → relaxed entry (strong, clear regime)
    multiplier = 1.0  → no adjustment (TCN not loaded)
    multiplier > 1.0  → tightened entry (noisy / uncertain regime)

Architecture:
    4 × dilated causal Conv1D layers (dilation 1, 2, 4, 8)
    GlobalAveragePooling → Dense(32) → sigmoid output

Training target (profit-based):
    1 if the next bar's direction matches the current HMM regime
    0 otherwise
"""

import gc
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TCN_WINDOW    = 100      # sequence length (bars)
TCN_N_STATES  = 3        # default number of HMM states
TCN_TEMP      = 1.5      # temperature for probability calibration
TCN_MAX_AGE   = 7        # days before model is considered stale
TCN_BASE_DIR  = Path("models/tcn")

_INPUT_COLUMNS = [
    "log_return",
    "volatility",
    "rsi_normalized",
    "atr_normalized",
    "volume_ratio",
    "bb_position",
    "gmm_vol_cluster",
    "dist_from_sma50",
]


# ── Feature helpers ────────────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TCN-specific derived columns on a copy of df.

    Mirrors the identical helper that was previously in engine_lstm.py.
    ``live_df`` from ``compute_live_features`` carries the raw OHLCV + computed
    base columns; this function fills the remaining TCN input columns.
    """
    df = df.copy()

    if "rsi" in df.columns:
        df["rsi_normalized"] = df["rsi"] / 100.0
    else:
        df["rsi_normalized"] = 0.5

    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20, min_periods=1).mean()
        df["volume_ratio"] = (
            df["Volume"] / vol_ma.replace(0, np.nan).fillna(1)
        ).clip(0.1, 5.0)
    else:
        df["volume_ratio"] = 1.0

    if "Close" in df.columns:
        ma     = df["Close"].rolling(20, min_periods=1).mean()
        std    = df["Close"].rolling(20, min_periods=1).std().fillna(0)
        bb_rng = (4 * std).replace(0, np.nan)
        df["bb_position"] = (
            (df["Close"] - (ma - 2 * std)) / bb_rng
        ).clip(0, 1).fillna(0.5)

        sma50 = df["Close"].rolling(50, min_periods=1).mean()
        df["dist_from_sma50"] = ((df["Close"] - sma50) / sma50).fillna(0.0)
    else:
        df["bb_position"]    = 0.5
        df["dist_from_sma50"] = 0.0

    return df


# ── Path helpers ───────────────────────────────────────────────────────────────

def get_tcn_dir(tf: str, broker: str) -> str:
    """Return the directory path for a TF/broker TCN model."""
    return str(TCN_BASE_DIR / f"{tf.upper()}_{broker}")


def load_tcn_classifier(tf: str, broker: str) -> Optional["SignalConfidenceTCN"]:
    """Load a saved TCN from ``models/tcn/{tf}_{broker}/``.

    Returns ``None`` (silently) if no model has been trained yet.
    """
    model_dir = get_tcn_dir(tf, broker)
    model_path = os.path.join(model_dir, "tcn_confidence_model.keras")
    if not os.path.exists(model_path):
        return None
    try:
        tcn = SignalConfidenceTCN()
        tcn.load(model_dir)
        logger.info(
            "TCN confidence model loaded [%s/%s] — seq_len=%d  n_states=%d"
            "  trained=%s",
            tf, broker, tcn.seq_len, tcn.n_states, tcn.trained_at,
        )
        return tcn
    except Exception as exc:
        logger.warning("Failed to load TCN [%s/%s]: %s", tf, broker, exc)
        return None


# ── Model class ────────────────────────────────────────────────────────────────

class SignalConfidenceTCN:
    """TCN that scores how trustworthy the current regime signal is.

    Output: confidence_multiplier in [0.7, 1.3].
    High raw confidence (0→1) maps to lower multiplier (easier entry).
    Low raw confidence maps to higher multiplier (harder entry / noise filter).
    """

    def __init__(
        self,
        seq_len: int     = TCN_WINDOW,
        n_features: int  = len(_INPUT_COLUMNS),
        n_states: int    = TCN_N_STATES,
        temperature: float = TCN_TEMP,
    ):
        self.seq_len     = seq_len
        self.n_features  = n_features
        self.n_states    = n_states
        self.temperature = temperature
        self.model       = None
        self.feature_scaler: Optional[object] = None   # sklearn StandardScaler
        self.trained_at: Optional[str] = None
        self.metadata: dict = {}

    # ── Architecture ─────────────────────────────────────────────────────────

    def build_model(self):
        """TCN with dilated causal convolutions — no look-ahead."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input,
        )
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Input(shape=(self.seq_len, self.n_features)),
            Conv1D(64, 3, dilation_rate=1, padding="causal", activation="relu"),
            Dropout(0.3),
            Conv1D(64, 3, dilation_rate=2, padding="causal", activation="relu"),
            Dropout(0.3),
            Conv1D(64, 3, dilation_rate=4, padding="causal", activation="relu"),
            Dropout(0.3),
            Conv1D(64, 3, dilation_rate=8, padding="causal", activation="relu"),
            GlobalAveragePooling1D(),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Training ─────────────────────────────────────────────────────────────

    def prepare_sequences(
        self,
        features_df: pd.DataFrame,
        hmm_states: pd.Series,
        returns_series: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) training pairs.

        Target = 1 if the *next* bar's log-return direction matches the
        *current* HMM regime:
            Bull (0)  + positive next return → 1
            Bear (1)  + negative next return → 1
            Chop (2+) + |return| < 0.003    → 1
        """
        feature_cols = [c for c in _INPUT_COLUMNS if c in features_df.columns]
        X, y = [], []

        for i in range(len(features_df) - self.seq_len - 1):
            seq            = features_df[feature_cols].iloc[i : i + self.seq_len].values
            current_state  = int(hmm_states.iloc[i + self.seq_len])
            next_return    = float(returns_series.iloc[i + self.seq_len + 1])

            if current_state == 0:       # Bull
                target = 1 if next_return > 0 else 0
            elif current_state == 1:     # Bear
                target = 1 if next_return < 0 else 0
            else:                        # Chop
                target = 1 if abs(next_return) < 0.003 else 0

            X.append(seq)
            y.append(target)

        return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)

    def train(
        self,
        features_df: pd.DataFrame,
        hmm_states: pd.Series,
        returns_series: pd.Series,
        epochs: int             = 100,
        batch_size: int         = 64,
        validation_split: float = 0.2,
    ):
        """Full training from scratch."""
        from sklearn.preprocessing import StandardScaler
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        features_df = _add_derived_features(features_df)
        logger.info("TCN training — %d bars", len(features_df))

        X, y = self.prepare_sequences(features_df, hmm_states, returns_series)
        logger.info("Sequences: %d  shape: %s  positive_rate: %.1f%%",
                    len(X), X.shape, 100 * y.mean())

        # Update n_features to match actual available columns (may be < 8 if
        # some columns like 'Volume' or 'rsi' are absent from the training df)
        self.n_features = X.shape[-1]

        # Fit scaler on flattened feature matrix
        self.feature_scaler = StandardScaler()
        X_flat   = X.reshape(-1, X.shape[-1])
        X_scaled = self.feature_scaler.fit_transform(X_flat).reshape(X.shape).astype(np.float32)

        self.model = self.build_model()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]

        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        self.trained_at = datetime.now(timezone.utc).isoformat()

        val_acc  = max(history.history.get("val_accuracy", [0.5]))
        baseline = max(float(y.mean()), 1.0 - float(y.mean()))
        logger.info(
            "TCN training complete: val_acc=%.4f  baseline=%.4f  "
            "improvement=%.1f%%",
            val_acc, baseline, (val_acc / baseline - 1) * 100,
        )

        gc.collect()
        return history

    def fine_tune(
        self,
        features_df: pd.DataFrame,
        hmm_states: pd.Series,
        returns_series: pd.Series,
        epochs: int       = 20,
        recent_years: int = 2,
    ):
        """Fine-tune on recent data at a reduced learning rate."""
        features_df = _add_derived_features(features_df)
        if recent_years:
            # Approximate bars per year for common TFs
            bars_per_year = len(features_df) // max(
                1,
                int((features_df.index[-1] - features_df.index[0]).days / 365),
            )
            cutoff_bars = recent_years * bars_per_year
            if cutoff_bars < len(features_df):
                features_df    = features_df.iloc[-cutoff_bars:]
                hmm_states     = hmm_states.iloc[-cutoff_bars:]
                returns_series = returns_series.iloc[-cutoff_bars:]
                logger.info(
                    "Fine-tune: last %d years → %d bars from %s",
                    recent_years, len(features_df), features_df.index[0],
                )

        X, y     = self.prepare_sequences(features_df, hmm_states, returns_series)
        X_scaled = self.feature_scaler.transform(
            X.reshape(-1, X.shape[-1])
        ).reshape(X.shape).astype(np.float32)

        self.model.optimizer.learning_rate.assign(1e-4)

        self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            verbose=1,
        )

        self.trained_at = datetime.now(timezone.utc).isoformat()
        gc.collect()

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_confidence(
        self,
        live_df: pd.DataFrame,
    ) -> Optional[float]:
        """Predict confidence multiplier from the last ``seq_len`` bars.

        Args:
            live_df: Raw live DataFrame (≥ seq_len rows) with the columns
                     produced by ``_build_live_df`` plus ``gmm_vol_cluster``.
                     Derived columns (rsi_normalized, bb_position, etc.) are
                     added internally.

        Returns:
            float in [0.7, 1.3] — multiply against base Z-cutoff.
            None  — model not loaded or feature error.
        """
        if self.model is None or self.feature_scaler is None:
            return None

        try:
            df_p       = _add_derived_features(live_df)
            active_cols = [c for c in _INPUT_COLUMNS if c in df_p.columns]

            if len(active_cols) < len(_INPUT_COLUMNS):
                missing = set(_INPUT_COLUMNS) - set(active_cols)
                logger.debug("TCN: missing feature columns %s — skipping", missing)
                return None

            feat = df_p[active_cols].values.astype(np.float64)
            feat = np.nan_to_num(feat, nan=0.0)

            # Pad if shorter than window
            if len(feat) < self.seq_len:
                pad  = np.zeros((self.seq_len - len(feat), feat.shape[1]))
                feat = np.vstack([pad, feat])
            feat = feat[-self.seq_len :].astype(np.float32)

            # Scale
            feat_flat   = feat.reshape(-1, self.n_features)
            feat_scaled = self.feature_scaler.transform(feat_flat).reshape(
                1, self.seq_len, self.n_features
            ).astype(np.float32)

            raw_conf = float(self.model.predict(feat_scaled, verbose=0)[0][0])

            # Temperature calibration
            if self.temperature != 1.0:
                logit    = np.log(raw_conf / (1.0 - raw_conf + 1e-7))
                raw_conf = float(
                    1.0 / (1.0 + np.exp(-logit / self.temperature))
                )

            # Map to Z multiplier: high confidence → lower multiplier (relax)
            multiplier = 1.3 - (raw_conf * 0.6)   # [0.7, 1.3]
            return float(np.clip(multiplier, 0.7, 1.3))

        except Exception as exc:
            logger.warning(
                "[TCN predict_confidence] FAILED: %s", exc, exc_info=True
            )
            return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save model weights, scaler, and metadata to *directory*."""
        os.makedirs(directory, exist_ok=True)

        self.model.save(os.path.join(directory, "tcn_confidence_model.keras"))

        with open(os.path.join(directory, "tcn_feature_scaler.pkl"), "wb") as f:
            pickle.dump(self.feature_scaler, f)

        metadata = {
            "seq_len":     self.seq_len,
            "n_features":  self.n_features,
            "n_states":    self.n_states,
            "temperature": self.temperature,
            "trained_at":  self.trained_at,
        }
        with open(os.path.join(directory, "tcn_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("TCN model saved: %s  trained=%s", directory, self.trained_at)

    def load(self, directory: str) -> "SignalConfidenceTCN":
        """Load model, scaler, and metadata from *directory*."""
        from tensorflow.keras.models import load_model as _load_keras

        model_path   = os.path.join(directory, "tcn_confidence_model.keras")
        scaler_path  = os.path.join(directory, "tcn_feature_scaler.pkl")
        meta_path    = os.path.join(directory, "tcn_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TCN model not found: {model_path}")

        self.model = _load_keras(model_path)

        with open(scaler_path, "rb") as f:
            self.feature_scaler = pickle.load(f)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata    = json.load(f)
            self.trained_at  = self.metadata.get("trained_at")
            self.seq_len     = self.metadata.get("seq_len",    self.seq_len)
            self.n_features  = self.metadata.get("n_features", self.n_features)
            self.n_states    = self.metadata.get("n_states",   self.n_states)
            self.temperature = self.metadata.get("temperature", self.temperature)

        return self

    # ── Health & staleness ────────────────────────────────────────────────────

    def health_check(self) -> Tuple[bool, str]:
        """Verify the model produces a valid multiplier on random input."""
        if self.model is None:
            return False, "No model loaded"

        test_seq = np.random.randn(1, self.seq_len, self.n_features).astype(np.float32)
        try:
            # Bypass scaler for the health probe — we just need shape/range check
            raw = float(self.model.predict(test_seq, verbose=0)[0][0])
            if not (0.0 <= raw <= 1.0):
                return False, f"Raw output out of [0,1]: {raw:.4f}"
            return True, "OK"
        except Exception as exc:
            return False, str(exc)

    def is_stale(self, max_age_days: int = TCN_MAX_AGE) -> bool:
        """Return True if the model is older than *max_age_days*."""
        if not self.trained_at:
            return True
        trained_dt = datetime.fromisoformat(self.trained_at)
        if trained_dt.tzinfo is None:
            trained_dt = trained_dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - trained_dt).days >= max_age_days
