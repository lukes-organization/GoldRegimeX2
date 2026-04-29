"""Temporal Convolutional Network — RCEV threshold adjustment.

Scores confidence in the current market conditions from a sequence of recent
feature bars.  Confidence is expressed as a multiplier [0.7, 1.3] that scales
the RCEV threshold in the live bridge:

    effective_threshold = rcev_threshold × confidence_multiplier

    multiplier < 1.0  → lower RCEV threshold → more trades fire (strong regime)
    multiplier = 1.0  → no adjustment (TCN not loaded or neutral conditions)
    multiplier > 1.0  → higher RCEV threshold → fewer trades fire (noisy/uncertain)

This replaces the previous Z-Score cutoff scaling role — the RCEV threshold is
now the tunable gate, and the TCN adjusts how much expected profit is required
to confirm a signal.

Architecture:
    4 × dilated causal Conv1D layers (dilation 1, 2, 4, 8)
    GlobalAveragePooling → Dense(32) → sigmoid output

Training target (regime-persistence):
    1 if the HMM regime at (current_bar + forward_window) matches the current
    regime — i.e. the regime is stable and the signal should be trusted.
    0 if the regime has changed — i.e. the signal should be treated cautiously.
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
    "kalman_return",
    "volatility",
    "rsi",
    "rsi_slope",
    "atr_normalized",
    "gmm_vol_cluster",
    "usdchf_log_return",
]


# ── Feature helpers ────────────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all _INPUT_COLUMNS are present on a copy of *df*.

    All eight input columns are native outputs of the processor pipeline
    (log_return, kalman_return, volatility, rsi, rsi_slope, atr_normalized,
    gmm_vol_cluster).  The only one that may be absent is usdchf_log_return
    when the USDCHF master file was not available — fill with 0.0 in that case.
    """
    df = df.copy()
    if "usdchf_log_return" not in df.columns:
        df["usdchf_log_return"] = 0.0
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
        from keras.models import Sequential
        from keras.layers import (
            Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input,
        )
        from keras.optimizers import Adam

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
        forward_window: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) training pairs using regime-persistence targets.

        Target = 1 if the HMM regime at bar (seq_end + forward_window) is the
        same as the regime at bar seq_end (current decision point).

        This maps directly to the TCN's output purpose:
            multiplier < 1.0  (high confidence) → regime persists
            multiplier > 1.0  (low confidence)  → regime is about to change

        Regime persistence is a clean, learnable signal because:
          - HMM self-transition probs are ≥ 0.65 by construction (optimizer gate)
          - Sequential feature patterns (RSI trends, vol changes) genuinely
            predict impending regime transitions
          - Unlike raw price direction, state labels are deterministic not noisy

        Typical forward windows by TF:
            H1:  5 bars  (~5 hours  — half a trading session)
            M15: 12 bars (~3 hours)
            M5:  24 bars (~2 hours)
        """
        feature_cols = [c for c in _INPUT_COLUMNS if c in features_df.columns]
        n = len(features_df)
        _ = returns_series  # kept for API compatibility; not used with persistence target
        X: list = []
        y: list = []
        skipped_edge = 0

        logger.info("Preparing sequences  forward_window=%d…", forward_window)

        for i in range(n - self.seq_len - forward_window):
            seq_end = i + self.seq_len

            seq = features_df[feature_cols].iloc[i : seq_end].values
            if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
                skipped_edge += 1
                continue

            current_state = int(hmm_states.iloc[seq_end])
            future_state  = int(hmm_states.iloc[seq_end + forward_window])

            # 1 = regime persists (confidence high → relax Z cutoff)
            # 0 = regime changes  (confidence low  → tighten Z cutoff)
            target = 1 if future_state == current_state else 0

            X.append(seq)
            y.append(target)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        n_total  = len(y_arr)
        n_stable = int(np.sum(y_arr == 1))
        n_change = n_total - n_stable
        ratio    = max(n_stable, n_change) / max(min(n_stable, n_change), 1)

        logger.info(
            "Sequences prepared: %d samples  (skipped %d edge cases)",
            n_total, skipped_edge,
        )
        logger.info(
            "  Regime stable: %d (%.1f%%)  Regime changes: %d (%.1f%%)  "
            "Ratio: %.2f:1",
            n_stable, 100 * n_stable / max(n_total, 1),
            n_change, 100 * n_change / max(n_total, 1),
            ratio,
        )
        if ratio > 4.0:
            logger.warning(
                "High class imbalance (%.1f:1) — class weights will compensate", ratio
            )
        return X_arr, y_arr

    def train(
        self,
        features_df: pd.DataFrame,
        hmm_states: pd.Series,
        returns_series: pd.Series,
        epochs: int             = 100,
        batch_size: int         = 64,
        validation_split: float = 0.2,
    ):
        """Full training from scratch with forward-window trade outcome targets."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.class_weight import compute_class_weight
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        # Ensure derived feature columns exist before building sequences
        features_df = _add_derived_features(features_df)

        # Auto-select forward window based on approximate TF (by data size).
        # Regime persistence windows: long enough to filter noise,
        # short enough that the current feature sequence still has predictive power.
        n_bars = len(features_df)
        if n_bars > 200_000:          # M5
            forward_window = 24       # ~2 hours
        elif n_bars > 80_000:         # M15
            forward_window = 12       # ~3 hours
        else:                         # H1
            forward_window = 5        # ~5 hours (half a session)

        logger.info("TCN training — %d bars  forward_window=%d", n_bars, forward_window)

        X, y = self.prepare_sequences(
            features_df, hmm_states, returns_series,
            forward_window=forward_window,
        )

        if len(X) < 1000:
            logger.error(
                "Insufficient training samples: %d — need at least 1000", len(X)
            )
            return None

        logger.info("Sequences: %d  shape: %s  positive_rate: %.1f%%",
                    len(X), X.shape, 100 * y.mean())

        # Update n_features to match actual available columns
        self.n_features = X.shape[-1]

        # Fit scaler on flattened feature matrix
        self.feature_scaler = StandardScaler()
        X_flat   = X.reshape(-1, self.n_features)
        X_scaled = self.feature_scaler.fit_transform(X_flat).reshape(X.shape).astype(np.float32)

        self.model = self.build_model()

        # Class weights to handle imbalanced profitable/unprofitable split
        unique_classes = np.unique(y)
        class_weights  = compute_class_weight("balanced", classes=unique_classes, y=y)
        class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, class_weights)}
        logger.info("Class weights: %s", class_weight_dict)

        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
                min_delta=0.005,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1,
        )

        self.trained_at = datetime.now(timezone.utc).isoformat()

        best_epoch    = int(np.argmax(history.history.get("val_accuracy", [0]))) + 1
        best_val_acc  = float(max(history.history.get("val_accuracy",  [0.5])))
        best_train_acc = float(history.history.get("accuracy", [0.5])[best_epoch - 1])
        baseline       = float(max(y.mean(), 1.0 - y.mean()))

        logger.info("TCN training complete (regime-persistence target):")
        logger.info("  Best epoch:   %d", best_epoch)
        logger.info("  Train acc:    %.4f (%.1f%%)", best_train_acc, best_train_acc * 100)
        logger.info("  Val acc:      %.4f (%.1f%%)", best_val_acc,   best_val_acc   * 100)
        logger.info("  Baseline:     %.4f (%.1f%%) (majority class)", baseline, baseline * 100)
        logger.info(
            "  Improvement: %+.1f%% over baseline",
            (best_val_acc / baseline - 1) * 100,
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
        """Fine-tune on recent data at a reduced learning rate.

        Recompiles the optimizer at a lower LR to reset Adam's momentum/variance
        accumulators — prevents catastrophic forgetting on the first gradient step.
        Class weights are intentionally omitted: the model already learned the base
        distribution during full training; fine-tuning only needs to adapt, not
        re-balance the output distribution.
        """
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping

        features_df = _add_derived_features(features_df)
        if recent_years:
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

        n_bars = len(features_df)
        forward_window = 24 if n_bars > 200_000 else (12 if n_bars > 80_000 else 5)

        X, y = self.prepare_sequences(
            features_df, hmm_states, returns_series,
            forward_window=forward_window,
        )

        X_scaled = self.feature_scaler.transform(
            X.reshape(-1, X.shape[-1])
        ).reshape(X.shape).astype(np.float32)

        # Recompile with fresh Adam at low LR — resets momentum/variance accumulators
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                min_delta=0.002,
            ),
        ]

        self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
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
        from keras.models import load_model as _load_keras

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
