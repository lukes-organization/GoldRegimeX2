"""
LSTM Regime Classifier for GoldRegime_X.

Predicts HMM state labels (Bull/Bear/Chop) from 100-bar sequences.
Ensembled with the HMM at inference time to give early warning of regime
transitions before the HMM commits to a new state.

Architecture:
    Input  (100, n_feats)
    → LSTM(64, return_sequences=True) + Dropout(0.3)
    → LSTM(32) + Dropout(0.3)
    → Dense(16, relu) + Dropout(0.2)
    → Dense(n_states, softmax, name='regime_output')

Save layout:
    models/lstm/{TF}_{broker}/lstm_regime_classifier.keras
    models/lstm/{TF}_{broker}/lstm_feature_scaler.pkl
    models/lstm/{TF}_{broker}/lstm_metadata.json
"""

import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.logger import setup_logger

logger = setup_logger(__name__)

# Backwards-compat: engine_xgb.py defines its own LSTM_CONTEXT_COLS; this stub
# prevents ImportError if anything still references the old name.
LSTM_CONTEXT_COLS: list[str] = []

LSTM_WINDOW   = 100
LSTM_N_STATES = 4     # default; overridden per model by n_states arg
STATE_NAMES   = {0: "Bull", 1: "Bear", 2: "Chop_Low", 3: "Chop_High"}


def get_lstm_dir(tf: str, broker: str = "headway_cent") -> Path:
    return Path(f"models/lstm/{tf.upper()}_{broker}")


# ── Feature helpers ────────────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LSTM-specific derived columns on a copy of df."""
    df = df.copy()

    if "rsi" in df.columns:
        df["rsi_normalized"] = df["rsi"] / 100.0
    else:
        df["rsi_normalized"] = 0.5

    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20, min_periods=1).mean()
        df["volume_ratio"] = (df["Volume"] / vol_ma.replace(0, np.nan).fillna(1)).clip(0.1, 5.0)
    else:
        df["volume_ratio"] = 1.0

    if "Close" in df.columns:
        ma     = df["Close"].rolling(20, min_periods=1).mean()
        std    = df["Close"].rolling(20, min_periods=1).std().fillna(0)
        bb_rng = (4 * std).replace(0, np.nan)
        df["bb_position"] = ((df["Close"] - (ma - 2 * std)) / bb_rng).clip(0, 1).fillna(0.5)

        sma50  = df["Close"].rolling(50, min_periods=1).mean()
        df["dist_from_sma50"] = ((df["Close"] - sma50) / sma50).fillna(0.0)
    else:
        df["bb_position"]    = 0.5
        df["dist_from_sma50"] = 0.0

    return df


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


# ── Model class ───────────────────────────────────────────────────────────────

class LSTMRegimeClassifier:
    """LSTM that classifies market regime from bar sequences.

    Trained to predict HMM state labels.  At inference time the prediction is
    ensembled with the HMM state to detect regime transitions early.
    """

    def __init__(
        self,
        sequence_length: int = LSTM_WINDOW,
        n_states: int = LSTM_N_STATES,
        temperature: float = 1.5,
    ):
        """
        Args:
            temperature: Softens the raw softmax output.
                         > 1.0 → more uncertain when inputs are unusual (better calibrated).
                         = 1.0 → raw model output, no change.
                         < 1.0 → sharper / over-confident.
        """
        self.sequence_length = sequence_length
        self.n_states        = n_states
        self.temperature     = temperature
        self._model          = None
        self.feature_scaler  = None
        self.input_columns   = list(_INPUT_COLUMNS)
        self._trained_at: Optional[str] = None   # ISO timestamp set on save/load

    # ── Temperature calibration ───────────────────────────────────────────────

    def _calibrate_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to the raw softmax output.

        Divides log-probs by ``self.temperature`` before re-normalising.
        T > 1 smooths the distribution (more honest about uncertainty on
        unusual inputs); T = 1 is an identity; T < 1 sharpens it.
        """
        if self.temperature == 1.0:
            return raw_probs
        log_probs    = np.log(raw_probs + 1e-8)
        scaled       = log_probs / self.temperature
        calibrated   = np.exp(scaled)
        calibrated  /= calibrated.sum()
        return calibrated

    # ── Architecture ──────────────────────────────────────────────────────────

    def _build_model(self, n_feats: int):
        from keras import Input
        from keras.layers import LSTM, Dense, Dropout
        from keras.models import Model
        from keras.optimizers import Adam

        inp = Input(shape=(self.sequence_length, n_feats), name="price_sequence")
        x   = LSTM(64, return_sequences=True,  name="lstm_1")(inp)
        x   = Dropout(0.3, name="dropout_1")(x)
        x   = LSTM(32, return_sequences=False, name="lstm_2")(x)
        x   = Dropout(0.3, name="dropout_2")(x)
        x   = Dense(16, activation="relu", name="dense_1")(x)
        x   = Dropout(0.2, name="dropout_3")(x)
        out = Dense(self.n_states, activation="softmax", name="regime_output")(x)

        model = Model(inputs=inp, outputs=out, name="lstm_regime_classifier")
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Training ──────────────────────────────────────────────────────────────

    def _prepare_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (X, y_onehot, y_raw) from a featurised DataFrame."""
        from keras.utils import to_categorical
        from sklearn.preprocessing import StandardScaler

        df = _add_derived_features(df)

        if "hmm_state" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'hmm_state' column. "
                "Pass HMM states via load_data_with_hmm_labels() before training."
            )

        active_cols = [c for c in self.input_columns if c in df.columns]
        df = df.dropna(subset=active_cols + ["hmm_state"])

        # Clamp labels to valid range
        df = df[df["hmm_state"].isin(range(self.n_states))].copy()

        logger.info("LSTM regime classifier — %d bars  states: %s", len(df),
                    dict(df["hmm_state"].value_counts().sort_index()))

        for s in range(self.n_states):
            cnt = (df["hmm_state"] == s).sum()
            logger.info("  State %d (%s): %d bars (%.1f%%)",
                        s, STATE_NAMES.get(s, "?"), cnt, cnt / len(df) * 100)

        feat  = df[active_cols].values.astype(np.float64)
        feat  = np.nan_to_num(feat, nan=0.0)
        self.feature_scaler = StandardScaler()
        feat  = self.feature_scaler.fit_transform(feat).astype(np.float32)

        labels = df["hmm_state"].values.astype(int)

        X, y = [], []
        for i in range(self.sequence_length, len(feat)):
            X.append(feat[i - self.sequence_length: i])
            y.append(labels[i])

        X       = np.array(X, dtype=np.float32)
        y_raw   = np.array(y, dtype=np.int32)
        y_onehot = to_categorical(y_raw, num_classes=self.n_states)

        logger.info("Sequences: %d  shape: %s", len(X), X.shape)
        return X, y_onehot, y_raw

    def fit(
        self,
        df: pd.DataFrame,
        tf: str = "H1",
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.1,
    ):
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        X, y_onehot, y_raw = self._prepare_sequences(df)

        self._model = self._build_model(X.shape[2])

        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = self._model.fit(
            X, y_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        val_acc      = history.history["val_accuracy"][-1]
        train_acc    = history.history["accuracy"][-1]
        majority_pct = np.bincount(y_raw, minlength=self.n_states).max() / len(y_raw)

        logger.info("LSTM regime classifier training complete [%s]:", tf)
        logger.info("  train accuracy : %.4f (%.1f%%)", train_acc, train_acc * 100)
        logger.info("  val accuracy   : %.4f (%.1f%%)", val_acc, val_acc * 100)
        logger.info("  baseline       : %.4f (%.1f%%)", majority_pct, majority_pct * 100)

        if val_acc > majority_pct * 1.05:
            logger.info(
                "  LSTM beats baseline by %.1f%%",
                (val_acc / majority_pct - 1) * 100,
            )
        else:
            logger.warning(
                "  LSTM does not beat baseline — may not add value. "
                "Consider more data or re-optimising the HMM."
            )

        return history

    def fine_tune(
        self,
        df_recent: pd.DataFrame,
        tf: str = "H1",
        epochs: int = 20,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
    ):
        """Fine-tune a pre-trained model on recent data.

        Uses a much lower learning rate than full training to preserve the
        model's general knowledge while adapting to recent market conditions.
        The feature scaler is re-fitted on the recent data so scaling stays
        consistent with the fine-tuning window.

        Args:
            df_recent:     DataFrame with recent bars (1–2 years recommended).
            tf:            Timeframe string (for logging only).
            epochs:        Maximum fine-tuning epochs.
            learning_rate: Lower than full training (e.g. 1e-4 vs 1e-3).
            batch_size:    Smaller batch helps generalise on small windows.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.optimizers import Adam

        if self._model is None:
            raise ValueError("Model must be trained before fine-tuning. Call fit() first.")

        logger.info(
            "Fine-tuning LSTM [%s] on %d recent bars (lr=%.1e)…",
            tf, len(df_recent), learning_rate,
        )

        X, y_onehot, y_raw = self._prepare_sequences(df_recent)

        if len(X) < 500:
            logger.warning(
                "Only %d sequences for fine-tuning — minimum 500 recommended. "
                "Results may be noisy.", len(X),
            )

        # Recompile with lower learning rate — preserves weights
        self._model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(
                monitor="loss",
                patience=5,
                restore_best_weights=True,
                min_delta=5e-4,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        history = self._model.fit(
            X, y_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        ft_acc = history.history["accuracy"][-1]
        logger.info("Fine-tuning complete [%s]: final accuracy=%.4f", tf, ft_acc)
        return history

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, df_recent: pd.DataFrame) -> Optional[Dict[int, float]]:
        """Return state probability dict from recent bars.

        Needs at least ``sequence_length`` rows; pads with zeros if fewer.
        Returns None when the model output is collapsed (range < 0.05) —
        callers should treat None as "LSTM unreliable, use HMM only".
        Returns uniform distribution on unexpected error.
        """
        if self._model is None:
            return {s: 1.0 / self.n_states for s in range(self.n_states)}

        try:
            df_p = _add_derived_features(df_recent)
            active_cols = [c for c in self.input_columns if c in df_p.columns]

            feat = df_p[active_cols].values.astype(np.float64)
            feat = np.nan_to_num(feat, nan=0.0)
            if self.feature_scaler is not None:
                feat = self.feature_scaler.transform(feat)

            # Pad if shorter than window
            if len(feat) < self.sequence_length:
                pad  = np.zeros((self.sequence_length - len(feat), feat.shape[1]))
                feat = np.vstack([pad, feat])
            feat = feat[-self.sequence_length:].astype(np.float32)

            probs = self._model.predict(feat[np.newaxis], verbose=0)[0]

            # Temperature calibration — softens over-confident raw softmax
            probs = self._calibrate_probs(probs)

            # Detect collapsed model — output range near zero means all classes
            # are equally likely, which is indistinguishable from random.
            if float(probs.max() - probs.min()) < 0.02:
                logger.warning(
                    "[LSTM COLLAPSED] probs=%s — near-uniform output, model unreliable. "
                    "Retrain with --mode train_lstm.", probs,
                )
                return None

            return {int(i): float(probs[i]) for i in range(self.n_states)}

        except Exception as exc:
            logger.debug("LSTM predict_proba error: %s", exc)
            return {s: 1.0 / self.n_states for s in range(self.n_states)}

    def predict_state(self, df_recent: pd.DataFrame) -> Optional[int]:
        probs = self.predict_proba(df_recent)
        if probs is None:
            return None
        return max(probs, key=probs.get)

    def ensemble_predict(
        self,
        df_recent: pd.DataFrame,
        hmm_state: int,
        lstm_weight: float = 0.3,
    ) -> Tuple[int, Dict]:
        """Ensemble LSTM probabilities with the HMM state.

        Returns (ensemble_state, confidence_info).

        confidence_info keys:
            agreement       bool   — LSTM and HMM agree on state
            lstm_confidence float  — LSTM's max prob
            transition_risk float  — 0→1, higher = regime likely changing
            lstm_state      int
            hmm_state       int
            ensemble_state  int
            lstm_probs      dict
            lstm_status     str    — 'OK' | 'COLLAPSED'
        """
        lstm_probs = self.predict_proba(df_recent)

        # Collapsed LSTM — return HMM-only fallback with zero influence
        if lstm_probs is None:
            return hmm_state, {
                "agreement":       True,   # don't penalise HMM on broken LSTM
                "lstm_confidence": 0.0,
                "transition_risk": 0.0,    # never tighten Z on a broken model
                "lstm_state":      hmm_state,
                "hmm_state":       hmm_state,
                "ensemble_state":  hmm_state,
                "lstm_probs":      {},
                "ensemble_scores": {s: 1.0 if s == hmm_state else 0.0 for s in range(self.n_states)},
                "lstm_status":     "COLLAPSED",
            }

        lstm_state      = max(lstm_probs, key=lstm_probs.get)
        lstm_confidence = lstm_probs[lstm_state]
        agreement       = (lstm_state == hmm_state)

        # transition_risk: only meaningful when LSTM is CONFIDENT about a
        # DIFFERENT state than HMM.  Agreement (or low confidence) → no risk.
        if lstm_state != hmm_state and lstm_confidence > 0.50:
            # LSTM disagrees AND is confident — genuine transition signal
            transition_risk = lstm_confidence
        elif lstm_state != hmm_state and lstm_confidence > 0.35:
            # LSTM disagrees but uncertain — mild risk, won't reach 0.5 gate
            transition_risk = 0.30
        else:
            # Agreement, or LSTM not confident enough to matter
            transition_risk = 0.0

        ensemble_scores = {}
        for s in range(self.n_states):
            hmm_score = 1.0 if s == hmm_state else 0.0
            ensemble_scores[s] = (
                (1 - lstm_weight) * hmm_score
                + lstm_weight * lstm_probs.get(s, 0.0)
            )
        ensemble_state = max(ensemble_scores, key=ensemble_scores.get)

        info = {
            "agreement":       agreement,
            "lstm_confidence": lstm_confidence,
            "transition_risk": transition_risk,
            "lstm_state":      lstm_state,
            "hmm_state":       hmm_state,
            "ensemble_state":  ensemble_state,
            "lstm_probs":      lstm_probs,
            "ensemble_scores": ensemble_scores,
            "lstm_status":     "OK",
        }
        return ensemble_state, info

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, dirpath) -> None:
        from datetime import datetime, timezone
        path = Path(dirpath)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "lstm_regime_classifier.keras"))
        joblib.dump(self.feature_scaler, path / "lstm_feature_scaler.pkl")
        trained_at = datetime.now(timezone.utc).isoformat()
        meta = {
            "sequence_length": self.sequence_length,
            "n_states":        self.n_states,
            "input_columns":   self.input_columns,
            "temperature":     self.temperature,
            "trained_at":      trained_at,
        }
        (path / "lstm_metadata.json").write_text(json.dumps(meta, indent=2))
        self._trained_at = trained_at
        logger.info("LSTM regime classifier saved: %s", path)

    @classmethod
    def load(cls, dirpath) -> "LSTMRegimeClassifier":
        from keras.models import load_model as _load_model

        path = Path(dirpath)
        meta = json.loads((path / "lstm_metadata.json").read_text())

        inst = cls(
            sequence_length=meta["sequence_length"],
            n_states=meta["n_states"],
            temperature=meta.get("temperature", 1.5),
        )
        inst.input_columns  = meta["input_columns"]
        inst._model         = _load_model(str(path / "lstm_regime_classifier.keras"))
        inst.feature_scaler = joblib.load(path / "lstm_feature_scaler.pkl")
        inst._trained_at    = meta.get("trained_at")

        logger.info(
            "LSTM regime classifier loaded: %s  (n_states=%d  T=%.1f  trained=%s)",
            path, inst.n_states, inst.temperature, inst._trained_at or "unknown",
        )
        return inst

    def is_stale(self, max_age_days: int = 7) -> bool:
        """Return True if the model is older than *max_age_days*.

        Models without a recorded training date are treated as stale.
        """
        from datetime import datetime, timezone

        if not self._trained_at:
            return True
        try:
            trained_dt = datetime.fromisoformat(self._trained_at)
            if trained_dt.tzinfo is None:
                trained_dt = trained_dt.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - trained_dt
            return age.days >= max_age_days
        except Exception:
            return True


# ── Ensemble ──────────────────────────────────────────────────────────────────

class LSTMEnsemble:
    """Ensemble of multiple LSTMRegimeClassifiers trained with different seeds.

    Averaging predictions across models smooths overconfident single-model
    outputs: when the models disagree, the averaged probabilities naturally
    approach uniform — no artificial threshold needed.

    Save / load layout  (same root dir as a single model):
        models/lstm/{TF}_{broker}/
            ensemble_metadata.json
            model_0/  lstm_regime_classifier.keras  …
            model_1/  …
            model_2/  …
    """

    def __init__(
        self,
        n_models: int = 3,
        sequence_length: int = LSTM_WINDOW,
        temperature: float = 1.5,
    ):
        self.n_models        = n_models
        self.sequence_length = sequence_length
        self.temperature     = temperature
        self.n_states: Optional[int] = None
        self._models: list[LSTMRegimeClassifier] = []
        self._trained_at: Optional[str] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        tf: str = "H1",
        epochs: int = 100,
        batch_size: int = 64,
    ):
        """Train *n_models* classifiers with different random seeds."""
        import tensorflow as _tf

        self._models = []
        for i in range(self.n_models):
            logger.info(
                "Training ensemble model %d/%d [%s]…", i + 1, self.n_models, tf
            )
            _tf.random.set_seed(42 + i * 7)
            np.random.seed(42 + i * 7)

            m = LSTMRegimeClassifier(
                sequence_length=self.sequence_length,
                n_states=int(df["hmm_state"].nunique()),
                temperature=self.temperature,
            )
            m.fit(df, tf=tf, epochs=epochs, batch_size=batch_size)
            self._models.append(m)

        self.n_states = self._models[0].n_states
        self._log_agreement(df.iloc[-min(5000, len(df)):])

    def fine_tune(
        self,
        df_recent: pd.DataFrame,
        tf: str = "H1",
        epochs: int = 20,
        learning_rate: float = 1e-4,
    ):
        """Fine-tune all member models on recent data."""
        for i, m in enumerate(self._models):
            logger.info("Fine-tuning ensemble model %d/%d [%s]…", i + 1, self.n_models, tf)
            m.fine_tune(df_recent, tf=tf, epochs=epochs, learning_rate=learning_rate)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, df_recent: pd.DataFrame) -> Optional[Dict[int, float]]:
        """Average probabilities across all member models.

        Returns None if a majority of models have collapsed.
        """
        all_probs: list[Dict[int, float]] = []
        n_collapsed = 0
        for m in self._models:
            p = m.predict_proba(df_recent)
            if p is None:
                n_collapsed += 1
            else:
                all_probs.append(p)

        if n_collapsed > len(self._models) // 2:
            logger.warning(
                "[LSTM ENSEMBLE] %d/%d models collapsed — ensemble unreliable.",
                n_collapsed, len(self._models),
            )
            return None

        n_states = self.n_states or len(all_probs[0])
        avg = {
            s: float(np.mean([p.get(s, 0.0) for p in all_probs]))
            for s in range(n_states)
        }
        return avg

    def predict_state(self, df_recent: pd.DataFrame) -> Optional[int]:
        probs = self.predict_proba(df_recent)
        if probs is None:
            return None
        return max(probs, key=probs.get)

    def ensemble_predict(
        self,
        df_recent: pd.DataFrame,
        hmm_state: int,
        lstm_weight: float = 0.3,
    ) -> Tuple[int, Dict]:
        """Same interface as ``LSTMRegimeClassifier.ensemble_predict``."""
        lstm_probs = self.predict_proba(df_recent)
        n_states   = self.n_states or LSTM_N_STATES

        if lstm_probs is None:
            return hmm_state, {
                "agreement":       True,
                "lstm_confidence": 0.0,
                "transition_risk": 0.0,
                "lstm_state":      hmm_state,
                "hmm_state":       hmm_state,
                "ensemble_state":  hmm_state,
                "lstm_probs":      {},
                "ensemble_scores": {s: 1.0 if s == hmm_state else 0.0 for s in range(n_states)},
                "lstm_status":     "ENSEMBLE_COLLAPSED",
                "n_active_models": 0,
            }

        lstm_state      = max(lstm_probs, key=lstm_probs.get)
        lstm_confidence = lstm_probs[lstm_state]
        agreement       = (lstm_state == hmm_state)

        if lstm_state != hmm_state and lstm_confidence > 0.50:
            transition_risk = lstm_confidence
        elif lstm_state != hmm_state and lstm_confidence > 0.35:
            transition_risk = 0.30
        else:
            transition_risk = 0.0

        ensemble_scores = {
            s: (1 - lstm_weight) * (1.0 if s == hmm_state else 0.0)
               + lstm_weight * lstm_probs.get(s, 0.0)
            for s in range(n_states)
        }
        ensemble_state = max(ensemble_scores, key=ensemble_scores.get)

        return ensemble_state, {
            "agreement":         agreement,
            "lstm_confidence":   lstm_confidence,
            "transition_risk":   transition_risk,
            "lstm_state":        lstm_state,
            "hmm_state":         hmm_state,
            "ensemble_state":    ensemble_state,
            "lstm_probs":        lstm_probs,
            "ensemble_scores":   ensemble_scores,
            "lstm_status":       "OK",
            "n_active_models":   len(self._models),
        }

    # ── Staleness ─────────────────────────────────────────────────────────────

    def is_stale(self, max_age_days: int = 7) -> bool:
        """True if the ensemble was trained more than *max_age_days* ago."""
        from datetime import datetime, timezone

        if not self._trained_at:
            return True
        try:
            trained_dt = datetime.fromisoformat(self._trained_at)
            if trained_dt.tzinfo is None:
                trained_dt = trained_dt.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - trained_dt).days >= max_age_days
        except Exception:
            return True

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, dirpath) -> None:
        from datetime import datetime, timezone

        path = Path(dirpath)
        path.mkdir(parents=True, exist_ok=True)

        for i, m in enumerate(self._models):
            m.save(path / f"model_{i}")

        trained_at = datetime.now(timezone.utc).isoformat()
        meta = {
            "n_models":        self.n_models,
            "n_states":        self.n_states,
            "sequence_length": self.sequence_length,
            "temperature":     self.temperature,
            "trained_at":      trained_at,
            "model_dirs":      [f"model_{i}" for i in range(self.n_models)],
        }
        (path / "ensemble_metadata.json").write_text(json.dumps(meta, indent=2))
        self._trained_at = trained_at
        logger.info("LSTM ensemble (%d models) saved: %s", self.n_models, path)

    @classmethod
    def load(cls, dirpath) -> "LSTMEnsemble":
        path = Path(dirpath)
        meta = json.loads((path / "ensemble_metadata.json").read_text())

        inst = cls(
            n_models=meta["n_models"],
            sequence_length=meta["sequence_length"],
            temperature=meta.get("temperature", 1.5),
        )
        inst.n_states    = meta["n_states"]
        inst._trained_at = meta.get("trained_at")

        for model_dir in meta["model_dirs"]:
            inst._models.append(LSTMRegimeClassifier.load(path / model_dir))

        logger.info(
            "LSTM ensemble loaded: %s  (%d models  n_states=%d  trained=%s)",
            path, len(inst._models), inst.n_states, inst._trained_at or "unknown",
        )
        return inst

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log_agreement(self, df_val: pd.DataFrame) -> None:
        """Log what fraction of recent bars the ensemble models agree on."""
        n_check  = min(200, len(df_val) - self.sequence_length)
        if n_check <= 0:
            return
        agree = 0
        for i in range(len(df_val) - n_check, len(df_val)):
            window = df_val.iloc[max(0, i - self.sequence_length): i]
            votes  = []
            for m in self._models:
                p = m.predict_proba(window)
                if p is not None:
                    votes.append(max(p, key=p.get))
            if votes and len(set(votes)) == 1:
                agree += 1
        logger.info(
            "[LSTM ENSEMBLE] Model agreement on validation sample: %.1f%%",
            agree / n_check * 100,
        )


# ── Standalone helpers ────────────────────────────────────────────────────────

def load_lstm_classifier(
    tf: str, broker: str = "headway_cent"
) -> Optional[LSTMRegimeClassifier]:
    """Load the best available LSTM classifier for *tf* / *broker*.

    Priority:
      1. ``LSTMEnsemble``  — if ``ensemble_metadata.json`` exists in the dir.
      2. ``LSTMRegimeClassifier`` — if ``lstm_regime_classifier.keras`` exists.
      3. ``None``          — nothing found, caller runs HMM-only.
    """
    path = get_lstm_dir(tf, broker)

    # 1) Try ensemble first
    if (path / "ensemble_metadata.json").exists():
        try:
            clf = LSTMEnsemble.load(path)
            logger.info(
                "LSTM ensemble loaded [%s/%s] — %d models  n_states=%d",
                tf, broker, clf.n_models, clf.n_states,
            )
            return clf  # type: ignore[return-value]  LSTMEnsemble has same interface
        except Exception as exc:
            logger.warning(
                "Failed to load LSTM ensemble from %s: %s — trying single model.", path, exc
            )

    # 2) Fall back to single model
    if (path / "lstm_regime_classifier.keras").exists():
        try:
            clf = LSTMRegimeClassifier.load(path)
            logger.info(
                "LSTM single model loaded [%s/%s] — n_states=%d",
                tf, broker, clf.n_states,
            )
            return clf
        except Exception as exc:
            logger.warning(
                "Failed to load LSTM classifier from %s: %s — running HMM-only.", path, exc
            )

    logger.debug("No LSTM model found at %s — running HMM-only.", path)
    return None
