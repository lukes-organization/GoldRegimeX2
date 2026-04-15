import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from src.logger import setup_logger, log_regime_transition

logger = setup_logger(__name__)

MODEL_PATH = Path("models/hmm_model.pkl")

# TF-specific diagonal boost applied to the HMM transition matrix after fitting.
# Adds this value to each self-transition probability then renormalises rows.
# Higher boost = stickier regime (fewer flips per bar).
# H1: 0.25 — hourly bars have genuine multi-hour trends; bias hard toward persistence.
# M15: 0.15 — moderate; 15-min noise warrants some smoothing but less than H1.
# M5: 0.10 — flexible; scalping benefits from faster regime detection.
TF_TRANS_BOOST = {"H1": 0.25, "M15": 0.15, "M5": 0.10}


def get_model_path(tf: str, broker: str = "headway_cent") -> Path:
    """Return the TF+broker-specific HMM model path.

    Example: get_model_path("H1", "headway_cent") → models/hmm_model_H1_headway_cent.pkl
    Falls back to the generic models/hmm_model_H1.pkl (then MODEL_PATH) if absent.
    """
    return Path(f"models/hmm_model_{tf.upper()}_{broker}.pkl")
STATE_NAMES_3 = {0: "Bull", 1: "Bear", 2: "Chop"}
STATE_NAMES_2 = {0: "Bull", 1: "Bear"}
STATE_NAMES_4 = {0: "Bull", 1: "Bear", 2: "Chop_Low", 3: "Chop_High"}


def _sort_states(model: GaussianHMM, raw_states: np.ndarray, n_states: int):
    mean_returns = model.means_[:, 0]
    sorted_idx = np.argsort(mean_returns)

    if n_states == 2:
        remap = {int(sorted_idx[0]): 1, int(sorted_idx[1]): 0}
        state_names = STATE_NAMES_2
    elif n_states == 3:
        remap = {
            int(sorted_idx[0]): 1,  # lowest mean -> Bear
            int(sorted_idx[1]): 2,  # middle      -> Chop
            int(sorted_idx[2]): 0,  # highest     -> Bull
        }
        state_names = STATE_NAMES_3
    else:
        remap = {int(sorted_idx[i]): i for i in range(n_states)}
        remap[int(sorted_idx[0])] = 1   # Bear
        remap[int(sorted_idx[-1])] = 0  # Bull
        for j in range(1, n_states - 1):
            remap[int(sorted_idx[j])] = j + 1
        state_names = STATE_NAMES_4

    remapped = np.array([remap[s] for s in raw_states])

    perm = [0] * n_states
    for old, new in remap.items():
        perm[new] = old
    model.means_ = model.means_[perm]
    model.covars_ = model.covars_[perm]
    model.transmat_ = model.transmat_[np.ix_(perm, perm)]
    model.startprob_ = model.startprob_[perm]

    return remapped, state_names


def _log_transition_matrix(model: GaussianHMM, state_names: dict):
    n = model.n_components
    logger.info("Transition Matrix:")
    header = "         " + "  ".join(f"{state_names.get(j, str(j)):>8}" for j in range(n))
    logger.info(header)
    for i in range(n):
        row = f"{state_names.get(i, str(i)):>8} " + "  ".join(
            f"{model.transmat_[i, j]:8.4f}" for j in range(n)
        )
        logger.info(row)
        if model.transmat_[i, i] < 0.85:
            logger.warning(
                "State %s persistence %.3f < 0.85 -- consider higher Kalman smoothing",
                state_names.get(i, str(i)), model.transmat_[i, i],
            )

    logger.info("State means (kalman_return, volatility):")
    for i in range(n):
        logger.info(
            "  %s: return=%.6f, vol=%.6f",
            state_names.get(i, str(i)), model.means_[i, 0], model.means_[i, 1],
        )


def fit_hmm(
    df: pd.DataFrame,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
    tf: str = "H1",
):
    X = df[["kalman_return", "volatility"]].values
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)

    # ── HMM Persistence Boost ─────────────────────────────────────────────────
    # Add a TF-specific value to each diagonal of the transition matrix, then
    # renormalise rows to sum to 1.  This biases the model toward staying in the
    # current regime, reducing bar-to-bar jitter ("regime flip every 2 bars").
    boost = TF_TRANS_BOOST.get(tf.upper(), 0.10)
    mat = model.transmat_.copy()
    np.fill_diagonal(mat, mat.diagonal() + boost)
    model.transmat_ = mat / mat.sum(axis=1, keepdims=True)

    raw_states = model.predict(X)

    states, state_names = _sort_states(model, raw_states, n_states)
    _log_transition_matrix(model, state_names)

    transitions = np.where(np.diff(states) != 0)[0] + 1
    logger.info("Total regime transitions: %d", len(transitions))
    for idx in transitions[:10]:
        log_regime_transition(
            logger, df.index[idx], states[idx - 1], states[idx], state_names
        )

    return model, states, state_names


def predict_states(model: GaussianHMM, df: pd.DataFrame) -> np.ndarray:
    X = df[["kalman_return", "volatility"]].values
    raw = model.predict(X)
    mean_returns = model.means_[:, 0]
    sorted_idx = np.argsort(mean_returns)
    n = model.n_components
    if n == 2:
        remap = {int(sorted_idx[0]): 1, int(sorted_idx[1]): 0}
    elif n == 3:
        remap = {int(sorted_idx[0]): 1, int(sorted_idx[1]): 2, int(sorted_idx[2]): 0}
    else:
        remap = {int(sorted_idx[0]): 1, int(sorted_idx[-1]): 0}
        for j in range(1, n - 1):
            remap[int(sorted_idx[j])] = j + 1
    return np.array([remap.get(s, s) for s in raw])


def save_model(model: GaussianHMM, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("HMM model saved to %s", path)


def load_model(path: Path = MODEL_PATH) -> GaussianHMM:
    model = joblib.load(path)
    logger.info("HMM model loaded from %s", path)
    return model
