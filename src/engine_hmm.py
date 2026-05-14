import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
        # Bull = highest kalman_return, Bear = lowest.
        # For the two middle states, sort by volatility (col 1) so Chop_Low
        # always maps to the lower-vol state and Chop_High to the higher-vol
        # one.  Sorting middles by return alone caused them to swap labels
        # across IS windows — the root cause of OOS regime inconsistency.
        middle_orig   = [int(sorted_idx[j]) for j in range(1, n_states - 1)]
        middle_by_vol = sorted(middle_orig, key=lambda s: model.means_[s, 1])
        remap = {}
        remap[int(sorted_idx[0])]  = 1  # Bear       (lowest return)
        remap[int(sorted_idx[-1])] = 0  # Bull       (highest return)
        remap[middle_by_vol[0]]    = 2  # Chop_Low   (lower vol)
        remap[middle_by_vol[1]]    = 3  # Chop_High  (higher vol)
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

    logger.info("State means (scaled: kalman_return, volatility, rsi_slope):")
    for i in range(n):
        logger.info(
            "  %s: return=%.6f, vol=%.6f, rsi_slope=%.6f",
            state_names.get(i, str(i)), model.means_[i, 0], model.means_[i, 1], model.means_[i, 2],
        )


def _kmeans_init(X: np.ndarray, n_states: int, random_state: int):
    """Seed HMM parameters from k-means clusters.

    Pre-seeding means_, covars_, transmat_, and startprob_ from k-means puts
    the Baum-Welch algorithm near a good solution from iteration 0, reducing
    local-minima variance across Optuna trials (especially beneficial on the
    long H1/M15 series where random init shows higher result scatter).

    Returns (means, covars, transmat, startprob) ready to assign to a
    GaussianHMM instance before calling fit() with init_params="".
    """
    km = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    n_features = X.shape[1]

    means = km.cluster_centers_                               # (k, f)
    covars = np.array([
        np.cov(X[labels == k].T) + np.eye(n_features) * 1e-6 # regularise
        if (labels == k).sum() > 1
        else np.eye(n_features)
        for k in range(n_states)
    ])                                                         # (k, f, f)

    # Sequential transition counts → row-normalised transmat
    transmat = np.zeros((n_states, n_states))
    for t in range(len(labels) - 1):
        transmat[labels[t], labels[t + 1]] += 1
    row_sums = transmat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transmat /= row_sums

    counts = np.bincount(labels, minlength=n_states).astype(float)
    startprob = counts / counts.sum()

    return means, covars, transmat, startprob


def fit_hmm(
    df: pd.DataFrame,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
    tf: str = "H1",
):
    X_raw = df[["kalman_return", "volatility", "rsi_slope"]].values
    # Normalise all three features to zero mean / unit variance so that
    # rsi_slope (typical range ±5) cannot dominate the regime classification
    # over kalman_return (typical range ±0.003) and volatility (±0.001).
    # Without scaling the HMM clusters on the highest-variance signal alone,
    # producing an extreme Bear state (rsi_slope -1.07) that only fires when
    # the market is already crashing — the move is usually exhausted by the
    # time the H1 bar closes.  StandardScaler ensures equal contribution.
    obs_scaler = StandardScaler()
    X = obs_scaler.fit_transform(X_raw)

    # Seed HMM from k-means clusters — reduces local-minima variance vs random init.
    # init_params="" tells hmmlearn not to overwrite our priors before fitting.
    means, covars, transmat, startprob = _kmeans_init(X, n_states, random_state)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
        init_params="",
    )
    model.means_     = means
    model.covars_    = covars
    model.transmat_  = transmat
    model.startprob_ = startprob
    model.fit(X)
    # Persist the scaler as an attribute so predict_states can reuse it.
    # joblib.dump preserves custom attributes, so save_model/load_model
    # transparently round-trips this without any API changes.
    model._obs_scaler = obs_scaler

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
    X_raw = df[["kalman_return", "volatility", "rsi_slope"]].values
    # Reuse the scaler fitted during training so the live/validation observation
    # space matches the space the HMM was fitted on.  Old models without this
    # attribute degrade gracefully (no scaling applied).
    scaler = getattr(model, "_obs_scaler", None)
    X = scaler.transform(X_raw) if scaler is not None else X_raw
    raw = model.predict(X)
    mean_returns = model.means_[:, 0]
    sorted_idx = np.argsort(mean_returns)
    n = model.n_components
    if n == 2:
        remap = {int(sorted_idx[0]): 1, int(sorted_idx[1]): 0}
    elif n == 3:
        remap = {int(sorted_idx[0]): 1, int(sorted_idx[1]): 2, int(sorted_idx[2]): 0}
    else:
        # Mirror the vol-based Chop ordering used in _sort_states so the
        # IS-fitted label assignment is reproduced exactly on OOS bars.
        middle_orig   = [int(sorted_idx[j]) for j in range(1, n - 1)]
        middle_by_vol = sorted(middle_orig, key=lambda s: model.means_[s, 1])
        remap = {int(sorted_idx[0]): 1, int(sorted_idx[-1]): 0}
        remap[middle_by_vol[0]] = 2  # Chop_Low  (lower vol)
        remap[middle_by_vol[1]] = 3  # Chop_High (higher vol)
    return np.array([remap.get(s, s) for s in raw])


def save_model(model: GaussianHMM, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("HMM model saved to %s", path)


def load_model(path: Path = MODEL_PATH) -> GaussianHMM:
    model = joblib.load(path)
    logger.info("HMM model loaded from %s", path)
    return model
