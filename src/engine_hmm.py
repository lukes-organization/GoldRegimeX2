import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
from src.logger import setup_logger, log_regime_transition

logger = setup_logger(__name__)

MODEL_PATH = Path("models/hmm_model.pkl")

# ── Semantic regime constants (replace Bull/Bear/Chop everywhere) ─────────────
REGIME_TREND = "TREND"
REGIME_MR    = "MEAN_REVERSION"
REGIME_SHOCK = "VOLATILITY_SHOCK"

# Canonical integer IDs — stable across runs and timeframes.
# 0 = TREND, 1 = MEAN_REVERSION, 2 = VOLATILITY_SHOCK
CANONICAL_REGIME_ID: dict[str, int] = {
    REGIME_TREND: 0,
    REGIME_MR:    1,
    REGIME_SHOCK: 2,
}
# Reverse lookup: canonical integer → semantic label
STATE_NAMES: dict[int, str] = {v: k for k, v in CANONICAL_REGIME_ID.items()}

# STATE_POLICY: True = trading allowed, False = strict no-trade
STATE_POLICY: dict[str, bool] = {
    REGIME_TREND: True,
    REGIME_MR:    False,   # mean-reversion: always no-trade
    REGIME_SHOCK: True,
}

# Legacy aliases so old notebooks/imports don't crash
STATE_NAMES_3 = STATE_NAMES
STATE_NAMES_2 = {0: REGIME_TREND, 1: REGIME_MR}
STATE_NAMES_4 = {0: REGIME_TREND, 1: REGIME_MR, 2: REGIME_SHOCK, 3: REGIME_SHOCK}

# ── Canonical-state contract enforcement ──────────────────────────────────────
_CANONICAL_STATE_IDS: frozenset = frozenset({0, 1, 2})


def _assert_canonical_states(states: np.ndarray, context: str) -> None:
    """Raise if any state id is outside {0, 1, 2}.

    Call this after every fit_hmm / predict_states so 4-state drift cannot
    propagate silently into features, backtest, or trial scoring.
    """
    uniq = sorted({int(x) for x in np.asarray(states).tolist()})
    bad = [u for u in uniq if u not in _CANONICAL_STATE_IDS]
    if bad:
        raise RuntimeError(
            f"{context}: non-canonical state ids detected: {bad}. "
            f"Only {{0=TREND, 1=MEAN_REVERSION, 2=VOLATILITY_SHOCK}} are allowed. "
            f"All unique ids found: {uniq}."
        )

# ── Regime classification thresholds (applied to StandardScaler-normalised stats) ─
# vol_mean:    means_[i, 1] — normalised average volatility of state i.
# ret_disp:    sqrt(covars_[i, 0, 0]) — normalised std of return feature.
# persistence: transmat_[i, i]
# entropy:     Shannon entropy (bits) of row i in transmat_.
_SHOCK_VOL_THR   = 0.75   # scaled vol mean above this → SHOCK candidate
_SHOCK_DISP_THR  = 0.75   # scaled return std above this → SHOCK candidate
_TREND_PERSIST   = 0.90   # self-transition prob above this → TREND candidate
_TREND_ENTROPY   = 0.55   # transition entropy below this → TREND candidate

# TF-specific diagonal boost applied to the HMM transition matrix after fitting.
# Adds this value to each self-transition probability then renormalises rows.
# Higher boost = stickier regime (fewer flips per bar).
# H1: 0.25 — hourly bars have genuine multi-hour trends; bias hard toward persistence.
# M15: 0.15 — moderate; 15-min noise warrants some smoothing but less than H1.
# M5: 0.10 — flexible; scalping benefits from faster regime detection.
TF_TRANS_BOOST = {"H1": 0.25, "M15": 0.15, "M5": 0.10}

# Median-filter kernel size applied to raw HMM states after fitting/prediction.
# Eliminates 1-bar and 2-bar regime flips that create noise in the XGB feature.
# H1 already benefits from a large transition boost so a smaller kernel suffices.
_MEDFILT_KERNEL = {"H1": 3, "M15": 5, "M5": 5}


def get_model_path(tf: str, broker: str = "headway_cent") -> Path:
    """Return the TF+broker-specific HMM model path.

    Example: get_model_path("H1", "headway_cent") → models/hmm_model_H1_headway_cent.pkl
    Falls back to the generic models/hmm_model_H1.pkl (then MODEL_PATH) if absent.
    """
    return Path(f"models/hmm_model_{tf.upper()}_{broker}.pkl")
def _classify_hmm_states(
    model: GaussianHMM,
    raw_states: np.ndarray,
    n_states: int,
) -> tuple[np.ndarray, dict, dict]:
    """Map raw HMM state IDs to canonical TREND / MEAN_REVERSION / VOLATILITY_SHOCK IDs.

    State labels are assigned independently from per-state descriptors, with no
    forced one-of-each constraint. This allows valid outcomes such as:
      - multiple TREND states,
      - multiple SHOCK states,
      - no MEAN_REVERSION state in a specific fit.

    Returns:
        remapped    — int32 array aligned with raw_states using canonical IDs
        state_names — {canonical_id: label}
        remap       — {raw_state_id: canonical_id} stored on model as _regime_remap
    """
    eps = 1e-12
    raw_counts = {int(i): int(np.sum(raw_states == i)) for i in range(n_states)}
    per_state: dict[int, dict] = {}
    for i in range(n_states):
        pers  = float(model.transmat_[i, i])
        row   = np.clip(model.transmat_[i], eps, 1.0)
        ent   = float(-np.sum(row * np.log2(row)))
        vol_m = float(model.means_[i, 1])
        ret_d = float(np.sqrt(max(float(model.covars_[i, 0, 0]), 0.0)))
        # Volatility proxy combines latent vol feature mean and return dispersion.
        vol_p = float(max(abs(vol_m), ret_d))
        per_state[i] = {
            "persistence": pers, "entropy": ent,
            "vol_mean": vol_m,   "ret_disp": ret_d,
            "volatility": vol_p,
        }

    def _classify_state(stats: dict) -> str:
        if (
            stats["persistence"] >= _TREND_PERSIST
            and stats["entropy"] <= _TREND_ENTROPY
        ):
            return REGIME_TREND
        if (
            stats["volatility"] >= _SHOCK_VOL_THR
            or stats["ret_disp"] >= _SHOCK_DISP_THR
        ):
            return REGIME_SHOCK
        return REGIME_MR

    assigned: dict[int, str] = {i: _classify_state(per_state[i]) for i in range(n_states)}

    # Mapping used for the currently predicted raw state sequence.
    old_remap: dict[int, int] = {
        raw: CANONICAL_REGIME_ID[label] for raw, label in assigned.items()
    }
    remapped = np.array([old_remap.get(int(s), 1) for s in raw_states], dtype=np.int32)

    # Best-effort permutation: choose a representative raw state for each
    # canonical bucket so model.transmat_[0/1/2] remains semantically meaningful.
    groups: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for raw, can in old_remap.items():
        groups[int(can)].append(int(raw))

    reps: dict[int, int] = {}
    for can_id in (0, 1, 2):
        cand = groups.get(can_id, [])
        if cand:
            reps[can_id] = max(cand, key=lambda s: raw_counts.get(int(s), 0))

    perm: list[int] = []
    used: set[int] = set()
    for can_id in (0, 1, 2):
        r = reps.get(can_id)
        if r is not None and r not in used:
            perm.append(int(r))
            used.add(int(r))
            continue
        for raw in range(n_states):
            if raw not in used:
                perm.append(int(raw))
                used.add(int(raw))
                break

    if len(perm) < n_states:
        for raw in range(n_states):
            if raw not in used:
                perm.append(int(raw))
                used.add(int(raw))

    remap: dict[int, int] = old_remap
    try:
        if sorted(perm) == list(range(n_states)):
            model.means_ = model.means_[perm]
            model.covars_ = model.covars_[perm]
            model.transmat_ = model.transmat_[np.ix_(perm, perm)]
            model.startprob_ = model.startprob_[perm]
            remap = {
                int(new_idx): old_remap.get(int(old_idx), 1)
                for new_idx, old_idx in enumerate(perm)
            }
    except (IndexError, ValueError):
        remap = old_remap

    state_names: dict[int, str] = {
        CANONICAL_REGIME_ID[lbl]: lbl
        for lbl in (REGIME_TREND, REGIME_MR, REGIME_SHOCK)
    }
    return remapped, state_names, remap


def _save_regime_diagnostics(
    model: GaussianHMM,
    state_names: dict,
    remap: dict,
    tf: str,
) -> None:
    """Write per-state classification statistics to models/regime_diagnostics_{tf}.json."""
    eps = 1e-12
    n = model.n_components
    states_info: dict = {}
    for raw_id in range(n):
        can_id = remap.get(raw_id, raw_id)
        label  = STATE_NAMES.get(can_id, f"state_{raw_id}")
        row    = np.clip(model.transmat_[raw_id], eps, 1.0)
        states_info[f"raw_state_{raw_id}"] = {
            "label":             label,
            "raw_id":            raw_id,
            "canonical_id":      int(can_id),
            "persistence":       round(float(model.transmat_[raw_id, raw_id]), 6),
            "entropy_bits":      round(float(-np.sum(row * np.log2(row))), 6),
            "vol_mean_scaled":   round(float(model.means_[raw_id, 1]), 6),
            "return_mean_scaled":round(float(model.means_[raw_id, 0]), 6),
            "return_dispersion": round(float(np.sqrt(max(float(model.covars_[raw_id, 0, 0]), 0.0))), 6),
            "tradeable":         STATE_POLICY.get(label, False),
        }
    out = {
        "tf":       tf.upper(),
        "n_states": n,
        "taxonomy": "TREND/MEAN_REVERSION/VOLATILITY_SHOCK",
        "states":   states_info,
    }
    path = Path(f"models/regime_diagnostics_{tf.upper()}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info("Regime diagnostics written → %s", path)


def _log_transition_matrix(model: GaussianHMM, state_names: dict):
    n = model.n_components
    if n != 3:
        raise RuntimeError(
            f"_log_transition_matrix: expected 3 states, got {n}. "
            f"Canonical contract requires exactly TREND/MEAN_REVERSION/VOLATILITY_SHOCK."
        )
    # Always log in canonical order 0→1→2 with semantic labels — never numeric fallback.
    _labels = [STATE_NAMES.get(j, f"UNKNOWN_{j}") for j in range(n)]
    logger.info("Transition Matrix:")
    header = "         " + "  ".join(f"{lbl:>16}" for lbl in _labels)
    logger.info(header)
    for i in range(n):
        row = f"{_labels[i]:>16} " + "  ".join(
            f"{model.transmat_[i, j]:8.4f}" for j in range(n)
        )
        logger.info(row)
        if model.transmat_[i, i] < 0.85:
            logger.warning(
                "State %s persistence %.3f < 0.85 -- consider higher Kalman smoothing",
                _labels[i], model.transmat_[i, i],
            )

    logger.info("State means (scaled: kalman_return, volatility, rsi_slope):")
    for i in range(n):
        logger.info(
            "  %s: return=%.6f, vol=%.6f, rsi_slope=%.6f",
            _labels[i], model.means_[i, 0], model.means_[i, 1], model.means_[i, 2],
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
    if int(n_states) != 3:
        raise ValueError(
            f"fit_hmm[{tf}]: n_states must be 3 (canonical contract), got {n_states}."
        )

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

    states, state_names, remap = _classify_hmm_states(model, raw_states, n_states)
    # Enforce canonical contract: no state id outside {0,1,2} is allowed.
    _assert_canonical_states(states, f"fit_hmm[{tf}]")
    # Store the remap on the model so predict_states can reproduce IS labels exactly.
    model._regime_remap = remap

    # ── HMM State Smoothing ───────────────────────────────────────────────────
    # Apply median filter *after* state remapping so it smooths semantically
    # consistent labels (0=Bull, 1=Bear, …) rather than arbitrary HMM indices.
    kernel = _MEDFILT_KERNEL.get(tf.upper(), 5)
    states = medfilt(states.astype(np.float64), kernel_size=kernel).astype(np.int32)

    _log_transition_matrix(model, state_names)
    _save_regime_diagnostics(model, state_names, remap, tf)

    transitions = np.where(np.diff(states) != 0)[0] + 1
    logger.info("Total regime transitions: %d", len(transitions))
    for idx in transitions[:10]:
        log_regime_transition(
            logger, df.index[idx], states[idx - 1], states[idx], state_names
        )

    return model, states, state_names


def predict_states(model: GaussianHMM, df: pd.DataFrame, tf: str = "H1") -> np.ndarray:
    """Predict HMM states on new data using the IS-fitted canonical remap.

    Uses model._regime_remap (stored by fit_hmm) so OOS labels match the
    canonical IDs produced during training:
        0 = TREND, 1 = MEAN_REVERSION, 2 = VOLATILITY_SHOCK

    Falls back to legacy return-sorted ordering for pre-rebuild model files.
    """
    if int(getattr(model, "n_components", 0)) != 3:
        raise ValueError(
            f"predict_states[{tf}]: expected model.n_components == 3, got {getattr(model, 'n_components', None)}."
        )

    X_raw = df[["kalman_return", "volatility", "rsi_slope"]].values
    scaler = getattr(model, "_obs_scaler", None)
    X = scaler.transform(X_raw) if scaler is not None else X_raw
    raw = model.predict(X)

    stored_remap = getattr(model, "_regime_remap", None)
    if stored_remap is not None:
        states = np.array([stored_remap.get(int(s), 1) for s in raw], dtype=np.int32)
    else:
        # Legacy fallback: return-sorted Bull=0 / Bear=1 / Chop=2 ordering.
        logger.warning(
            "predict_states: model has no _regime_remap — using legacy "
            "return-sorted fallback. Re-train to get semantic regime labels."
        )
        mean_returns = model.means_[:, 0]
        sorted_idx   = np.argsort(mean_returns)
        n = model.n_components
        if n == 2:
            _rm = {int(sorted_idx[0]): 1, int(sorted_idx[1]): 0}
        elif n == 3:
            _rm = {int(sorted_idx[0]): 1, int(sorted_idx[1]): 2, int(sorted_idx[2]): 0}
        else:
            middle_orig   = [int(sorted_idx[j]) for j in range(1, n - 1)]
            middle_by_vol = sorted(middle_orig, key=lambda s: model.means_[s, 1])
            _rm = {int(sorted_idx[0]): 1, int(sorted_idx[-1]): 0}
            _rm[middle_by_vol[0]] = 2
            _rm[middle_by_vol[1]] = 3
        states = np.array([_rm.get(s, s) for s in raw], dtype=np.int32)

    # ── HMM State Smoothing ───────────────────────────────────────────────────
    kernel = _MEDFILT_KERNEL.get(tf.upper(), 5)
    if len(states) >= kernel:
        states = medfilt(states.astype(np.float64), kernel_size=kernel).astype(np.int32)

    _assert_canonical_states(states, f"predict_states[{tf}]")
    return states


def save_model(model: GaussianHMM, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("HMM model saved to %s", path)


def load_model(path: Path = MODEL_PATH) -> GaussianHMM:
    model = joblib.load(path)
    logger.info("HMM model loaded from %s", path)
    return model
