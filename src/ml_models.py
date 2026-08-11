"""src/ml_models.py -- model classes for the live bundle (VERBATIM from the
Explorer notebook: the ML dependency shims + HMMXGBComposite).  Required so
models/goldregimex_live_model.pkl can be un-pickled by the live app / backtester
seam.  strategy_backtest registers these on __main__ before loading the bundle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Tuple
import itertools
import math
import time

# ---- Project root bootstrap (critical for joblib child processes on Windows) ----
# Walk up from CWD until we find main.py — the canonical repo-root marker.
# Handles notebooks/, pipeline_verification_bundle/, or any nested subdirectory.
_here = Path.cwd().resolve()
_project_root = _here
for _candidate in [_here, *_here.parents]:
    if (_candidate / "main.py").exists() and (_candidate / "data").exists():
        _project_root = _candidate
        break

os.chdir(_project_root)
print(f"[Bootstrap] CWD anchored to: {_project_root}")

_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

_prev_py_path = os.environ.get("PYTHONPATH", "")
if _prev_py_path:
    if _project_root_str not in _prev_py_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = _project_root_str + os.pathsep + _prev_py_path
else:
    os.environ["PYTHONPATH"] = _project_root_str

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional ML dependencies. In a full local quant environment these should import normally.
# Fallbacks only exist so the notebook can be smoke-tested in this restricted sandbox.
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    class BaseEstimator: pass
    class ClassifierMixin: pass
    class StandardScaler:
        def fit_transform(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)
            self.std_[self.std_ == 0] = 1.0
            return (X - self.mean_) / self.std_
        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.mean_) / self.std_

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except Exception:
    HMM_OK = False
    class GaussianHMM:
        def __init__(self, n_components=3, covariance_type="full", n_iter=120, random_state=42, **kwargs):
            # kwargs (e.g. min_covar) ignored by the lightweight fallback.
            self.n_components = int(n_components)
            self.random_state = int(random_state)
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.n_features_ = X.shape[1] if X.ndim == 2 else 1
            return self
        def predict(self, X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            score = np.nan_to_num(X[:, 0], nan=0.0)
            if len(score) >= 3:
                q1, q2 = np.quantile(score, [1/3, 2/3])
            else:
                q1, q2 = 0.0, 0.0
            return np.where(score <= q1, 0, np.where(score <= q2, 1, 2)).astype(int)

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False
    class _FallbackXGBClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def fit(self, X, y):
            y = np.asarray(y, dtype=int)
            counts = np.bincount(y, minlength=3).astype(float)
            if counts.sum() == 0:
                counts[:] = 1.0
            self.prior_ = counts / counts.sum()
            return self
        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            return np.tile(self.prior_, (len(X), 1))
        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1).astype(int)
    class _XGBModule:
        XGBClassifier = _FallbackXGBClassifier
    xgb = _XGBModule()

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

try:
    from joblib import Parallel, delayed
    JOBLIB_OK = True
except Exception:
    JOBLIB_OK = False
    class Parallel:
        def __init__(self, n_jobs=1, backend=None, verbose=0):
            self.n_jobs = n_jobs
        def __call__(self, jobs):
            return [job() if callable(job) else job for job in jobs]
    def delayed(fn):
        def wrapper(*args, **kwargs):
            return lambda: fn(*args, **kwargs)
        return wrapper

try:
    import importlib

    try:
        _backtester = importlib.import_module("src.backtester")
        _trade_lifecycle = importlib.import_module("src.trade_lifecycle")
        _risk_manager = importlib.import_module("src.risk_manager")
    except ImportError:
        _backtester = importlib.import_module("backtester")
        _trade_lifecycle = importlib.import_module("trade_lifecycle")
        _risk_manager = importlib.import_module("risk_manager")

    vectorized_backtest = _backtester.vectorized_backtest
    config_for_tf = _trade_lifecycle.config_for_tf
    BROKER_CONFIGS = _risk_manager.BROKER_CONFIGS
    SRC_OK = True
except Exception:
    SRC_OK = False
    vectorized_backtest = None
    def config_for_tf(*args, **kwargs):
        return {}
    BROKER_CONFIGS = {}

warnings.filterwarnings("ignore")
np.random.seed(42)

print("Working directory:", os.getcwd())
print("Project root:", _project_root_str)
print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
print("joblib available:", JOBLIB_OK)
print("optional deps:", {"sklearn": SKLEARN_OK, "hmmlearn": HMM_OK, "xgboost": XGB_OK, "numba": NUMBA_OK, "src": SRC_OK})


# -----------------------------
# HMM + XGBoost composite
# -----------------------------

class HMMXGBComposite(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_components=3,
        max_depth=3,
        learning_rate=0.03,
        n_estimators=600,
        random_state=42,
        min_child_weight=5.0,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=0.5,
        gamma=1.0,
        early_stopping_rounds=40,
        es_val_fraction=0.15,
    ):
        self.n_components = int(n_components)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.min_child_weight = float(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.gamma = float(gamma)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.es_val_fraction = float(es_val_fraction)

        self.hmm = None
        self.hmm_scaler = None
        self.xgb = None
        self.feature_names_ = None
        self.hmm_features_ = None

        self.y_to_cls_ = {-1: 0, 0: 1, 1: 2}
        self.cls_to_y_ = {0: -1, 1: 0, 2: 1}

    def fit(self, X: pd.DataFrame, y: pd.Series, hmm_features: list[str]):
        X = X.copy()
        y = pd.Series(y).astype(int)

        # FIX: Strictly isolate numeric columns to prevent strings like 'TREND' from crashing XGBoost
        X_numeric = X.select_dtypes(include=['number', 'bool'])
        self.feature_names_ = list(X_numeric.columns)
        self.hmm_features_ = list(hmm_features)

        X_hmm = X[self.hmm_features_].to_numpy(dtype=float)
        self.hmm_scaler = StandardScaler()
        X_hmm = self.hmm_scaler.fit_transform(X_hmm)

        # Robust HMM fit with convergence-aware fallback
        # Strategy: try multiple covariance configs; check monitor_.converged
        # after fit.  If a config does not converge, skip to the next one.
        # If none converge, use the one with the highest final log-likelihood.
        import logging

        # Temporarily suppress hmmlearn's "Model is not converging" logger noise
        _hmm_logger = logging.getLogger("hmmlearn")
        _prev_level = _hmm_logger.level
        _hmm_logger.setLevel(logging.CRITICAL + 1)

        covar_configs = [
            {"covariance_type": "diag", "min_covar": 0.01},
            {"covariance_type": "diag", "min_covar": 0.1},
            {"covariance_type": "full", "min_covar": 0.01},
            {"covariance_type": "full", "min_covar": 0.1},
            {"covariance_type": "spherical", "min_covar": 0.01},
        ]
        self.hmm = None
        last_err = None
        best_hmm = None
        best_ll = -np.inf

        for _cfg in covar_configs:
            try:
                _hmm = GaussianHMM(
                    n_components=self.n_components,
                    n_iter=300,
                    tol=1e-3,
                    random_state=self.random_state,
                    **_cfg,
                )
                _hmm.fit(X_hmm)
                _did_converge = bool(_hmm.monitor_.converged)
                _regimes = _hmm.predict(X_hmm)
                _ll = _hmm.score(X_hmm)

                if _did_converge:
                    self.hmm = _hmm
                    regimes = _regimes
                    last_err = None
                    _ct = _cfg["covariance_type"]
                    _mc = _cfg["min_covar"]
                    print(f"  HMM converged ({_ct}/{_mc}) LL={_ll:.1f}")
                    break
                else:
                    # Non-converged but usable - keep as fallback if LL is best
                    if _ll > best_ll:
                        best_ll = _ll
                        best_hmm = (_hmm, _regimes, _cfg)
                    continue

            except Exception as _e:
                last_err = _e
                continue

        # Restore hmmlearn logger level
        _hmm_logger.setLevel(_prev_level)

        # If no config fully converged, use the best non-converged one
        if self.hmm is None and best_hmm is not None:
            self.hmm = best_hmm[0]
            regimes = best_hmm[1]
            _cfg = best_hmm[2]
            _ct = _cfg["covariance_type"]
            _mc = _cfg["min_covar"]
            print(f"  HMM: no config fully converged; best LL ({_ct}/{_mc}) LL={best_ll:.1f}")

        if self.hmm is None:
            raise RuntimeError(
                f"HMM fit failed for all covariance configs: {last_err}"
            ) from last_err
        reg_oh = np.eye(self.n_components, dtype=float)[regimes]

        # FIX: Use the safely isolated X_numeric array instead of the raw X dataframe
        X_xgb = np.hstack([X_numeric.to_numpy(dtype=float), reg_oh])
        y_cls = y.map(self.y_to_cls_).to_numpy(dtype=int)

        # --- Regularized XGB with PURGED time-tail early stopping (anti-overfit) ---
        # Mirrors Explorer notebook cell 9.  Lower capacity + stronger L1/L2 + subsampling;
        # early stopping selects the tree count from a held-out purged TAIL of the training
        # rows.  Guarded so a differing XGBoost API falls back to a full regularized fit.
        xgb_kwargs = dict(
            objective="multi:softprob",
            num_class=3,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            random_state=self.random_state,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,
        )

        n_rows = X_xgb.shape[0]
        purge = 50
        es_ok = False
        if self.early_stopping_rounds and n_rows >= 800:
            cut = int(n_rows * (1.0 - self.es_val_fraction))
            tr_hi = max(0, cut - purge)
            if tr_hi >= 400 and (n_rows - cut) >= 200:
                X_tr, y_tr = X_xgb[:tr_hi], y_cls[:tr_hi]
                X_val, y_val = X_xgb[cut:], y_cls[cut:]
                try:
                    _clf = xgb.XGBClassifier(early_stopping_rounds=int(self.early_stopping_rounds), **xgb_kwargs)
                    _clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    self.xgb = _clf
                    es_ok = True
                except Exception:
                    es_ok = False

        if not es_ok:
            self.xgb = xgb.XGBClassifier(**xgb_kwargs)
            self.xgb.fit(X_xgb, y_cls)

        return self

    def _augment(self, X: pd.DataFrame) -> np.ndarray:
        # Use self.feature_names_ which now safely contains ONLY numeric features
        X_num = X[self.feature_names_].to_numpy(dtype=float)
        X_hmm = X[self.hmm_features_].to_numpy(dtype=float)
        X_hmm = self.hmm_scaler.transform(X_hmm)
        regimes = self.hmm.predict(X_hmm)
        reg_oh = np.eye(self.n_components, dtype=float)[regimes]
        return np.hstack([X_num, reg_oh])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aug = self._augment(X)
        cls = self.xgb.predict(X_aug).astype(int)
        y = np.array([self.cls_to_y_[int(c)] for c in cls], dtype=np.int8)
        return y

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        X_aug = self._augment(X)
        return self.xgb.predict_proba(X_aug)
