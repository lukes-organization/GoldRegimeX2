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
    from src.backtester import vectorized_backtest
    src_path = _project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from src.trade_lifecycle import config_for_tf  # type: ignore[import]
    from src.risk_manager import BROKER_CONFIGS
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