# -----------------------------
# HMM + XGBoost composite
# -----------------------------

class HMMXGBComposite(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_components=3,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
    ):
        self.n_components = int(n_components)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

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

        self.xgb = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=self.random_state,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,
        )
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