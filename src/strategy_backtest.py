"""src/strategy_backtest.py -- deployed-model signal for the live/demo app.

latest_signal(tf) reproduces the SAME decision the Explorer backtester makes on
the most recent bar:
  1. build the shared features on the freshest CSVs using the NOTEBOOK'S OWN
     load_panel + build_features (loaded via notebook_runner, defs only),
  2. run the deployed HMM+XGB model from the exported bundle,
  3. apply the plateau-selected probability threshold and the regime gate.
Because it uses the notebook's feature code + the exported model + threshold,
online signals match backtest -- the backtester is a mirror of live/demo.

Everything degrades gracefully: if the bundle or optional deps are unavailable it
returns None / a reason string, which the app surfaces to the user instead of
crashing.
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np

from . import ml_models
from . import notebook_runner as nr

BUNDLE_PKL = nr.REPO_ROOT / "pipeline_verification_bundle" / "models" / "goldregimex_live_model.pkl"

_BUNDLE = None
_DEFS = None


def _register_pickle_classes():
    "Expose the model classes under __main__ so the notebook-pickled bundle loads."
    main = sys.modules.get("__main__")
    if main is None:
        return
    for name in dir(ml_models):
        obj = getattr(ml_models, name)
        if isinstance(obj, type) and not hasattr(main, name):
            setattr(main, name, obj)


def load_live_bundle(path=None, force=False):
    """Load and cache the exported live model bundle."""
    global _BUNDLE
    if _BUNDLE is not None and not force:
        return _BUNDLE
    p = Path(path) if path else BUNDLE_PKL
    if not p.exists():
        raise FileNotFoundError(
            "Live model bundle not found at %s. Build it first: python main.py explore" % p)
    _register_pickle_classes()
    with open(p, "rb") as fh:
        _BUNDLE = pickle.load(fh)
    return _BUNDLE


def _explorer_defs():
    """Load the Explorer notebook's config + loaders + feature code (defs only)."""
    global _DEFS
    if _DEFS is None:
        import numpy as _np, pandas as _pd, math as _math, os as _os
        ns = {"np": _np, "pd": _pd, "math": _math, "os": _os}
        # RAW cells: 1 bootstrap, 2 ML shims, 3 config, 6 loaders, 7 features
        nr.run_notebook("explorer", namespace=ns, only=[1, 2, 3, 6, 7],
                        allow_fail=True, quiet=True)
        _DEFS = ns
    return _DEFS


def latest_signal(tf, df=None, bundle=None):
    """Return the deployed signal on the latest bar, or None if unavailable.

    dict shape: signal(-1/0/1), prob_up, prob_down, threshold, regime_code, atr,
    close, base_params, timestamp, timeframe, reason.
    """
    tf = str(tf).upper()
    try:
        b = bundle or load_live_bundle()
    except Exception:
        return None
    models = b.get("models", {})
    if tf not in models:
        return {"signal": 0, "timeframe": tf, "reason": "no deployed model for %s" % tf}
    model = models[tf]
    thr = float(b.get("thresholds", {}).get(tf, 0.5))
    base_params = dict(b.get("base_params", {}).get(tf, {}))
    try:
        defs = _explorer_defs()
        build_features = defs.get("build_features")
        load_panel = defs.get("load_panel")
        if build_features is None or load_panel is None:
            return {"signal": 0, "timeframe": tf,
                    "reason": "notebook feature code unavailable (install requirements)"}
        if df is None:
            df = load_panel(tf)
        feat = build_features(df, tf).dropna()
        if len(feat) == 0:
            return {"signal": 0, "timeframe": tf, "reason": "no feature rows"}
        proba = np.asarray(model.predict_proba_raw(feat.tail(1)))[-1]
        prob_down = float(proba[0])
        prob_up = float(proba[-1])
        last = feat.iloc[-1]
        regime_code = int(last["regime_code"]) if "regime_code" in feat.columns else 0
        if "atr" in feat.columns:
            atr_v = float(last["atr"])
        elif "atr14" in feat.columns:
            atr_v = float(last["atr14"])
        else:
            atr_v = float("nan")
        close = float(last["Close"]) if "Close" in feat.columns else float("nan")
        signal = 0
        if prob_up >= thr and prob_up >= prob_down:
            signal = 1
        elif prob_down >= thr and prob_down > prob_up:
            signal = -1
        if regime_code == 2:  # SHOCK regime: stand aside (matches backtester gate)
            signal = 0
        return {
            "signal": int(signal), "prob_up": prob_up, "prob_down": prob_down,
            "threshold": thr, "regime_code": regime_code, "atr": atr_v, "close": close,
            "base_params": base_params, "timestamp": str(feat.index[-1]),
            "timeframe": tf, "reason": "ok",
        }
    except Exception as e:
        return {"signal": 0, "timeframe": tf, "reason": "signal error: %s" % e}
