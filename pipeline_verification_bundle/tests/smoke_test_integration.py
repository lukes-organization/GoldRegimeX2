#!/usr/bin/env python3
"""Pre-flight smoke test for the Observability V2 bundle.

Run this from anywhere inside your repo BEFORE running the notebooks. It does
NOT need trained models, market data, xgboost, hmmlearn, numba or a Jupyter
kernel. It checks the two integration points that the sandbox could not verify:

  1. The shared/ modules import cleanly and expose the expected V2 surface.
  2. src/engine_hmm.py exposes predict_states(...) and its signature is
     compatible with the diagnostics call predict_states_canonical(model, X).

Usage:
    python smoke_test_integration.py

Exit code 0 = ready to run the notebooks. Non-zero = fix the reported item first.
"""
import importlib
import inspect
import os
import sys
from pathlib import Path

MARKER = "pipeline_verification_bundle"


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for cand in (start, *start.parents):
        if (cand / MARKER / "shared" / "pipeline_profiler.py").exists():
            return cand
    for cand in (start, *start.parents):
        if (cand / MARKER).is_dir():
            return cand
    raise FileNotFoundError("Could not locate %s from %s" % (MARKER, start))


def main() -> int:
    problems = []
    warnings = []

    # --- locate repo + set up import paths --------------------------------
    try:
        repo_root = find_repo_root(Path(__file__).parent)
    except FileNotFoundError:
        repo_root = find_repo_root(Path.cwd())
    print("[smoke] repo_root =", repo_root)
    for p in (repo_root, repo_root / MARKER, repo_root / MARKER / "shared"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # --- 1) shared modules import + expected surface ----------------------
    expected = {
        "shared.pipeline_profiler": [
            "PipelineEvent", "PipelineProfiler", "PipelineIntegrityError",
            "PIPELINE_STAGES", "write_oos_reports", "write_is_debug_report",
            "purge_reports_dir", "CandidateTrade", "make_candidate_id",
            "parameter_set_id", "RejectionReason",
        ],
        "shared.session_filter": ["SessionFilter", "ENABLE_SESSION_FILTER", "session_col_from_value"],
        "shared.features": ["build_features", "build_ml_features", "feature_hash"],
        "shared.prediction": ["PredictionResult", "make_prediction", "directional_probability"],
        "shared.pipeline_certification": ["certify", "render_certification", "self_validate_notebooks"],
    }
    for mod_name, attrs in expected.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            problems.append("import %s failed: %s" % (mod_name, e))
            continue
        missing = [a for a in attrs if not hasattr(mod, a)]
        if missing:
            problems.append("%s missing attributes: %s" % (mod_name, missing))
        else:
            print("[smoke] OK  %s" % mod_name)

    # Canonical stage order sanity.
    try:
        from shared.pipeline_profiler import PIPELINE_STAGES
        want = ["GENERATED", "FEATURE_ENGINEERING", "SESSION", "TBM",
                "HMM", "PROBABILITY", "RISK", "EXECUTED"]
        if list(PIPELINE_STAGES) != want:
            problems.append("PIPELINE_STAGES != canonical 8-stage order: %s" % (PIPELINE_STAGES,))
        else:
            print("[smoke] OK  PIPELINE_STAGES canonical order")
    except Exception as e:
        problems.append("PIPELINE_STAGES check failed: %s" % e)

    # --- 2) src.engine_hmm.predict_states signature (Phase 6) -------------
    try:
        eng = importlib.import_module("src.engine_hmm")
    except Exception as e:
        problems.append(
            "import src.engine_hmm failed (%s). "
            "Diagnostics need predict_states from the production HMM engine." % e
        )
        eng = None

    if eng is not None:
        ps = getattr(eng, "predict_states", None)
        if ps is None or not callable(ps):
            problems.append(
                "src.engine_hmm has no callable predict_states(...). "
                "Phase 6 requires the production canonical-inference function."
            )
        else:
            try:
                sig = inspect.signature(ps)
                params = [
                    p for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                required = [p for p in params if p.default is inspect._empty
                            and p.kind is not p.VAR_POSITIONAL]
                print("[smoke] found src.engine_hmm.predict_states%s" % (sig,))
                has_varargs = any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values())
                # predict_states_canonical calls predict_states(model, X_hmm)
                if not has_varargs and len(required) not in (2,):
                    if len(required) < 2:
                        warnings.append(
                            "predict_states takes %d required positional arg(s); the diagnostics "
                            "call passes 2 (model, X_hmm). If it actually expects a single arg "
                            "(e.g. a DataFrame), adjust predict_states_canonical in the bootstrap "
                            "cell accordingly." % len(required)
                        )
                    else:
                        warnings.append(
                            "predict_states takes %d required positional args; the diagnostics call "
                            "passes only (model, X_hmm). Provide the extra arg(s) or wrap it in "
                            "predict_states_canonical." % len(required)
                        )
                else:
                    print("[smoke] OK  predict_states signature is compatible with (model, X_hmm)")
            except (ValueError, TypeError) as e:
                warnings.append("could not introspect predict_states signature: %s" % e)

    # --- report -----------------------------------------------------------
    print("\n" + "=" * 64)
    if warnings:
        print("WARNINGS (review, may need a small tweak):")
        for w in warnings:
            print("  ! " + w)
    if problems:
        print("NOT READY - fix these first:")
        for p in problems:
            print("  X " + p)
        return 1
    print("READY - shared modules import and predict_states is reachable.")
    print("You can now run Strategy_Tester.ipynb then GoldRegimeX_Explorer.ipynb.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
