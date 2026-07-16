"""shared/pipeline_certification.py -- Pipeline Certification + Self-Validation.

Phase 13: run a fixed battery of checks over the reconstructed observability
layer and emit a single Pipeline Certification report. ANY failed check forces
Overall Status = FAIL.

Phase 14: `self_validate_notebooks` statically scans the notebooks to prove the
old architecture is gone (no duplicated feature/session/profiler classes, no
alias mapping tables, no union-based hydration).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from shared.pipeline_profiler import (
    PIPELINE_STAGES,
    PipelineIntegrityError,
    PipelineProfiler,
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _stage_order_ok() -> Check:
    expected = ["GENERATED", "FEATURE_ENGINEERING", "SESSION", "TBM", "HMM", "PROBABILITY", "RISK", "EXECUTED"]
    ok = PIPELINE_STAGES == expected
    return Check("Stage order", ok, "" if ok else "got %s" % PIPELINE_STAGES)


def _monotonic_ok(profiler: PipelineProfiler, timeframes: List[str]) -> Check:
    try:
        for tf in timeframes:
            profiler.assert_monotonic(tf)
        return Check("Stage counts monotonic", True)
    except PipelineIntegrityError as exc:
        return Check("Stage counts monotonic", False, str(exc))


def _unique_ids_ok(profiler: PipelineProfiler, timeframes: List[str]) -> Check:
    bad = [tf for tf in timeframes if not profiler.candidate_ids_unique_per_stage(tf)]
    return Check("Candidate IDs unique", not bad, "" if not bad else "dup events in %s" % bad)


def _no_stale_reports_ok(outdir: str, run_uuid: str) -> Check:
    if not os.path.isdir(outdir):
        return Check("No stale reports", False, "missing dir %s" % outdir)
    stale = []
    for name in os.listdir(outdir):
        if name == ".gitkeep":
            continue
        path = os.path.join(outdir, name)
        if not os.path.isfile(path):
            continue
        try:
            head = open(path, "r", errors="ignore").readline()
        except Exception:
            head = ""
        if ("run_uuid=%s" % run_uuid) not in head:
            stale.append(name)
    return Check("No stale reports", not stale, "" if not stale else "stale/foreign: %s" % stale)


def _oos_diag_ok(profiler: PipelineProfiler) -> Check:
    ok = profiler.dataset == "OOS"
    return Check("OOS diagnostics", ok, "" if ok else "report profiler dataset=%s" % profiler.dataset)


def _canonical_states_ok(observed_states) -> Check:
    uniq = sorted({int(s) for s in observed_states})
    bad = [s for s in uniq if s not in (0, 1, 2)]
    return Check("Canonical HMM states", not bad, "" if not bad else "non-canonical: %s" % bad)


def _shared_module_ok(name: str, module_obj: Any, marker: str) -> Check:
    ok = module_obj is not None and hasattr(module_obj, marker)
    return Check(name, ok, "" if ok else "missing %s" % marker)


def _model_uuid_ok(train_uuid: Optional[str], eval_uuid: Optional[str], export_uuid: Optional[str]) -> Check:
    vals = [train_uuid, eval_uuid, export_uuid]
    ok = all(v is not None for v in vals) and len(set(vals)) == 1
    return Check("Model UUID consistency", ok, "" if ok else "train=%s eval=%s export=%s" % tuple(vals))


def certify(
    report_profiler: PipelineProfiler,
    timeframes: List[str],
    reports_dir: str,
    observed_hmm_states,
    modules: Dict[str, Any],
    model_uuids: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Run all Phase-13 checks. `modules` must contain keys: session_filter,
    features, pipeline_profiler, prediction (the imported shared modules).
    `model_uuids` must contain keys train/eval/export."""
    checks: List[Check] = [
        _stage_order_ok(),
        _monotonic_ok(report_profiler, timeframes),
        _unique_ids_ok(report_profiler, timeframes),
        _no_stale_reports_ok(reports_dir, report_profiler.run_uuid),
        _oos_diag_ok(report_profiler),
        _canonical_states_ok(observed_hmm_states),
        _shared_module_ok("Shared Session Filter", modules.get("session_filter"), "SessionFilter"),
        _shared_module_ok("Shared Feature Engineering", modules.get("features"), "build_features"),
        _shared_module_ok("Shared Profiler", modules.get("pipeline_profiler"), "PipelineProfiler"),
        _shared_module_ok("Prediction Contract", modules.get("prediction"), "PredictionResult"),
        _model_uuid_ok(model_uuids.get("train"), model_uuids.get("eval"), model_uuids.get("export")),
    ]
    overall = "PASS" if all(c.passed for c in checks) else "FAIL"
    return {"overall": overall, "checks": [c.__dict__ for c in checks]}


def render_certification(result: Dict[str, Any]) -> str:
    lines = ["=" * 30 + "  PIPELINE CERTIFICATION  " + "=" * 30]
    for c in result["checks"]:
        mark = "\u2713" if c["passed"] else "\u2717"
        extra = "" if c["passed"] else "   <-- %s" % c["detail"]
        lines.append("  %s %-28s %s%s" % (mark, c["name"], "PASS" if c["passed"] else "FAIL", extra))
    lines.append("-" * 86)
    lines.append("  OVERALL STATUS: %s" % result["overall"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 14: static self-validation of the notebooks.
# ---------------------------------------------------------------------------
_FORBIDDEN_PATTERNS = {
    "union hydration": re.compile(r"setdefault\([^)]*\)\.update\("),
    "alias mapping (stage_alias)": re.compile(r"stage_alias"),
    "alias mapping (import_from_profiler)": re.compile(r"def\s+import_from_profiler"),
    "duplicate profiler class": re.compile(r"class\s+PipelineProfiler\b"),
    "duplicate CandidateTrade class": re.compile(r"class\s+CandidateTrade\b"),
    "duplicate session filter class": re.compile(r"class\s+SessionFilter\b"),
    "manual proba indexing": re.compile(r"predict_proba\w*\([^)]*\)\s*\[\s*:\s*,\s*\d+\s*\]"),
    "direct hmm.predict in diagnostics": re.compile(r"model\.hmm\.predict\("),
}
_REQUIRED_SUBSTRINGS = [
    "from shared.pipeline_profiler import",
    "from shared.session_filter import",
    "from shared.features import",
]


def self_validate_notebooks(notebook_paths: List[str]) -> Dict[str, Any]:
    violations: List[str] = []
    for path in notebook_paths:
        nb = json.load(open(path))
        text = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")
        for label, rx in _FORBIDDEN_PATTERNS.items():
            if rx.search(text):
                violations.append("%s: forbidden pattern present -> %s" % (os.path.basename(path), label))
        for req in _REQUIRED_SUBSTRINGS:
            if req not in text:
                violations.append("%s: missing required import -> %s" % (os.path.basename(path), req))
    return {"passed": not violations, "violations": violations}
