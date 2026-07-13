"""audit_notebooks.py -- PHASE 1 local notebook audit tool.

This produces the Phase 1 deliverable required by the FINAL IMPLEMENTATION
PROMPT. The audit MUST run on the machine where the notebooks live because
remote tools cannot reliably fetch full multi-megabyte .ipynb files.

Usage (from repo root):

    python pipeline_verification_bundle/audit_notebooks.py

Outputs (under reports/audit/):

    * strategy_tester_audit.md   -- per-cell walkthrough of the Strategy Tester
    * explorer_audit.md          -- per-cell walkthrough of the Explorer
    * feature_checklist.md       -- Already / Partially / Missing table for
                                    every one of the 18 requested features
    * audit_summary.json         -- machine-readable version of both

The checklist search is heuristic (regex + AST scan) but conservative: it
only reports "Already implemented" when a matching artifact is actually
found in the notebook. Anything ambiguous is flagged "Partially implemented"
with the evidence, so the researcher can adjudicate.

No notebook contents are modified by this script.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_TARGETS = {
    "Strategy Tester": "pipeline_verification_bundle/Strategy_Tester_fixed.ipynb",
    "Explorer":        "pipeline_verification_bundle/GoldRegimeX_Explorer_fixed.ipynb",
}

# ---------------------------------------------------------------------------
# Feature signatures used to detect existing implementations.
# For each feature we accept several alternative signals so the audit doesn't
# false-negative simply because the user chose a different variable name.
# ---------------------------------------------------------------------------
FEATURE_SIGNATURES: Dict[str, Dict[str, Any]] = {
    "CandidateTrace (Phase 2)": {
        "any": [r"class\s+CandidateTrace\b", r"@dataclass[^\n]*\n\s*class\s+CandidateTrace"],
        "strong": [r"class\s+CandidateTrace\b.*\n(?:.*\n){0,40}?.*rejection_stage"],
    },
    "PipelineLogger (Phase 3)": {
        "any": [r"class\s+PipelineLogger\b", r"class\s+PipelineProfiler\b",
                r"logger\.log\(.*stage", r"profiler\.record\("],
    },
    "candidate_decisions.csv (Phase 4)": {
        "any": [r"candidate_decisions\.csv", r"decisions[_\.]to_csv"],
        "strong": [r"candidate_decisions\.csv"],
    },
    "pipeline_audit.json (Phase 5)": {
        "any": [r"pipeline_audit\.json"],
    },
    "Stage survival matrix (Phase 6)": {
        "any": [r"stage_survival", r"survival[_ ]matrix", r"funnel_table"],
    },
    "feature_drift_report.csv (Phase 7)": {
        "any": [r"feature_drift(?:_report)?\.csv", r"\bPSI\b", r"ks_2samp"],
        "strong": [r"feature_drift_report\.csv"],
    },
    "HMM diagnostics (Phase 8)": {
        "any": [r"hmm_diagnostics", r"hmm[_ ]state[_ ]occupancy",
                r"transition_matrix", r"dwell[_ ]time"],
    },
    "probability_report.csv (Phase 9)": {
        "any": [r"probability_report\.csv", r"probability_summary\.csv",
                r"probability_diagnostics"],
        "strong": [r"probability_report\.csv"],
    },
    "session_audit.csv (Phase 10)": {
        "any": [r"session_audit\.csv", r"session[_ ]audit"],
    },
    "candidate_integrity.csv (Phase 11)": {
        "any": [r"candidate_integrity\.csv", r"candidate_reconciliation\.csv",
                r"reconcile_candidates"],
        "strong": [r"candidate_integrity\.csv"],
    },
    "top100_rejected_M15.csv (Phase 12)": {
        "any": [r"top100_rejected_M15\.csv", r"top_?n_?rejected", r"top100_rejected"],
        "strong": [r"top100_rejected_M15\.csv"],
    },
    "top100_rejected_M5.csv (Phase 13)": {
        "any": [r"top100_rejected_M5\.csv", r"top100_rejected"],
        "strong": [r"top100_rejected_M5\.csv"],
    },
    "Model UUID consistency (Phase 14)": {
        "any": [r"ModelUUIDTracker", r"model_uuid", r"uuid\.uuid4\(\)"],
    },
    "pipeline_manifest.json (Phase 15)": {
        "any": [r"pipeline_manifest\.json", r"PipelineManifest",
                r"feature_hash.*session_filter_hash.*strategy_hash"],
    },
    "Pipeline Health Dashboard (Phase 16)": {
        "any": [r"pipeline_health\.txt", r"PIPELINE HEALTH",
                r"render_health_dashboard", r"PipelineHealthDashboard"],
    },
}

# Stage signatures used in the per-cell walkthrough.
STAGE_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Candidate Generation",   [r"generate.*candidate", r"candidate.*generate", r"attach_candidate_id",
                                r"trend_pullback", r"strategy.*signal"]),
    ("Feature Engineering",    [r"feature[_ ]engineer", r"compute.*features?", r"build_features",
                                r"feature_matrix"]),
    ("Triple Barrier Method",  [r"\bTBM\b", r"triple[_ ]barrier", r"tbm_labels"]),
    ("Session Filtering",      [r"session_filter", r"is_in_session", r"session[_ ]window",
                                r"london|new_york|overlap"]),
    ("HMM",                    [r"\bHMM\b", r"hidden[_ ]markov", r"regime", r"gaussian[_ ]hmm"]),
    ("XGBoost",                [r"\bxgb\b", r"xgboost", r"XGBClassifier", r"XGBRegressor"]),
    ("Probability threshold",  [r"probability.*threshold", r"threshold.*probability",
                                r"prob_?threshold", r"proba?\s*>=?\s*[0-9\.]+"]),
    ("Risk Manager",           [r"risk[_ ]manager", r"position[_ ]size", r"max[_ ]exposure",
                                r"drawdown[_ ]guard"]),
    ("Execution",              [r"execute[_ ]trade", r"place_order", r"run_ml_filtered_backtest",
                                r"backtest", r"executed_trades"]),
]


@dataclass
class CellAudit:
    index: int
    cell_type: str
    detected_stages: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    defined_names: List[str] = field(default_factory=list)   # top-level assignments / defs / classes
    referenced_names: List[str] = field(default_factory=list)
    first_line: str = ""
    line_count: int = 0


@dataclass
class NotebookAudit:
    name: str
    path: str
    cells: List[CellAudit] = field(default_factory=list)
    stage_to_cells: Dict[str, List[int]] = field(default_factory=dict)
    feature_findings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_cells: int = 0


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _extract_names(src: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (imports, defined_top_level_names, referenced_names) via AST when possible."""
    imports: List[str] = []
    defined: List[str] = []
    referenced: set = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return imports, defined, sorted(referenced)
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.extend(f"{mod}.{a.name}" if mod else a.name for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.append(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            defined.append(elt.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                referenced.add(node.value.id)
    return imports, defined, sorted(referenced)


def _detect_stages(src: str) -> List[str]:
    hits = []
    for stage, patterns in STAGE_KEYWORDS:
        for pat in patterns:
            if re.search(pat, src, flags=re.IGNORECASE):
                hits.append(stage)
                break
    return hits


def audit_notebook(name: str, path: Path) -> NotebookAudit:
    audit = NotebookAudit(name=name, path=str(path))
    if not path.exists():
        return audit
    with open(path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    cells = nb.get("cells", [])
    audit.total_cells = len(cells)

    # Aggregate feature findings across the whole notebook.
    full_source_by_cell: List[str] = [_cell_source(c) for c in cells]
    joined_source = "\n\n".join(full_source_by_cell)

    for i, c in enumerate(cells):
        src = full_source_by_cell[i]
        ct = c.get("cell_type", "")
        stages = _detect_stages(src) if ct == "code" else []
        imports, defined, referenced = ([], [], [])
        if ct == "code":
            imports, defined, referenced = _extract_names(src)
        first_line = src.splitlines()[0] if src.strip() else ""
        ca = CellAudit(
            index=i, cell_type=ct, detected_stages=stages,
            imports=imports, defined_names=defined, referenced_names=referenced,
            first_line=first_line[:120], line_count=len(src.splitlines()),
        )
        audit.cells.append(ca)
        for s in stages:
            audit.stage_to_cells.setdefault(s, []).append(i)

    # Feature checklist (Phase 17): search whole notebook.
    for feature, sig in FEATURE_SIGNATURES.items():
        finding: Dict[str, Any] = {"status": "Missing", "cells": [], "evidence": []}
        strong_hits = []
        any_hits = []
        for pat in sig.get("strong", []):
            for i, s in enumerate(full_source_by_cell):
                if re.search(pat, s, flags=re.IGNORECASE):
                    strong_hits.append({"cell": i, "pattern": pat})
        for pat in sig.get("any", []):
            for i, s in enumerate(full_source_by_cell):
                if re.search(pat, s, flags=re.IGNORECASE):
                    any_hits.append({"cell": i, "pattern": pat})
        if strong_hits:
            finding["status"] = "Already implemented"
            finding["cells"] = sorted({h["cell"] for h in strong_hits})
            finding["evidence"] = strong_hits
        elif any_hits:
            finding["status"] = "Partially implemented"
            finding["cells"] = sorted({h["cell"] for h in any_hits})
            finding["evidence"] = any_hits
        audit.feature_findings[feature] = finding

    return audit


def render_walkthrough_md(audit: NotebookAudit) -> str:
    lines = [
        f"# {audit.name} Audit",
        "",
        f"* File: `{audit.path}`",
        f"* Total cells: {audit.total_cells}",
        "",
        "## Stage → Cell map (execution order proxy)",
        "",
    ]
    for stage, _ in STAGE_KEYWORDS:
        cells = audit.stage_to_cells.get(stage, [])
        cells_disp = ", ".join(str(c) for c in cells) if cells else "_not found_"
        lines.append(f"* **{stage}**: {cells_disp}")
    lines += ["", "## Per-cell summary", ""]
    lines.append("| Cell | Type | Stages | Defined | First line |")
    lines.append("| ---: | ---- | ------ | ------- | ---------- |")
    for ca in audit.cells:
        defined = ", ".join(ca.defined_names[:6]) + ("…" if len(ca.defined_names) > 6 else "")
        stages = ", ".join(ca.detected_stages) if ca.detected_stages else ""
        first = ca.first_line.replace("|", "\\|").strip()
        lines.append(f"| {ca.index} | {ca.cell_type} | {stages} | {defined} | `{first}` |")
    lines += ["", "## Inputs / Outputs / Shared / Exported (heuristic)", ""]
    # Aggregate imports and top-level defs across the notebook.
    all_imports = sorted({imp for ca in audit.cells for imp in ca.imports})
    all_defs    = sorted({d for ca in audit.cells for d in ca.defined_names})
    lines += [
        "### Imports (module inputs)",
        "",
        "\n".join(f"* `{i}`" for i in all_imports) or "_none detected_",
        "",
        "### Top-level definitions (exported objects / shared variables)",
        "",
        "\n".join(f"* `{d}`" for d in all_defs) or "_none detected_",
        "",
    ]
    return "\n".join(lines)


def render_feature_checklist_md(audits: Dict[str, NotebookAudit]) -> str:
    lines = [
        "# Phase 17 -- Implementation Feature Checklist",
        "",
        "| Feature | Strategy Tester | Explorer | Overall Status |",
        "| ------- | --------------- | -------- | -------------- |",
    ]
    for feature in FEATURE_SIGNATURES:
        row = [feature]
        overall_statuses = []
        for nb_name in ("Strategy Tester", "Explorer"):
            f = audits.get(nb_name, NotebookAudit(nb_name, "")).feature_findings.get(
                feature, {"status": "Missing", "cells": []},
            )
            cell_str = ",".join(str(c) for c in f["cells"][:4])
            row.append(f"{_status_glyph(f['status'])} {f['status']} (cells: {cell_str or '-'})")
            overall_statuses.append(f["status"])
        # Overall: worst status across notebooks (Missing > Partially > Already).
        rank = {"Already implemented": 0, "Partially implemented": 1, "Missing": 2}
        worst = max(overall_statuses, key=lambda s: rank[s])
        row.append(f"{_status_glyph(worst)} {worst}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    for nb_name, audit in audits.items():
        counts = {"Already implemented": 0, "Partially implemented": 0, "Missing": 0}
        for f in audit.feature_findings.values():
            counts[f["status"]] = counts.get(f["status"], 0) + 1
        lines.append(
            f"* **{nb_name}**: {counts.get('Already implemented',0)} implemented / "
            f"{counts.get('Partially implemented',0)} partial / {counts.get('Missing',0)} missing"
        )
    return "\n".join(lines)


def _status_glyph(status: str) -> str:
    return {
        "Already implemented":   "✅",
        "Partially implemented": "⚠️",
        "Missing":               "❌",
    }.get(status, "❓")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-tester", default=DEFAULT_TARGETS["Strategy Tester"])
    parser.add_argument("--explorer",        default=DEFAULT_TARGETS["Explorer"])
    parser.add_argument("--out", default="pipeline_verification_bundle/reports/audit",
                        help="Output directory for audit deliverables (default: pipeline_verification_bundle/reports/audit)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "Strategy Tester": Path(args.strategy_tester),
        "Explorer":        Path(args.explorer),
    }

    audits: Dict[str, NotebookAudit] = {}
    any_missing = False
    for name, path in targets.items():
        if not path.exists():
            print(f"  {name}: FILE NOT FOUND at {path}")
            any_missing = True
            audits[name] = NotebookAudit(name=name, path=str(path))
            continue
        print(f"  Auditing {name} at {path} ...")
        audits[name] = audit_notebook(name, path)

    # Emit per-notebook walkthrough markdown.
    for name, audit in audits.items():
        out_path = out_dir / (name.lower().replace(" ", "_") + "_audit.md")
        out_path.write_text(render_walkthrough_md(audit), encoding="utf-8")
        print(f"  Wrote {out_path}")

    # Emit feature checklist (Phase 17 shape).
    checklist_path = out_dir / "feature_checklist.md"
    checklist_path.write_text(render_feature_checklist_md(audits), encoding="utf-8")
    print(f"  Wrote {checklist_path}")

    # Machine-readable summary.
    summary = {
        "audits": {
            name: {
                "path": a.path,
                "total_cells": a.total_cells,
                "stage_to_cells": a.stage_to_cells,
                "feature_findings": a.feature_findings,
            }
            for name, a in audits.items()
        },
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Wrote {out_dir / 'audit_summary.json'}")

    if any_missing:
        print("\nOne or more notebooks were missing. Pass explicit paths with:")
        print("  python audit_notebooks.py --strategy-tester PATH --explorer PATH")
        return 2
    print("\nPhase 1 audit complete. Review pipeline_verification_bundle/reports/audit/feature_checklist.md")
    print("before proceeding to the observability instrument step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
