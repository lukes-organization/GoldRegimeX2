"""instrument_notebooks.py -- one-command observability injector.

Usage from repo root:

    python pipeline_verification_bundle/instrument_notebooks.py

What it does
------------

Inserts / updates tagged hook cells in each target notebook so the pipeline
observability module (shared/pipeline_observability.py) hooks into the
existing notebook code without changing any model, strategy, HMM, XGBoost,
or CPCV logic.

Inserted cells (identified by tag `pipeline_observability_v1`):

    * observability_init      -- constructs PipelineObservability, PipelineLogger,
                                 ModelUUIDTracker, PipelineManifest.
    * observability_finalize  -- hydrates from any existing PipelineProfiler,
                                 writes all 18-phase artifacts including the
                                 top-100 rejected CSVs, manifest, and health
                                 dashboard. Also invokes the Explorer-side
                                 manifest validation gate.

Backups
-------

On first modification, each target notebook is copied to <name>.orig.ipynb.
Re-runs are idempotent -- tagged hook cells are replaced in place, not
duplicated.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

DEFAULT_TARGETS = {
    "pipeline_verification_bundle/GoldRegimeX_Explorer_fixed.ipynb": "explorer",
    "pipeline_verification_bundle/Strategy_Tester_fixed.ipynb":       "strategy_tester",
}

HOOK_TAG = "pipeline_observability_v1"

# ---------------------------------------------------------------------------
# Init cell -- constructs all observability objects and exposes them as module-
# level globals so any downstream cell can call `logger.log(...)`,
# `model_uuid_tracker.mint_training_uuid(...)`, etc. WITHOUT needing to import
# anything.
# ---------------------------------------------------------------------------
INIT_CELL_SOURCE = [
    "# --- Pipeline Observability init (tag: pipeline_observability_v1) --------\n",
    "# Auto-inserted by instrument_notebooks.py. Do not edit manually --\n",
    "# re-run the injector to update.\n",
    "from pathlib import Path\n",
    "\n",
    "try:\n",
    "    from pipeline_verification_bundle.shared.pipeline_observability import (\n",
    "        PipelineObservability, PipelineLogger, ModelUUIDTracker,\n",
    "        PipelineManifest, CandidateTrace,\n",
    "    )\n",
    "except ImportError:\n",
    "    import sys as _sys\n",
    "    _sys.path.insert(0, str(Path.cwd()))\n",
    "    from pipeline_verification_bundle.shared.pipeline_observability import (\n",
    "        PipelineObservability, PipelineLogger, ModelUUIDTracker,\n",
    "        PipelineManifest, CandidateTrace,\n",
    "    )\n",
    "\n",
    "OBS_OUTPUT_DIR = Path(\"reports/observability\")\n",
    "OBS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# Phase 2 & 3 -- lifecycle ledger + unified event logger.\n",
    "obs = PipelineObservability(\n",
    "    output_dir=OBS_OUTPUT_DIR,\n",
    "    expected_session_by_tf={\"M15\": \"London_NY\", \"M5\": \"London_NY\"},\n",
    "    material_survival_gap_pct=10.0,\n",
    "    lost_trade_limit=100,\n",
    "    lost_trade_tf=\"M15\",\n",
    ")\n",
    "logger = PipelineLogger(obs.ledger, obs.decisions)\n",
    "\n",
    "# Phase 14 -- model UUID consistency tracker.\n",
    "model_uuid_tracker = ModelUUIDTracker()\n",
    "\n",
    "# Phase 15 -- cross-notebook manifest.\n",
    "pipeline_manifest = PipelineManifest(\n",
    "    path=Path(\"reports/pipeline_manifest.json\"),\n",
    ")\n",
    "\n",
    "print(f\"[observability] run_id={obs.run_id} -> {OBS_OUTPUT_DIR}\")\n",
    "print(\"[observability] logger, model_uuid_tracker, pipeline_manifest are now defined.\")\n",
]

# ---------------------------------------------------------------------------
# Explorer-only cell that runs immediately after init: validates that the
# Strategy Tester wrote a matching pipeline_manifest.json (Phase 15).
# On the Strategy Tester side we do NOT insert this cell; instead the finalize
# hook writes the manifest.
# ---------------------------------------------------------------------------
EXPLORER_MANIFEST_VALIDATE_CELL = [
    "# --- Pipeline Manifest validation (tag: pipeline_observability_v1) -------\n",
    "# Explorer must validate the manifest written by Strategy Tester.\n",
    "# Abort the run if hashes do not match.\n",
    "_manifest_path = Path(\"reports/pipeline_manifest.json\")\n",
    "if _manifest_path.exists():\n",
    "    try:\n",
    "        _loaded = pipeline_manifest.load()\n",
    "        print(\"[manifest] loaded:\", {k: _loaded[k] for k in _loaded if k != 'extras'})\n",
    "        # If any of the following hash variables are already defined by the\n",
    "        # notebook, cross-check them against the manifest. Otherwise, just\n",
    "        # print the manifest for the researcher to inspect.\n",
    "        _expected = {}\n",
    "        for _k in ('feature_hash','session_filter_hash','strategy_hash',\n",
    "                   'candidate_hash','model_hash','pipeline_version'):\n",
    "            if _k in dir():\n",
    "                _expected[_k] = eval(_k)\n",
    "        if _expected:\n",
    "            _report = pipeline_manifest.validate(_expected, strict=False)\n",
    "            print(\"[manifest] validation:\", _report['status'])\n",
    "            if _report['status'] != 'PASS':\n",
    "                for _m in _report['mismatches']:\n",
    "                    print(\"  MISMATCH:\", _m)\n",
    "                raise RuntimeError(\"Pipeline manifest mismatch -- aborting Explorer run.\")\n",
    "    except FileNotFoundError:\n",
    "        print(\"[manifest] no manifest found; skipping validation.\")\n",
    "else:\n",
    "    print(\"[manifest] no manifest at\", _manifest_path, \"-- run Strategy Tester first.\")\n",
]

# ---------------------------------------------------------------------------
# Finalize cell (Strategy Tester side writes the manifest, Explorer side just
# reads it, but both call obs.finalize()).
# ---------------------------------------------------------------------------
def _finalize_source(role: str) -> List[str]:
    lines = [
        "# --- Pipeline Observability finalize (tag: pipeline_observability_v1) ----\n",
        "# Hydrates from any existing PipelineProfiler traceability layer, then\n",
        "# writes all 18-phase artifacts to reports/observability/.\n",
        "try:\n",
        "    _existing_profiler = profiler  # from the traceability layer, if any\n",
        "except NameError:\n",
        "    _existing_profiler = None\n",
        "if _existing_profiler is not None:\n",
        "    _hydrated = obs.import_from_profiler(_existing_profiler)\n",
        "    print(\"[observability] hydrated from PipelineProfiler:\", _hydrated)\n",
        "\n",
        "# Phase 14 -- verify model UUID consistency across training/eval/export.\n",
        "_uuid_report = model_uuid_tracker.verify_all()\n",
        "for _tf, _r in (_uuid_report or {}).items():\n",
        "    print(f\"[model-uuid] {_tf}: {_r['status']} uuids={_r['uuids']}\")\n",
        "\n",
        "# Wire in integrity results if present.\n",
        "_integrity = {}\n",
        "for _flag_name, _var_name in [\n",
        "    (\"Candidate Integrity\",  \"candidate_integrity_result\"),\n",
        "    (\"Model Integrity\",      \"model_integrity_result\"),\n",
        "    (\"Train/OOS Separation\", \"train_oos_separation_result\"),\n",
        "]:\n",
        "    if _var_name in dir():\n",
        "        _val = eval(_var_name)\n",
        "        _status = getattr(_val, \"status\", None) or (\"PASS\" if _val else \"FAIL\")\n",
        "        _integrity[_flag_name] = str(_status)\n",
        "# Fold model-UUID results into integrity flags.\n",
        "if _uuid_report:\n",
        "    all_pass = all(r['status'] == 'PASS' for r in _uuid_report.values())\n",
        "    _integrity['Model UUID Consistency'] = 'PASS' if all_pass else 'FAIL'\n",
        "\n",
        "_result = obs.finalize(integrity_flags=_integrity or None, verbose=True)\n",
    ]
    if role == "strategy_tester":
        lines += [
            "\n",
            "# Phase 15 -- Strategy Tester writes the pipeline manifest for Explorer.\n",
            "_hashes = {\n",
            "    'feature_hash':        PipelineManifest.hash_object(sorted(dir())),\n",
            "    'session_filter_hash': PipelineManifest.hash_object('London_NY|7-16|13-21'),\n",
            "    'strategy_hash':       PipelineManifest.hash_object('trend_pullback_v1'),\n",
            "    'candidate_hash':      PipelineManifest.hash_object(sorted(str(t.candidate_id) for t in obs.ledger.list_all())),\n",
            "    'model_hash':          PipelineManifest.hash_object(_uuid_report or {}),\n",
            "    'pipeline_version':    'v1.0.0',\n",
            "}\n",
            "# Allow the notebook to override any of these before this cell runs.\n",
            "for _k in list(_hashes.keys()):\n",
            "    if _k in dir():\n",
            "        _hashes[_k] = str(eval(_k))\n",
            "_mpath = pipeline_manifest.write(**_hashes)\n",
            "print(f\"[manifest] wrote {_mpath}\")\n",
        ]
    lines += [
        "\n",
        "print(\"\\nObservability artifacts written:\")\n",
        "for _k, _v in _result.items():\n",
        "    if _v is None or _k in (\"run_id\", \"survival_gap_warnings\"):\n",
        "        continue\n",
        "    print(f\"  {_k}: {_v}\")\n",
    ]
    return lines


def _cell_is_hook(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return HOOK_TAG in src


def _new_hook_cell(source_lines: List[str], cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"tags": [HOOK_TAG]},
        "outputs": [],
        "source": source_lines,
    }


def instrument_notebook(path: Path, role: str, *, dry_run: bool = False) -> dict:
    if not path.exists():
        return {"path": str(path), "status": "skipped_missing"}

    with open(path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)

    cells = nb.get("cells", [])
    n_before = len(cells)

    # Build target hook cells for this role.
    init_cell = _new_hook_cell(INIT_CELL_SOURCE, cell_id="obs_init_v1")
    hook_cells = [init_cell]
    if role == "explorer":
        hook_cells.append(_new_hook_cell(
            EXPLORER_MANIFEST_VALIDATE_CELL, cell_id="obs_manifest_validate_v1",
        ))
    finalize_cell = _new_hook_cell(_finalize_source(role), cell_id="obs_finalize_v1")

    # Remove any pre-existing hook cells (walk in reverse to preserve indices).
    hook_indices = [i for i, c in enumerate(cells) if _cell_is_hook(c)]
    for idx in reversed(hook_indices):
        del cells[idx]

    # Insert init cell (+ manifest validate for Explorer) right after cell 0.
    insert_at = 1 if len(cells) >= 1 else 0
    for i, hc in enumerate(hook_cells):
        cells.insert(insert_at + i, hc)
    # Append finalize cell at the end.
    cells.append(finalize_cell)
    action = "updated" if hook_indices else "inserted"

    nb["cells"] = cells

    backup_path = path.with_suffix(".orig.ipynb")
    made_backup = False
    if not backup_path.exists() and not dry_run:
        shutil.copy2(path, backup_path)
        made_backup = True

    if not dry_run:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")

    return {
        "path": str(path),
        "role": role,
        "status": action,
        "cells_before": n_before,
        "cells_after": len(cells),
        "backup_created": made_backup,
        "backup_path": str(backup_path),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*",
                        help="Notebook paths to instrument (default: standard pair).")
    parser.add_argument("--role", choices=["auto", "explorer", "strategy_tester"],
                        default="auto",
                        help="Force a role. 'auto' infers from filename.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.paths:
        targets: dict = {}
        for p in args.paths:
            role = args.role if args.role != "auto" else (
                "explorer" if "explorer" in Path(p).name.lower() else "strategy_tester"
            )
            targets[p] = role
    else:
        targets = dict(DEFAULT_TARGETS)

    print("Instrumenting notebooks with pipeline_observability hooks:")
    any_missing = False
    for p, role in targets.items():
        result = instrument_notebook(Path(p), role=role, dry_run=args.dry_run)
        if result["status"] == "skipped_missing":
            any_missing = True
            print(f"  {p}: (file not found)")
            continue
        print(f"  {result['path']}: {result['status']}  role={result['role']}  "
              f"cells {result['cells_before']} -> {result['cells_after']}",
              end="")
        if result.get("backup_created"):
            print(f"  backup={result['backup_path']}", end="")
        if result.get("dry_run"):
            print("  [dry-run: no write]", end="")
        print()

    if any_missing:
        print("\nOne or more notebooks were missing. Pass explicit paths, e.g.:")
        print("  python pipeline_verification_bundle/instrument_notebooks.py path/to/notebook.ipynb")
        return 2
    print("\nDone. Restart the notebook kernel and run all cells.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
