#!/usr/bin/env python3
"""
patch_explorer_cell46.py

Fixes the ModuleNotFoundError in the PIPELINE VERIFICATION import cell of
GoldRegimeX_Explorer.ipynb by making it auto-discover pipeline_verification.py
and shared/session_filter.py whether they sit in the project root or the
notebooks/ folder.

Usage:
    python patch_explorer_cell46.py [path/to/GoldRegimeX_Explorer.ipynb]

- If no path is given, defaults to ./GoldRegimeX_Explorer.ipynb, then
  ./notebooks/GoldRegimeX_Explorer.ipynb.
- Writes a timestamped .bak next to the notebook before modifying.
- Locates the target cell by content signature (not by index), so it still
  works even if you insert/remove cells above it.
- Idempotent: running again after a successful patch is a no-op.
- Validates every code cell with ast.parse before writing.

Do-not-modify list respected: no changes to Trend Pullback, TBM, CPCV,
HMM/XGBoost architecture, Risk Circuit Breaker, threshold optimisation,
or Grid Sensitivity Plateau logic. This only touches the verification
bootstrap cell.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Target cell signature - substring that uniquely identifies the cell we edit.
# ---------------------------------------------------------------------------
CELL_SIGNATURE = "PIPELINE VERIFICATION (ADDED) - Global flag + imports"

# Marker used to detect an already-patched cell (idempotency).
PATCH_MARKER = "FIX (2026-07-07): auto-discover pipeline_verification.py"

# ---------------------------------------------------------------------------
# New cell source.
# ---------------------------------------------------------------------------
NEW_CELL_SRC = '''\
# ============================================================
# PIPELINE VERIFICATION (ADDED) - Global flag + imports
# Phases 1, 8. Purely additive - subsequent cells gated on
# VERIFY_PIPELINE. Does not modify CPCV, HMM/XGBoost architecture,
# Strategy logic, TBM, Grid Sensitivity Plateau, Risk Circuit Breaker,
# or threshold optimisation logic.
#
# FIX (2026-07-07): auto-discover pipeline_verification.py and
# shared/session_filter.py regardless of whether they sit in the
# project root or the notebooks/ folder. Previously this cell
# raised ModuleNotFoundError when the files were placed next to
# the .ipynb but Cell 1 had already chdir\'d to the project root.
# ============================================================
import sys, os, json, uuid, hashlib
from pathlib import Path

VERIFY_PIPELINE = True  # set False to skip verification with zero runtime overhead

if VERIFY_PIPELINE:
    # Candidate directories to search for pipeline_verification.py and shared/.
    # Order matters: nearest to the notebook wins.
    _cwd = Path.cwd().resolve()
    _candidates = [
        _cwd,
        _cwd / "notebooks",
        _cwd.parent,
        _cwd.parent / "notebooks",
        Path.home() / "GoldRegime_X",
        Path.home() / "GoldRegime_X" / "notebooks",
    ]
    _seen = set(); _search = []
    for _p in _candidates:
        _s = str(_p)
        if _s not in _seen:
            _seen.add(_s); _search.append(_p)

    _pv_dir = None
    _sf_dir = None
    for _p in _search:
        if _pv_dir is None and (_p / "pipeline_verification.py").is_file():
            _pv_dir = _p
        if _sf_dir is None and (_p / "shared" / "session_filter.py").is_file():
            _sf_dir = _p
        if _pv_dir is not None and _sf_dir is not None:
            break

    _missing = []
    if _pv_dir is None:
        _missing.append("pipeline_verification.py")
    if _sf_dir is None:
        _missing.append("shared/session_filter.py")
    if _missing:
        _tried = "\\n  ".join(str(p) for p in _search)
        raise ModuleNotFoundError(
            "Could not locate: " + ", ".join(_missing) + ".\\n"
            "Place them alongside the notebook (project root OR notebooks/ folder).\\n"
            "Searched:\\n  " + _tried
        )

    for _dir in {_pv_dir, _sf_dir}:
        _s = str(_dir)
        if _s not in sys.path:
            sys.path.insert(0, _s)

    # Ensure shared/ is importable as a package even if __init__.py is missing.
    _shared_init = _sf_dir / "shared" / "__init__.py"
    if not _shared_init.is_file():
        try:
            _shared_init.write_text("")
        except Exception:
            pass

    from shared.session_filter import SessionFilter, run_session_filter_test_suite
    import pipeline_verification as pv
    from pipeline_verification import (
        DatasetIntegrityVerifier,
        FeatureLeakageVerifier,
        EvaluationVerifier,
        PredictionAlignmentVerifier,
        SessionVerifier,
        ThresholdVerifier,
        CandidateIntegrityVerifier,
        PipelineCertification,
        RootCauseAnalyzerV2,
        assign_model_uuid,
        verify_model_identity,
        assert_no_train_oos_leakage,
        train_hash_of,
        expected_calibration_error,
        build_pipeline_waterfall,
        build_manifest,
        verify_manifest_match,
        feature_set_hash,
        candidate_ids_hash,
    )
    session_filter = SessionFilter()
    certification = PipelineCertification()
    print("VERIFY_PIPELINE=True.")
    print("  pipeline_verification.py loaded from:", _pv_dir)
    print("  shared/session_filter.py loaded from:", _sf_dir / "shared")
    print("  SessionFilter hash:", session_filter.version_hash())
else:
    print("VERIFY_PIPELINE=False. Skipping all verification cells.")
'''


def resolve_notebook_path(cli_arg: str | None) -> Path:
    if cli_arg:
        p = Path(cli_arg).expanduser().resolve()
        if not p.is_file():
            sys.exit(f"ERROR: notebook not found at {p}")
        return p
    for candidate in (
        Path("GoldRegimeX_Explorer.ipynb"),
        Path("notebooks/GoldRegimeX_Explorer.ipynb"),
    ):
        if candidate.is_file():
            return candidate.resolve()
    sys.exit(
        "ERROR: no notebook path given and none of the defaults exist:\n"
        "  ./GoldRegimeX_Explorer.ipynb\n"
        "  ./notebooks/GoldRegimeX_Explorer.ipynb"
    )


def find_target_cell(nb: dict) -> int:
    matches = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if CELL_SIGNATURE in src:
            matches.append(i)
    if not matches:
        sys.exit(
            "ERROR: could not find target cell. Signature not found:\n  "
            + CELL_SIGNATURE
        )
    if len(matches) > 1:
        sys.exit(
            f"ERROR: signature matched {len(matches)} cells at indices {matches}. "
            "Refusing to guess."
        )
    return matches[0]


def validate_all_code_cells(nb: dict) -> None:
    failed = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            ast.parse(src)
        except SyntaxError as e:
            failed.append((i, str(e)))
    if failed:
        msg = "\n".join(f"  cell {i}: {err}" for i, err in failed)
        sys.exit(f"ERROR: syntax errors after patch:\n{msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", help="Path to Explorer notebook")
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip writing a .bak file"
    )
    args = parser.parse_args()

    nb_path = resolve_notebook_path(args.notebook)
    print(f"Patching: {nb_path}")

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    idx = find_target_cell(nb)
    old_src = "".join(nb["cells"][idx].get("source", []))

    if PATCH_MARKER in old_src:
        print(f"Cell {idx} already patched (marker present). No-op.")
        return

    # Replace cell source.
    new_lines = NEW_CELL_SRC.split("\n")
    if new_lines and new_lines[-1] == "":
        new_lines.pop()
    nb["cells"][idx]["source"] = [line + "\n" for line in new_lines[:-1]] + [
        new_lines[-1]
    ]
    # Clear stale execution output so the diff is clean.
    if "outputs" in nb["cells"][idx]:
        nb["cells"][idx]["outputs"] = []
    if "execution_count" in nb["cells"][idx]:
        nb["cells"][idx]["execution_count"] = None

    validate_all_code_cells(nb)

    if not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = nb_path.with_suffix(nb_path.suffix + f".bak_{ts}")
        shutil.copy2(nb_path, bak)
        print(f"Backup written: {bak}")

    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

    print(f"Patched cell {idx} in place. {len(old_src)} chars -> {len(NEW_CELL_SRC)} chars.")
    print("Reload the notebook in Jupyter (File -> Revert / close & reopen).")


if __name__ == "__main__":
    main()
