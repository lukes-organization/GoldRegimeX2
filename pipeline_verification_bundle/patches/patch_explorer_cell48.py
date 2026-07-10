#!/usr/bin/env python3
"""
patch_explorer_cell48.py

Fixes NameError: name 'TIMEFRAMES' is not defined in Pipeline Verification
cell #3 (Phases 2 + 3: Dataset Integrity + Train/OOS leakage) of
GoldRegimeX_Explorer.ipynb.

Root cause:
    That cell iterates `for tf in TIMEFRAMES:`, but TIMEFRAMES is defined in
    an earlier setup cell (cell 2). If the user runs the verification cells
    in isolation (kernel restart, then jumping straight to the verification
    block), TIMEFRAMES is not in globals.

Fix:
    Insert a tiny resolver at the top of the cell:
        - If TIMEFRAMES is already defined -> use it (no-op path).
        - Else if `pipeline` dict exists   -> derive TIMEFRAMES from its keys
          and expose it as a global so subsequent verification cells inherit
          it (they all reference TIMEFRAMES too, so this fix propagates).
        - Else raise a clear RuntimeError telling the user to run the
          earlier pipeline-build cells first.

    No other cells are touched. The do-not-modify list is respected.

Usage:
    python patch_explorer_cell48.py [path/to/GoldRegimeX_Explorer.ipynb]

- If no path is given, defaults to ./GoldRegimeX_Explorer.ipynb, then
  ./notebooks/GoldRegimeX_Explorer.ipynb.
- Writes a timestamped .bak next to the notebook before modifying.
- Locates the target cell by content signature, not by index.
- Idempotent: running twice is a no-op the second time.
- Validates every code cell with ast.parse before writing.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

CELL_SIGNATURE = "PIPELINE VERIFICATION - Phases 2 + 3: Dataset Integrity"
PATCH_MARKER = "FIX (2026-07-07-b): resolve TIMEFRAMES from pipeline"

# Anchor: the exact text we search for inside the cell.
OLD_ANCHOR = "if VERIFY_PIPELINE:\n    integrity_verifier = DatasetIntegrityVerifier()"

# Replacement: same anchor with a resolver block injected between the `if`
# and `integrity_verifier = ...`. Indentation matches the existing cell.
NEW_BLOCK = (
    "if VERIFY_PIPELINE:\n"
    "    # FIX (2026-07-07-b): resolve TIMEFRAMES from pipeline if not already\n"
    "    # defined. Lets verification cells run even after a kernel restart\n"
    "    # where the earlier constant-defining cells weren't re-executed.\n"
    "    if \"TIMEFRAMES\" not in globals():\n"
    "        if \"pipeline\" in globals() and isinstance(pipeline, dict) and pipeline:\n"
    "            TIMEFRAMES = list(pipeline.keys())\n"
    "            print(\"[verify] TIMEFRAMES inferred from pipeline:\", TIMEFRAMES)\n"
    "        else:\n"
    "            raise RuntimeError(\n"
    "                \"TIMEFRAMES is not defined and 'pipeline' dict is unavailable. \"\n"
    "                \"Run the earlier notebook cells that build the pipeline first.\"\n"
    "            )\n"
    "    integrity_verifier = DatasetIntegrityVerifier()"
)


def resolve_notebook_path(cli_arg):
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


def find_target_cell(nb):
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


def validate_all_code_cells(nb):
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


def main():
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

    if OLD_ANCHOR not in old_src:
        sys.exit(
            f"ERROR: expected anchor not found in cell {idx}. The cell may have\n"
            "been hand-edited. Bailing out rather than guessing.\n"
            "Expected anchor:\n" + OLD_ANCHOR
        )

    new_src = old_src.replace(OLD_ANCHOR, NEW_BLOCK, 1)

    # Split back into ipynb source-lines format (each line keeps its trailing \n).
    lines = new_src.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    nb["cells"][idx]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
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

    print(f"Patched cell {idx} in place. {len(old_src)} chars -> {len(new_src)} chars.")
    print("Reload the notebook in Jupyter (File -> Revert / close & reopen).")


if __name__ == "__main__":
    main()
