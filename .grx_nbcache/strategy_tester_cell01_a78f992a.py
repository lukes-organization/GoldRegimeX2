# === Shared library imports (feature engineering + session helpers) ===
# The strategy backtester reuses the shared feature-engineering and session
# helpers in pipeline_verification_bundle/shared so they stay identical across
# notebooks. This cell only locates the bundle and imports those helpers.
import os, sys
from pathlib import Path

_MARKER = "pipeline_verification_bundle"

def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for cand in (start, *start.parents):
        if (cand / _MARKER / "shared" / "features.py").exists():
            return cand
    for cand in (start, *start.parents):
        if (cand / _MARKER).is_dir():
            return cand
    raise FileNotFoundError("Could not locate %s from %s" % (_MARKER, start))

_search = []
for _v in ("__vsc_ipynb_file__", "__session__"):
    _val = globals().get(_v)
    if isinstance(_val, str) and _val:
        _search.append(Path(_val).parent)
_search.append(Path.cwd())
_repo_root = None
for _s in _search:
    try:
        _repo_root = _find_repo_root(_s); break
    except FileNotFoundError:
        continue
if _repo_root is None:
    raise FileNotFoundError("pipeline_verification_bundle not found from: %s" % _search)
os.chdir(_repo_root)
for _p in (_repo_root, _repo_root / _MARKER, _repo_root / _MARKER / "shared"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.features import build_features, ema, true_range, atr, rsi, adx
from shared.session_filter import session_col_from_value

print("[shared] repo_root =", _repo_root)