"""src/notebook_runner.py -- faithful executor for the bundle notebooks.

Runs the code cells of Strategy_Tester.ipynb / GoldRegimeX_Explorer.ipynb inside
a shared namespace, exactly as Jupyter would, so the .py pipeline modules reuse
the notebooks' OWN code (the most robust source, per the brief) instead of a
re-authored copy.  Jupyter line/cell magics are neutralised and get_ipython /
display shims are injected.  Cell indices are RAW notebook indices (matching the
landmark constants below).  os.chdir moves to the repo root so every relative
path resolves like a notebook run -- which also preserves the Strategy-Tester ->
Explorer handoff via reports/strategy_winners_for_explorer.csv.

Two robustness measures for headless terminal runs:
 1. Each code cell is materialised to a REAL .py file under .grx_nbcache/ and
    compiled from that path, because numba's @njit(cache=True) needs a real
    source file on disk to build its cache locator (a synthetic "<cell>" name
    raises "cannot cache function ...: no locator available").
 2. Optional VIZ/progress-only libraries (seaborn, plotly, tqdm) that are not
    installed are replaced with a no-op stub so a missing plotting dependency
    can't abort training/backtesting.  matplotlib is forced to the headless
    "Agg" backend so plot cells never open a window or block the terminal.
    CORE libraries (numpy, pandas, sklearn, xgboost, hmmlearn, numba, scipy,
    joblib) are NEVER stubbed -- if one is missing you get a real error.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, re, sys, types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "pipeline_verification_bundle"
CELL_SRC_DIR = REPO_ROOT / ".grx_nbcache"   # real cell sources for numba caching
NOTEBOOKS = {
    "strategy_tester": BUNDLE / "Strategy_Tester.ipynb",
    "explorer": BUNDLE / "GoldRegimeX_Explorer.ipynb",
}

EXPLORER_EXPORT_CELL = 22       # exports models/goldregimex_live_model.pkl
EXPLORER_LIVESIM_CELL = 24      # Live Trading Simulation == backtester mirror
EXPLORER_PARITY_CELL = 28       # MT5 parity / engine self-check
EXPLORER_MONTECARLO_CELL = 30   # Monte Carlo robustness verdict

# Optional, non-essential (viz / progress) modules that are safe to stub when
# absent.  Never add a core numeric/ML dependency here.
OPTIONAL_STUB_MODULES = ("seaborn", "plotly", "tqdm")

_MAGIC = re.compile(r"^\s*%{1,2}[A-Za-z]")
_SHELL = re.compile(r"^\s*![A-Za-z./_]")


def _strip_magics(src):
    out = []
    for ln in src.splitlines():
        if _MAGIC.match(ln) or _SHELL.match(ln):
            out.append("pass  # [notebook magic stripped] " + ln.strip())
        else:
            out.append(ln)
    return "\n".join(out)


class _IPShim:
    def run_line_magic(self, *a, **k):
        return None
    def run_cell_magic(self, *a, **k):
        return None
    def system(self, *a, **k):
        return None
    def magic(self, *a, **k):
        return None
    def __getattr__(self, name):
        return lambda *a, **k: None


def _get_ipython():
    return _IPShim()


class _Stub:
    """A permissive no-op stand-in for a missing optional viz module.

    Any attribute access returns another _Stub; calling it returns a _Stub; it is
    iterable (empty), indexable, and usable as a context manager.  This lets
    notebook plotting code such as ``sns.set_style(...)`` or
    ``ax = sns.heatmap(...); ax.set_title(...)`` run without raising.
    """

    def __init__(self, name="stub"):
        self.__name__ = name

    def __call__(self, *a, **k):
        return self
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return _Stub("%s.%s" % (self.__name__, n))
    def __iter__(self):
        return iter(())
    def __len__(self):
        return 0
    def __getitem__(self, k):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def __bool__(self):
        return False
    def __repr__(self):
        return "<stub %s>" % self.__name__


def _install_stub_module(modname):
    stub = _Stub(modname)
    mod = types.ModuleType(modname)
    mod.__dict__["__grx_stub__"] = True
    mod.__getattr__ = lambda n, _s=stub: getattr(_s, n)
    sys.modules[modname] = mod
    return mod


def _prepare_optional_env(quiet=False):
    """Force headless matplotlib + stub any missing optional viz modules."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    for name in OPTIONAL_STUB_MODULES:
        if name in sys.modules:
            continue
        try:
            found = importlib.util.find_spec(name) is not None
        except Exception:
            found = False
        if not found:
            _install_stub_module(name)
            if not quiet:
                print("[notebook_runner] optional module '%s' not installed -- "
                      "using a no-op stub (plots from it are skipped)." % name)


def _prepare_syspath():
    for p in (REPO_ROOT, BUNDLE, BUNDLE / "shared"):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


def _apply_overrides(src, overrides):
    if not overrides:
        return src
    for k, v in overrides.items():
        src = re.sub(r"(?m)^(%s)\s*=.*$" % re.escape(k), "%s = %r" % (k, v), src)
    return src


def _materialise_cell(name, raw_idx, src):
    """Write the (transformed) cell source to a REAL .py file and return its path.
    numba needs a real on-disk source file for @njit(cache=True)."""
    try:
        CELL_SRC_DIR.mkdir(exist_ok=True)
        digest = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
        cell_path = CELL_SRC_DIR / ("%s_cell%02d_%s.py" % (name, raw_idx, digest))
        if not cell_path.exists() or cell_path.read_text(encoding="utf-8") != src:
            cell_path.write_text(src, encoding="utf-8")
        return str(cell_path)
    except Exception:
        return "<%s#cell%d>" % (name, raw_idx)


def _exec_cell(src, filename, ns, name, raw_idx):
    """Exec a cell; if it fails importing an allowlisted optional viz module,
    stub that module and retry once."""
    try:
        exec(compile(src, filename, "exec"), ns)
        return
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None)
        base = (missing or "").split(".")[0]
        if base in OPTIONAL_STUB_MODULES and base not in sys.modules:
            _install_stub_module(base)
            print("[notebook_runner] cell %d: stubbed missing optional module '%s' "
                  "and retried." % (raw_idx, base))
            exec(compile(src, filename, "exec"), ns)
            return
        raise


def _sync_notebook_defs_to_main(ns):
    """Mirror notebook-defined classes/functions onto the real __main__ module.

    Notebook cells run with __name__ == "__main__", so classes they define (e.g.
    HMMXGBComposite in Explorer cell 9) get __module__ == "__main__".  pickle
    serialises classes BY REFERENCE -- it looks them up as attributes of
    sys.modules["__main__"] (i.e. main.py).  Because the cells execute in the
    runner's own dict rather than the real module object, that lookup fails with:
        Can't pickle <class '__main__.HMMXGBComposite'>: not found as __main__...
    Copying the exact class/function objects onto the real __main__ module makes
    the reference resolve to the SAME object, so exporting the live model bundle
    (Explorer cell 22) works -- exactly like Jupyter, where the user namespace IS
    the __main__ module dict.  Only top-level runs (namespace is None) sync.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None or getattr(main_mod, "__dict__", None) is ns:
        return
    for k, v in list(ns.items()):
        if k.startswith("__"):
            continue
        if isinstance(v, type) or isinstance(v, types.FunctionType):
            if getattr(v, "__module__", None) == "__main__":
                try:
                    setattr(main_mod, k, v)
                except Exception:
                    pass


def run_notebook(name, namespace=None, overrides=None, only=None,
                 stop_after=None, allow_fail=False, quiet=False):
    """Execute a bundle notebook's code cells in a shared namespace.

    only        : iterable of RAW cell indices to run (defs-only extraction).
    stop_after  : run every code cell up to and including this RAW index.
    allow_fail  : keep going if a cell raises (defs-only extraction when optional
                  deps such as numba are missing).
    """
    path = NOTEBOOKS.get(name)
    path = Path(path) if path else Path(name)
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns = namespace if namespace is not None else {}
    ns.setdefault("__name__", "__main__")
    ns.setdefault("get_ipython", _get_ipython)
    ns.setdefault("display", lambda *a, **k: None)
    only_set = set(only) if only is not None else None
    sync_main = namespace is None

    prev_cwd = Path.cwd()
    _prepare_syspath()
    _prepare_optional_env(quiet=quiet)
    os.chdir(REPO_ROOT)
    (REPO_ROOT / "reports").mkdir(exist_ok=True)
    (REPO_ROOT / "notebooks" / "reports").mkdir(parents=True, exist_ok=True)
    try:
        for raw_idx, cell in enumerate(nb["cells"]):
            if stop_after is not None and raw_idx > stop_after:
                break
            if cell.get("cell_type") != "code":
                continue
            if only_set is not None and raw_idx not in only_set:
                continue
            src = _apply_overrides(_strip_magics("".join(cell["source"])), overrides)
            filename = _materialise_cell(name, raw_idx, src)
            prev_file = ns.get("__file__")
            ns["__file__"] = filename
            try:
                _exec_cell(src, filename, ns, name, raw_idx)
                if sync_main:
                    _sync_notebook_defs_to_main(ns)
            except Exception as e:
                if allow_fail:
                    if not quiet:
                        print("[notebook_runner] cell %d skipped: %s" % (raw_idx, e))
                    if prev_file is not None:
                        ns["__file__"] = prev_file
                    continue
                raise
    finally:
        os.chdir(prev_cwd)
    return ns
