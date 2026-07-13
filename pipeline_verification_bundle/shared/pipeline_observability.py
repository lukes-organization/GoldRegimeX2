"""
Complete Pipeline Observability Layer for GoldRegime_X.

This module implements the 11-phase observability specification.

Design guarantees (per implementation prompt):
  * Does NOT modify HMM, XGBoost, CPCV, or strategy rules.
  * Every candidate trade is traceable birth->death.
  * Every rejection carries exactly one explicit reason.
  * All decisions are recorded; no stage silently discards candidates.

Usage (from a notebook cell, after your existing pipeline code):

    from shared.pipeline_observability import PipelineObservability

    obs = PipelineObservability(output_dir="reports/observability")

    # For every generated candidate:
    obs.record_generation(cid, tf="M15", timestamp=ts, strategy="Trend Pullback")

    # At each pipeline stage (call whichever apply to your pipeline):
    obs.record_stage(cid, tf, "Session",     passed=True/False, reason="Outside London")
    obs.record_stage(cid, tf, "TBM",         passed=True/False, reason="Label=NaN")
    obs.record_stage(cid, tf, "HMM",         passed=True/False, reason="State 2", hmm_state=2)
    obs.record_stage(cid, tf, "Probability", passed=True/False, reason="0.54 < 0.61",
                     probability=0.54, threshold=0.61)
    obs.record_stage(cid, tf, "Risk",        passed=True/False, reason="Spread cap")
    obs.record_stage(cid, tf, "Executed",    passed=True)

    # Diagnostics (call each after the corresponding pipeline step):
    obs.record_hmm_inference(tf, states=hmm_states_array, transmat=hmm_model.transmat_)
    obs.record_probability_snapshot(tf, stage="raw",         probabilities=raw_probs)
    obs.record_probability_snapshot(tf, stage="post_hmm",    probabilities=hmm_gated_probs)
    obs.record_probability_snapshot(tf, stage="post_thresh", probabilities=passed_probs)
    obs.record_feature_distributions(tf, is_df=train_feat, oos_df=oos_feat, feature_cols=[...])

    # End of run (all artifacts produced here):
    obs.finalize(
        integrity_flags={"Candidate Integrity": "PASS",
                         "Model Integrity": "PASS",
                         "Train/OOS Separation": "PASS"},
    )

Outputs produced under output_dir:
    candidate_decisions.csv    (Phase 2)
    stage_survival.csv         (Phase 3)
    hmm_diagnostics.txt        (Phase 4 + 8)
    probability_summary.csv    (Phase 5)
    probability_histograms.png (Phase 5, if matplotlib present)
    session_audit.csv          (Phase 6)
    feature_drift.csv          (Phase 7)
    lost_trades_m15.txt        (Phase 10)
    pipeline_health.txt        (Phase 11)
    pipeline_audit.json        (aggregated machine-readable artifact)
"""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib is optional; observability degrades gracefully if it's missing.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _MPL_OK = True
except Exception:
    _MPL_OK = False
    _plt = None


# =========================================================================
# PHASE 1 -- Candidate Lifecycle Ledger
# =========================================================================
@dataclass
class CandidateTrace:
    """One trace per generated candidate. Immutable identity, mutable stage flags.

    Matches the Phase 2 lifecycle spec verbatim. Some legacy field names
    (feature_engineering, probability, probability_pass) are preserved as
    aliases via @property so older code keeps working.
    """
    candidate_id: int
    timeframe: str
    timestamp: pd.Timestamp
    strategy: str
    generated: bool = True
    feature_engineering_pass: bool = False
    session_pass: bool = False
    session_name: Optional[str] = None
    tbm_pass: bool = False
    hmm_state: Optional[int] = None
    hmm_probability: Optional[float] = None
    hmm_pass: bool = False
    xgb_probability: Optional[float] = None
    threshold: Optional[float] = None
    threshold_pass: bool = False
    risk_pass: bool = False
    executed: bool = False
    rejection_stage: Optional[str] = None
    rejection_reason: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    # ---- Backwards-compat aliases --------------------------------------
    @property
    def feature_engineering(self) -> bool:
        return self.feature_engineering_pass

    @feature_engineering.setter
    def feature_engineering(self, v: bool) -> None:
        self.feature_engineering_pass = bool(v)

    @property
    def probability(self) -> Optional[float]:
        return self.xgb_probability

    @probability.setter
    def probability(self, v: Optional[float]) -> None:
        self.xgb_probability = None if v is None else float(v)

    @property
    def probability_pass(self) -> bool:
        return self.threshold_pass

    @probability_pass.setter
    def probability_pass(self, v: bool) -> None:
        self.threshold_pass = bool(v)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = pd.Timestamp(self.timestamp).isoformat()
        return d


# The canonical stage order. Every stage a candidate touches must appear here.
# Adding a stage in the future? Extend this list -- the survival matrix and
# health dashboard will pick it up automatically.
STAGE_ORDER: Tuple[str, ...] = (
    "Generated",
    "FeatureEngineering",
    "Session",
    "TBM",
    "HMM",
    "Probability",
    "Threshold",
    "Risk",
    "Executed",
)

# Human-friendly stage names used in reports (spec wording).
STAGE_DISPLAY_NAMES: Dict[str, str] = {
    "Generated":          "Generated",
    "FeatureEngineering": "Feature Engineering",
    "Session":            "Session",
    "TBM":                "TBM",
    "HMM":                "HMM",
    "Probability":        "Probability",
    "Threshold":          "Threshold",
    "Risk":               "Risk Manager",
    "Executed":           "Executed",
}

# Which CandidateTrace flag corresponds to each stage's "passed?" bit.
_STAGE_TO_FLAG: Dict[str, str] = {
    "Generated":          "generated",
    "FeatureEngineering": "feature_engineering_pass",
    "Session":            "session_pass",
    "TBM":                "tbm_pass",
    "HMM":                "hmm_pass",
    "Probability":        "threshold_pass",  # legacy: raw prob computed => set threshold_pass
    "Threshold":          "threshold_pass",
    "Risk":               "risk_pass",
    "Executed":           "executed",
}


class CandidateLedger:
    """Per-timeframe collection of CandidateTrace instances, keyed by candidate_id."""

    def __init__(self) -> None:
        self._traces: Dict[Tuple[str, int], CandidateTrace] = {}

    def _key(self, tf: str, cid: int) -> Tuple[str, int]:
        return (str(tf).upper(), int(cid))

    def create(self, cid: int, tf: str, timestamp, strategy: str) -> CandidateTrace:
        k = self._key(tf, cid)
        if k in self._traces:
            return self._traces[k]
        tr = CandidateTrace(
            candidate_id=int(cid),
            timeframe=str(tf).upper(),
            timestamp=pd.Timestamp(timestamp),
            strategy=str(strategy),
            generated=True,
        )
        self._traces[k] = tr
        return tr

    def get(self, cid: int, tf: str) -> Optional[CandidateTrace]:
        return self._traces.get(self._key(tf, cid))

    def all_for_tf(self, tf: str) -> List[CandidateTrace]:
        tf_u = str(tf).upper()
        return [t for (t_tf, _), t in self._traces.items() if t_tf == tf_u]

    def timeframes(self) -> List[str]:
        return sorted({t.timeframe for t in self._traces.values()})

    def as_frame(self) -> pd.DataFrame:
        if not self._traces:
            return pd.DataFrame(columns=list(CandidateTrace.__dataclass_fields__.keys()))
        return pd.DataFrame([t.to_dict() for t in self._traces.values()])

    def __len__(self) -> int:
        return len(self._traces)

    # ---- Compatibility shims used by PipelineLogger and helpers --------
    def add(self, trace: CandidateTrace) -> CandidateTrace:
        k = self._key(trace.timeframe, trace.candidate_id)
        if k not in self._traces:
            self._traces[k] = trace
        return self._traces[k]

    def list_all(self) -> List[CandidateTrace]:
        return list(self._traces.values())

    def list_by_tf(self, tf: str) -> List[CandidateTrace]:
        return self.all_for_tf(tf)

    def record_stage(self, cid: Any, tf: str, stage: str,
                     passed: bool, reason: str = "") -> CandidateTrace:
        """Mark a stage as passed/failed on the trace and populate rejection info."""
        # Coerce cid to int if possible (matches internal key type).
        try:
            cid_i = int(cid)
        except Exception:
            cid_i = hash(str(cid)) & 0x7fffffff
        tf_u = str(tf).upper()
        k = self._key(tf_u, cid_i)
        tr = self._traces.get(k)
        if tr is None:
            tr = CandidateTrace(
                candidate_id=cid_i, timeframe=tf_u,
                timestamp=pd.Timestamp.utcnow(), strategy="unknown",
            )
            self._traces[k] = tr
        flag = _STAGE_TO_FLAG.get(stage)
        if flag is not None:
            setattr(tr, flag, bool(passed))
        if not passed:
            if not reason:
                raise ValueError("record_stage requires a reason when passed=False")
            if tr.rejection_stage is None:  # first rejection wins
                tr.rejection_stage = stage
                tr.rejection_reason = reason
        return tr


# =========================================================================
# PHASE 2 -- Candidate Decision Log
# =========================================================================
class DecisionLog:
    """Append-only log of stage decisions.  One row per (candidate, stage)."""

    COLUMNS = ("candidate_id", "timeframe", "stage", "decision", "reason", "timestamp", "strategy")

    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []

    def append(
        self,
        candidate_id: int,
        timeframe: str,
        stage: str,
        decision: str,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None,
        strategy: str = "",
    ) -> None:
        self._rows.append({
            "candidate_id": int(candidate_id),
            "timeframe": str(timeframe).upper(),
            "stage": str(stage),
            "decision": str(decision).upper(),
            "reason": str(reason) if reason is not None else "",
            "timestamp": pd.Timestamp(timestamp).isoformat() if timestamp is not None else "",
            "strategy": str(strategy),
        })

    def as_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=list(self.COLUMNS))
        df = pd.DataFrame(self._rows)
        return df[list(self.COLUMNS)]

    def to_csv(self, path: os.PathLike) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.as_frame().to_csv(p, index=False)
        return p


# =========================================================================
# PHASE 3 -- Stage Survival Matrix
# =========================================================================
def build_stage_survival_matrix(
    ledger: CandidateLedger,
    stage_order: Sequence[str] = STAGE_ORDER,
    material_delta_pct: float = 10.0,
    drop_untouched_stages: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (matrix_df, warnings).  Warnings flag stages where survival differs
    materially between the timeframes present in the ledger.

    Survival semantics: a candidate survives *through* stage N iff its
    rejection_stage is None or comes strictly AFTER stage N in stage_order.
    This guarantees monotone non-increasing counts even when a pipeline
    skips some optional stages.

    material_delta_pct: absolute percentage-point gap that counts as "material".
    drop_untouched_stages: if True (default), remove stages that no candidate
        was ever rejected at AND that no ledger flag records passing through.
        This keeps the matrix focused on stages the pipeline actually uses.
    """
    tfs = ledger.timeframes()
    if not tfs:
        return pd.DataFrame({"Stage": list(stage_order)}), []

    stage_index = {s: i for i, s in enumerate(stage_order)}

    # Determine which stages any candidate has actually touched (either
    # rejected at, or explicitly passed via a truthy flag). Untouched stages
    # are optionally dropped so the matrix reflects reality.
    touched: set = set()
    for tf in tfs:
        for t in ledger.all_for_tf(tf):
            if t.rejection_stage is not None:
                touched.add(t.rejection_stage)
            for stage, flag in _STAGE_TO_FLAG.items():
                if getattr(t, flag, False):
                    touched.add(stage)

    effective_stages: List[str] = [
        s for s in stage_order
        if (not drop_untouched_stages) or (s in touched) or s == "Generated"
    ]

    per_tf_counts: Dict[str, List[int]] = {}
    for tf in tfs:
        traces = ledger.all_for_tf(tf)
        total_gen = len(traces)
        counts: List[int] = []
        for stage in effective_stages:
            si = stage_index[stage]
            surviving = 0
            for t in traces:
                rej = t.rejection_stage
                if rej is None:
                    # Never rejected -- survived every recorded stage.
                    if t.executed or si <= stage_index.get("Executed", si):
                        surviving += 1
                else:
                    if stage_index.get(rej, len(stage_order)) > si:
                        surviving += 1
            counts.append(surviving)
        if effective_stages and effective_stages[0] == "Generated":
            counts[0] = total_gen
        # "Executed" is authoritative: only candidates explicitly marked.
        if "Executed" in effective_stages:
            ei = effective_stages.index("Executed")
            counts[ei] = sum(1 for t in traces if t.executed)
        per_tf_counts[tf] = counts

    stage_order = tuple(effective_stages)  # for the row loop below

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for i, stage in enumerate(stage_order):
        row: Dict[str, Any] = {"Stage": stage}
        pct_by_tf: Dict[str, float] = {}
        for tf in tfs:
            remaining = per_tf_counts[tf][i]
            gen = per_tf_counts[tf][0] if per_tf_counts[tf][0] else 0
            pct = (remaining / gen * 100.0) if gen > 0 else 0.0
            row["%s Remaining" % tf] = int(remaining)
            row["%s %%" % tf] = round(pct, 2)
            pct_by_tf[tf] = pct
        # Material-delta warning across timeframe pairs.
        if len(tfs) >= 2:
            vals = list(pct_by_tf.values())
            gap = max(vals) - min(vals)
            if gap >= material_delta_pct and stage != "Generated":
                warnings.append(
                    "[%s] survival gap = %.1f pp across %s" % (
                        stage, gap, ", ".join("%s=%.1f%%" % (k, v) for k, v in pct_by_tf.items())
                    )
                )
        rows.append(row)

    return pd.DataFrame(rows), warnings


def bottleneck_for_tf(matrix: pd.DataFrame, tf: str) -> Tuple[Optional[str], float]:
    """Return (stage_name, drop_pct) for the largest single-stage drop for tf."""
    pct_col = "%s %%" % tf.upper()
    if pct_col not in matrix.columns or len(matrix) < 2:
        return None, 0.0
    pcts = matrix[pct_col].tolist()
    stages = matrix["Stage"].tolist()
    biggest_drop = 0.0
    biggest_stage: Optional[str] = None
    for i in range(1, len(pcts)):
        drop = pcts[i - 1] - pcts[i]
        if drop > biggest_drop:
            biggest_drop = drop
            biggest_stage = stages[i]
    return biggest_stage, biggest_drop


# =========================================================================
# PHASE 4 + 8 -- HMM State Diagnostics & Regime Transition Report
# =========================================================================
@dataclass
class HMMSnapshot:
    timeframe: str
    state_counts: Dict[int, int]
    state_occupancy_pct: Dict[int, float]
    transition_matrix: List[List[float]]
    mean_dwell_time: Dict[int, float]
    pass_by_state: Dict[int, int] = field(default_factory=dict)
    reject_by_state: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "state_counts": {int(k): int(v) for k, v in self.state_counts.items()},
            "state_occupancy_pct": {int(k): float(v) for k, v in self.state_occupancy_pct.items()},
            "transition_matrix": [[float(x) for x in row] for row in self.transition_matrix],
            "mean_dwell_time": {int(k): float(v) for k, v in self.mean_dwell_time.items()},
            "pass_by_state": {int(k): int(v) for k, v in self.pass_by_state.items()},
            "reject_by_state": {int(k): int(v) for k, v in self.reject_by_state.items()},
        }


def compute_hmm_snapshot(
    tf: str,
    states: Sequence[int],
    transmat: Optional[Sequence[Sequence[float]]] = None,
) -> HMMSnapshot:
    arr = np.asarray(list(states), dtype=int)
    if arr.size == 0:
        return HMMSnapshot(timeframe=str(tf).upper(), state_counts={},
                           state_occupancy_pct={}, transition_matrix=[], mean_dwell_time={})

    unique = sorted(set(arr.tolist()))
    counts = {int(s): int((arr == s).sum()) for s in unique}
    total = int(arr.size)
    occupancy = {s: (c / total) * 100.0 for s, c in counts.items()}

    # Transition matrix -- prefer the model's own transmat when supplied.
    if transmat is not None:
        tmat = [[float(x) for x in row] for row in np.asarray(transmat).tolist()]
    else:
        n = max(unique) + 1 if unique else 0
        raw = np.zeros((n, n), dtype=float)
        for a, b in zip(arr[:-1], arr[1:]):
            raw[int(a), int(b)] += 1.0
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        tmat = (raw / row_sums).tolist()

    # Mean dwell time: average length of contiguous runs per state.
    dwell: Dict[int, List[int]] = {s: [] for s in unique}
    if arr.size:
        run_state = int(arr[0])
        run_len = 1
        for v in arr[1:]:
            v = int(v)
            if v == run_state:
                run_len += 1
            else:
                dwell.setdefault(run_state, []).append(run_len)
                run_state = v
                run_len = 1
        dwell.setdefault(run_state, []).append(run_len)
    mean_dwell = {s: (float(np.mean(v)) if v else 0.0) for s, v in dwell.items()}

    return HMMSnapshot(
        timeframe=str(tf).upper(),
        state_counts=counts,
        state_occupancy_pct=occupancy,
        transition_matrix=tmat,
        mean_dwell_time=mean_dwell,
    )


def attach_pass_rates_by_state(snap: HMMSnapshot, ledger: CandidateLedger) -> None:
    """Fill snap.pass_by_state / reject_by_state from the ledger for snap.timeframe."""
    traces = ledger.all_for_tf(snap.timeframe)
    pass_ct: Dict[int, int] = {}
    reject_ct: Dict[int, int] = {}
    for t in traces:
        s = int(t.hmm_state)
        if s < 0:
            continue
        if t.hmm_pass:
            pass_ct[s] = pass_ct.get(s, 0) + 1
        else:
            reject_ct[s] = reject_ct.get(s, 0) + 1
    snap.pass_by_state = pass_ct
    snap.reject_by_state = reject_ct


def format_hmm_report(snap: HMMSnapshot) -> str:
    lines = ["TIMEFRAME %s" % snap.timeframe]
    for s in sorted(snap.state_counts):
        lines.append("  State %d : %d  (%.1f%%)" % (s, snap.state_counts[s], snap.state_occupancy_pct[s]))
    lines.append("  Transition Matrix:")
    for i, row in enumerate(snap.transition_matrix):
        lines.append("    row %d: %s" % (i, "  ".join("%.3f" % x for x in row)))
    lines.append("  Average Regime Duration:")
    for s in sorted(snap.mean_dwell_time):
        lines.append("    State %d : %.2f bars" % (s, snap.mean_dwell_time[s]))
    if snap.pass_by_state or snap.reject_by_state:
        lines.append("  Candidate outcome by state (accepted / rejected):")
        all_states = sorted(set(snap.pass_by_state) | set(snap.reject_by_state))
        for s in all_states:
            p = snap.pass_by_state.get(s, 0)
            r = snap.reject_by_state.get(s, 0)
            total = p + r
            rate = (p / total * 100.0) if total else 0.0
            lines.append("    State %d : %d / %d  (pass rate %.1f%%)" % (s, p, r, rate))
    return "\n".join(lines)


# =========================================================================
# PHASE 5 -- Probability Diagnostics
# =========================================================================
def probability_summary(probs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(list(probs), dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {k: float("nan") for k in ("n", "min", "p25", "median", "p75", "p95", "p99", "max", "mean", "std")}
    return {
        "n":      float(a.size),
        "min":    float(np.min(a)),
        "p25":    float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "p75":    float(np.quantile(a, 0.75)),
        "p95":    float(np.quantile(a, 0.95)),
        "p99":    float(np.quantile(a, 0.99)),
        "max":    float(np.max(a)),
        "mean":   float(np.mean(a)),
        "std":    float(np.std(a)),
    }


def plot_probability_histograms(
    snapshots: Dict[str, Dict[str, np.ndarray]],
    out_path: os.PathLike,
) -> Optional[Path]:
    """snapshots[tf][stage] -> np.ndarray of probabilities. Returns None if mpl absent."""
    if not _MPL_OK:
        return None
    tfs = list(snapshots.keys())
    stages = sorted({s for tf in tfs for s in snapshots[tf].keys()})
    if not tfs or not stages:
        return None
    fig, axes = _plt.subplots(len(tfs), len(stages), figsize=(4 * len(stages), 3 * len(tfs)), squeeze=False)
    for i, tf in enumerate(tfs):
        for j, stage in enumerate(stages):
            arr = snapshots[tf].get(stage)
            ax = axes[i][j]
            if arr is not None and len(arr) > 0:
                ax.hist(np.asarray(arr, dtype=float), bins=40, edgecolor="black", alpha=0.75)
            ax.set_title("%s / %s (n=%d)" % (tf, stage, 0 if arr is None else len(arr)))
            ax.set_xlabel("probability")
            ax.set_ylabel("count")
    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=120)
    _plt.close(fig)
    return p


# =========================================================================
# PHASE 7 -- Feature Drift Analysis (PSI, KS, mean shift, std shift)
# =========================================================================
def _psi(is_arr: np.ndarray, oos_arr: np.ndarray, bins: int = 10) -> float:
    is_arr = is_arr[~np.isnan(is_arr)]
    oos_arr = oos_arr[~np.isnan(oos_arr)]
    if is_arr.size == 0 or oos_arr.size == 0:
        return float("nan")
    edges = np.quantile(is_arr, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return float("nan")
    is_hist, _ = np.histogram(is_arr, bins=edges)
    oos_hist, _ = np.histogram(oos_arr, bins=edges)
    is_pct = is_hist / max(is_hist.sum(), 1)
    oos_pct = oos_hist / max(oos_hist.sum(), 1)
    eps = 1e-6
    is_pct = np.where(is_pct == 0, eps, is_pct)
    oos_pct = np.where(oos_pct == 0, eps, oos_pct)
    return float(np.sum((oos_pct - is_pct) * np.log(oos_pct / is_pct)))


def _ks_stat(is_arr: np.ndarray, oos_arr: np.ndarray) -> float:
    a = np.sort(is_arr[~np.isnan(is_arr)])
    b = np.sort(oos_arr[~np.isnan(oos_arr)])
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.union1d(a, b)
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def feature_drift_report(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    feature_columns: Sequence[str],
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
    ks_warn: float = 0.10,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for f in feature_columns:
        if f not in is_df.columns or f not in oos_df.columns:
            continue
        is_v = np.asarray(is_df[f], dtype=float)
        oos_v = np.asarray(oos_df[f], dtype=float)
        psi = _psi(is_v, oos_v)
        ks = _ks_stat(is_v, oos_v)
        is_c = is_v[~np.isnan(is_v)]
        oos_c = oos_v[~np.isnan(oos_v)]
        mean_shift = float(oos_c.mean() - is_c.mean()) if is_c.size and oos_c.size else float("nan")
        std_shift = float(oos_c.std() - is_c.std()) if is_c.size and oos_c.size else float("nan")
        if not math.isnan(psi) and psi >= psi_alert:
            flag = "ALERT"
        elif (not math.isnan(psi) and psi >= psi_warn) or (not math.isnan(ks) and ks >= ks_warn):
            flag = "WARN"
        else:
            flag = "OK"
        rows.append({
            "feature": f,
            "psi": round(psi, 6) if not math.isnan(psi) else float("nan"),
            "ks": round(ks, 6) if not math.isnan(ks) else float("nan"),
            "mean_shift": mean_shift,
            "std_shift": std_shift,
            "flag": flag,
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


# =========================================================================
# PHASE 6 -- Session Audit (rejection detail)
# =========================================================================
def session_audit_frame(
    decision_log: DecisionLog,
    ledger: CandidateLedger,
    broker_tz: str = "Europe/Athens",
    expected_session_by_tf: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    df = decision_log.as_frame()
    if df.empty:
        return pd.DataFrame(columns=[
            "candidate_id", "timeframe", "timestamp_utc", "broker_time",
            "detected_session", "expected_session", "reason",
        ])
    sess = df[(df["stage"] == "Session") & (df["decision"] == "FAIL")].copy()
    if sess.empty:
        return pd.DataFrame(columns=[
            "candidate_id", "timeframe", "timestamp_utc", "broker_time",
            "detected_session", "expected_session", "reason",
        ])
    rows: List[Dict[str, Any]] = []
    for _, r in sess.iterrows():
        tf = r["timeframe"]
        cid = int(r["candidate_id"])
        trace = ledger.get(cid, tf)
        ts = pd.Timestamp(trace.timestamp) if trace is not None else pd.Timestamp(r["timestamp"])
        try:
            ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            broker_time = ts_utc.tz_convert(broker_tz).strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            broker_time = str(ts)
        hour = ts.hour
        if 13 <= hour < 16:
            detected = "OVERLAP"
        elif 7 <= hour < 16:
            detected = "LONDON"
        elif 13 <= hour < 21:
            detected = "NEW_YORK"
        else:
            detected = "ASIA"
        expected = (expected_session_by_tf or {}).get(tf, "")
        rows.append({
            "candidate_id": cid,
            "timeframe": tf,
            "timestamp_utc": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "broker_time": broker_time,
            "detected_session": detected,
            "expected_session": expected,
            "reason": r["reason"],
        })
    return pd.DataFrame(rows)


# =========================================================================
# PHASE 9 -- Cross-Notebook Candidate Reconciliation
# =========================================================================
def reconcile_candidates(stage_counts: Dict[str, int]) -> pd.DataFrame:
    """stage_counts keyed by canonical stage names:
         Generated -> Exported -> Imported -> Processed -> Executed
    Any missing key is treated as 0. Returns per-transition losses."""
    order = ["Generated", "Exported", "Imported", "Processed", "Executed"]
    rows: List[Dict[str, Any]] = []
    prev = None
    for st in order:
        n = int(stage_counts.get(st, 0))
        row = {"stage": st, "count": n}
        if prev is not None:
            loss = prev - n
            row["loss_from_prev"] = loss
            row["loss_pct"] = round((loss / prev * 100.0) if prev > 0 else 0.0, 2)
        else:
            row["loss_from_prev"] = 0
            row["loss_pct"] = 0.0
        rows.append(row)
        prev = n
    return pd.DataFrame(rows)


# =========================================================================
# PHASE 10 -- Explain Top N Lost Trades (default 100, target M15)
# =========================================================================
def explain_lost_trades(
    ledger: CandidateLedger,
    timeframe: str = "M15",
    limit: int = 100,
) -> str:
    traces = ledger.all_for_tf(timeframe)
    # A candidate is "lost" if it did not reach the Executed stage.
    lost = [t for t in traces if not t.executed]
    # Preserve generation order (by timestamp then id).
    lost.sort(key=lambda t: (pd.Timestamp(t.timestamp), t.candidate_id))
    lost = lost[:int(limit)]

    def _fmt_stage(passed: bool, stage: str, rejected_at: Optional[str]) -> str:
        if rejected_at == stage:
            return "FAIL"
        return "PASS" if passed else "Not Evaluated"

    lines: List[str] = []
    for t in lost:
        rej_stage = t.rejection_stage or "(unknown)"
        _prob = t.probability
        _thr = t.threshold
        _prob_is_num = _prob is not None and not (isinstance(_prob, float) and math.isnan(_prob))
        _thr_is_num = _thr is not None and not (isinstance(_thr, float) and math.isnan(_thr))
        if _prob_is_num and _thr_is_num:
            prob_line = "Probability: %.4f (threshold %.4f)" % (float(_prob), float(_thr))
        elif _prob_is_num:
            prob_line = "Probability: %.4f (threshold N/A)" % float(_prob)
        else:
            prob_line = "Probability: Not Evaluated"
        lines.append(
            "Candidate %d\n"
            "  Strategy: %s\n"
            "  Time: %s\n"
            "  Session: %s\n"
            "  TBM: %s\n"
            "  HMM State: %s\n"
            "  HMM Decision: %s\n"
            "  %s\n"
            "  Risk: %s\n"
            "  Final Outcome: Rejected by %s (%s)" % (
                t.candidate_id,
                t.strategy,
                pd.Timestamp(t.timestamp).strftime("%Y-%m-%d %H:%M UTC"),
                _fmt_stage(t.session_pass, "Session", rej_stage),
                _fmt_stage(t.tbm_pass, "TBM", rej_stage),
                (str(t.hmm_state) if t.hmm_state is not None and t.hmm_state >= 0 else "Not Evaluated"),
                _fmt_stage(t.hmm_pass, "HMM", rej_stage),
                prob_line,
                _fmt_stage(t.risk_pass, "Risk", rej_stage),
                rej_stage,
                t.rejection_reason or "unspecified",
            )
        )
    header = "Top %d rejected %s candidates (of %d lost)\n%s\n" % (
        len(lines), timeframe, sum(1 for t in traces if not t.executed), "-" * 60,
    )
    return header + "\n\n".join(lines) if lines else header + "(none)"


# =========================================================================
# PHASE 11 -- Pipeline Health Dashboard
# =========================================================================
def render_health_dashboard(
    ledger: CandidateLedger,
    matrix: pd.DataFrame,
    integrity_flags: Optional[Dict[str, str]] = None,
) -> str:
    integrity_flags = integrity_flags or {}
    tfs = ledger.timeframes()
    lines: List[str] = []
    bar = "=" * 30
    lines.append(bar + " PIPELINE HEALTH " + bar)
    lines.append("")
    per_tf_status: Dict[str, bool] = {}
    for tf in tfs:
        traces = ledger.all_for_tf(tf)
        gen = len(traces)
        execd = sum(1 for t in traces if t.executed)
        surv = (execd / gen * 100.0) if gen else 0.0
        lines.append("%s Generated: %d" % (tf, gen))
        lines.append("%s Executed:  %d" % (tf, execd))
        lines.append("%s Survival:  %.2f%%" % (tf, surv))
        lines.append("")
        per_tf_status[tf] = execd > 0

    for tf in tfs:
        stage, drop = bottleneck_for_tf(matrix, tf)
        if stage is not None:
            lines.append("Largest %s Bottleneck: %s (-%.1f%%)" % (tf, stage, drop))
    if tfs:
        lines.append("")

    for name, flag in integrity_flags.items():
        lines.append("%s: %s" % (name, flag))

    # Overall status: PASS iff (a) every listed integrity flag is PASS AND
    # (b) at least one candidate executed per timeframe present.
    integrity_ok = all(str(v).upper() == "PASS" for v in integrity_flags.values()) if integrity_flags else True
    tf_ok = all(per_tf_status.values()) if per_tf_status else True
    overall = "PASS" if (integrity_ok and tf_ok) else "FAIL"
    lines.append("")
    lines.append("Pipeline Status: %s" % overall)
    lines.append("=" * (2 * len(bar) + len(" PIPELINE HEALTH ")))
    return "\n".join(lines)


# =========================================================================
# ORCHESTRATOR
# =========================================================================
class PipelineObservability:
    """Owns the ledger, decision log, and per-run diagnostic snapshots.

    Every recording method is idempotent-safe: calling it twice with the same
    (candidate, stage) key will overwrite the flag but append a second row to
    the DecisionLog so the history remains auditable.
    """

    def __init__(
        self,
        output_dir: os.PathLike = "reports/observability",
        run_id: Optional[str] = None,
        broker_tz: str = "Europe/Athens",
        expected_session_by_tf: Optional[Dict[str, str]] = None,
        material_survival_gap_pct: float = 10.0,
        psi_warn: float = 0.10,
        psi_alert: float = 0.25,
        lost_trade_limit: int = 100,
        lost_trade_tf: str = "M15",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.broker_tz = broker_tz
        self.expected_session_by_tf = dict(expected_session_by_tf or {})
        self.material_survival_gap_pct = float(material_survival_gap_pct)
        self.psi_warn = float(psi_warn)
        self.psi_alert = float(psi_alert)
        self.lost_trade_limit = int(lost_trade_limit)
        self.lost_trade_tf = str(lost_trade_tf).upper()

        self.ledger = CandidateLedger()
        self.decisions = DecisionLog()
        self.hmm_snapshots: Dict[str, HMMSnapshot] = {}
        self.probability_snapshots: Dict[str, Dict[str, np.ndarray]] = {}
        self.probability_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.feature_drift_frames: Dict[str, pd.DataFrame] = {}
        self.reconciliation_counts: Dict[str, Dict[str, int]] = {}
        self.thresholds: Dict[str, float] = {}
        self.model_hashes: Dict[str, str] = {}

    # ---- Phase 1 / 2 recording surface ----------------------------------
    def record_generation(self, candidate_id: int, tf: str, timestamp, strategy: str) -> CandidateTrace:
        tr = self.ledger.create(candidate_id, tf, timestamp, strategy)
        self.decisions.append(candidate_id, tf, "Generated", "PASS",
                              reason="generated by %s" % strategy,
                              timestamp=tr.timestamp, strategy=strategy)
        return tr

    def record_stage(
        self,
        candidate_id: int,
        tf: str,
        stage: str,
        passed: bool,
        reason: str = "",
        *,
        hmm_state: Optional[int] = None,
        probability: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        tr = self.ledger.get(candidate_id, tf)
        if tr is None:
            raise KeyError(
                "record_stage called for unknown candidate_id=%r tf=%r. "
                "Every candidate must be registered with record_generation() first." % (candidate_id, tf)
            )
        # Enforce: rejection => exactly one reason (Phase 2 rule).
        if not passed and not reason:
            raise ValueError(
                "Rejection at stage %r requires an explicit reason (candidate=%d, tf=%s)."
                % (stage, candidate_id, tf)
            )
        # Mutate the trace.
        flag = _STAGE_TO_FLAG.get(stage)
        if flag is not None:
            setattr(tr, flag, bool(passed))
        if hmm_state is not None:
            tr.hmm_state = int(hmm_state)
        if probability is not None:
            tr.probability = float(probability)
        if threshold is not None:
            tr.threshold = float(threshold)
        if not passed and tr.rejection_reason is None:
            # First rejection wins -- downstream stages are not re-evaluated.
            tr.rejection_reason = reason
            tr.rejection_stage = stage

        self.decisions.append(
            candidate_id=candidate_id,
            timeframe=tf,
            stage=stage,
            decision="PASS" if passed else "FAIL",
            reason=reason or "-",
            timestamp=tr.timestamp,
            strategy=tr.strategy,
        )

    def mark_executed(self, candidate_id: int, tf: str) -> None:
        self.record_stage(candidate_id, tf, "Executed", passed=True)

    # ---------------------------------------------------------------------
    # Integration with the existing notebook "traceability layer" objects.
    # If a PipelineProfiler / CandidateTrade scaffolding is already present
    # in the notebook (as in the current pipeline_verification_bundle
    # notebooks), hydrate our ledger from it at finalize() time so the two
    # systems produce a single consistent audit trail.
    # ---------------------------------------------------------------------
    def import_from_profiler(self, profiler) -> Dict[str, int]:
        """Hydrate the ledger from an existing PipelineProfiler-style object.

        Expected shape (matches the traceability layer already present in
        GoldRegimeX_Explorer_fixed.ipynb / Strategy_Tester_fixed.ipynb):
            profiler.records: list of dicts with keys
                "timeframe", "stage", "candidate_ids" (set|list), "metadata"
        Stage names are mapped case-insensitively; unknown stages are ignored.
        """
        records = getattr(profiler, "records", None)
        if not records:
            return {"records_seen": 0, "candidates_imported": 0}

        stage_alias = {
            "raw_bars":            None,
            "feature_engineering": "FeatureEngineering",
            "trend_pullback":      "Generated",
            "session":             "Session",
            "tbm":                 "TBM",
            "label_filter":        "TBM",
            "hmm":                 "HMM",
            "xgboost":             "HMM",
            "probability_filter":  "Probability",
            "backtest":            "Risk",
            "risk_manager":        "Risk",
            "executed":            "Executed",
        }

        seen: Dict[Tuple[str, str], set] = {}
        stages_by_tf: Dict[str, List[str]] = {}
        strategy_of_id: Dict[Tuple[str, object], str] = {}

        for rec in records:
            tf = rec.get("timeframe") or "UNKNOWN"
            raw_stage = str(rec.get("stage", "")).strip().lower()
            stage = stage_alias.get(raw_stage)
            if stage is None:
                continue
            ids_set = set(rec.get("candidate_ids") or [])
            seen.setdefault((tf, stage), set()).update(ids_set)
            stages_by_tf.setdefault(tf, [])
            if stage not in stages_by_tf[tf]:
                stages_by_tf[tf].append(stage)
            meta = rec.get("metadata") or {}
            strat_meta = str(meta.get("strategy_name", "unknown"))
            for cid in ids_set:
                strategy_of_id.setdefault((tf, cid), strat_meta)

        imported = 0
        for tf, stages in stages_by_tf.items():
            all_ids: set = set()
            for stage in stages:
                all_ids.update(seen.get((tf, stage), set()))
            for cid in all_ids:
                self.record_generation(
                    candidate_id=cid, tf=tf,
                    timestamp=pd.Timestamp.utcnow(),
                    strategy=strategy_of_id.get((tf, cid), "unknown"),
                )
                imported += 1
                for stage_name in STAGE_ORDER[1:]:
                    if stage_name not in stages:
                        continue
                    passed = cid in seen.get((tf, stage_name), set())
                    reason = "" if passed else "Filtered at %s (from profiler)" % stage_name
                    self.record_stage(cid, tf, stage_name, passed=passed, reason=reason)
                    if not passed:
                        break
        return {"records_seen": len(records), "candidates_imported": imported}

    # ---- Convenience helpers used by notebook "hook" cells --------------
    def record_session(self, cid: int, tf: str, passed: bool, reason: str = "") -> None:
        if not passed and not reason:
            reason = "Outside configured session window"
        self.record_stage(cid, tf, "Session", passed, reason)

    def record_tbm(self, cid: int, tf: str, passed: bool, reason: str = "") -> None:
        if not passed and not reason:
            reason = "Triple-barrier label invalid"
        self.record_stage(cid, tf, "TBM", passed, reason)

    def record_hmm(self, cid: int, tf: str, passed: bool, state: int, reason: str = "") -> None:
        if not passed and not reason:
            reason = "State %d not tradeable" % int(state)
        self.record_stage(cid, tf, "HMM", passed, reason, hmm_state=int(state))

    def record_probability(
        self, cid: int, tf: str, passed: bool, probability: float, threshold: float, reason: str = ""
    ) -> None:
        if not passed and not reason:
            reason = "%.4f < %.4f" % (float(probability), float(threshold))
        self.record_stage(cid, tf, "Probability", passed, reason,
                          probability=probability, threshold=threshold)
        self.thresholds[str(tf).upper()] = float(threshold)

    def record_risk(self, cid: int, tf: str, passed: bool, reason: str = "") -> None:
        if not passed and not reason:
            reason = "Risk / spread / capital constraint"
        self.record_stage(cid, tf, "Risk", passed, reason)

    # ---- Phase 4 / 8: HMM snapshot ---------------------------------------
    def record_hmm_inference(
        self,
        tf: str,
        states: Sequence[int],
        transmat: Optional[Sequence[Sequence[float]]] = None,
    ) -> HMMSnapshot:
        snap = compute_hmm_snapshot(tf, states, transmat)
        self.hmm_snapshots[snap.timeframe] = snap
        return snap

    # ---- Phase 5: probability snapshots ---------------------------------
    def record_probability_snapshot(
        self,
        tf: str,
        stage: str,
        probabilities: Sequence[float],
    ) -> None:
        tf_u = str(tf).upper()
        arr = np.asarray(list(probabilities), dtype=float)
        self.probability_snapshots.setdefault(tf_u, {})[stage] = arr
        self.probability_summaries.setdefault(tf_u, {})[stage] = probability_summary(arr)

    # ---- Phase 7: feature drift -----------------------------------------
    def record_feature_distributions(
        self,
        tf: str,
        is_df: pd.DataFrame,
        oos_df: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> pd.DataFrame:
        df = feature_drift_report(
            is_df, oos_df, feature_cols,
            psi_warn=self.psi_warn, psi_alert=self.psi_alert,
        )
        self.feature_drift_frames[str(tf).upper()] = df
        return df

    def record_feature_drift(self, tf: str, df: pd.DataFrame) -> None:
        """Attach a pre-computed drift DataFrame directly (convenience).

        Expected columns: `feature`, `psi`, `ks`, `mean_shift`, `std_shift`, `flag`.
        Missing columns are tolerated -- the frame is written out as-is.
        """
        self.feature_drift_frames[str(tf).upper()] = df.copy()

    # ---- Phase 9: reconciliation ----------------------------------------
    def record_reconciliation(self, tf: str, stage_counts: Dict[str, int]) -> None:
        self.reconciliation_counts[str(tf).upper()] = {k: int(v) for k, v in stage_counts.items()}

    def record_model_hash(self, tf: str, model_hash: str) -> None:
        self.model_hashes[str(tf).upper()] = str(model_hash)

    # ---- Finalisation: emit every artifact ------------------------------
    def finalize(
        self,
        integrity_flags: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        out = self.output_dir

        # Phase 2 -- decision log CSV.
        decisions_path = self.decisions.to_csv(out / "candidate_decisions.csv")

        # Phase 3 -- survival matrix.
        matrix, gap_warnings = build_stage_survival_matrix(
            self.ledger, material_delta_pct=self.material_survival_gap_pct,
        )
        survival_path = out / "stage_survival.csv"
        matrix.to_csv(survival_path, index=False)

        # Phase 4 + 8 -- HMM report (attach ledger-derived pass rates first).
        for snap in self.hmm_snapshots.values():
            attach_pass_rates_by_state(snap, self.ledger)
        hmm_lines: List[str] = []
        for tf in sorted(self.hmm_snapshots):
            hmm_lines.append(format_hmm_report(self.hmm_snapshots[tf]))
            hmm_lines.append("")
        hmm_path = out / "hmm_diagnostics.txt"
        hmm_path.write_text("\n".join(hmm_lines) if hmm_lines else "(no HMM inference recorded)")

        # Phase 5 -- probability summaries CSV + optional histogram PNG.
        prob_rows: List[Dict[str, Any]] = []
        for tf, stages in self.probability_summaries.items():
            for stage, stats in stages.items():
                row = {"timeframe": tf, "stage": stage}
                row.update(stats)
                prob_rows.append(row)
        prob_path = out / "probability_summary.csv"
        pd.DataFrame(prob_rows).to_csv(prob_path, index=False)
        hist_path: Optional[Path] = None
        if self.probability_snapshots:
            hist_path = plot_probability_histograms(
                self.probability_snapshots, out / "probability_histograms.png",
            )

        # Phase 6 -- session audit CSV.
        session_df = session_audit_frame(
            self.decisions, self.ledger,
            broker_tz=self.broker_tz,
            expected_session_by_tf=self.expected_session_by_tf,
        )
        session_path = out / "session_audit.csv"
        session_df.to_csv(session_path, index=False)

        # Phase 7 -- feature drift CSV (one combined file, tf column).
        drift_rows: List[Dict[str, Any]] = []
        for tf, df in self.feature_drift_frames.items():
            for _, r in df.iterrows():
                item = {"timeframe": tf}
                item.update(r.to_dict())
                drift_rows.append(item)
        drift_path = out / "feature_drift.csv"
        pd.DataFrame(drift_rows).to_csv(drift_path, index=False)

        # Phase 9 -- reconciliation CSV (long form).
        recon_rows: List[Dict[str, Any]] = []
        for tf, counts in self.reconciliation_counts.items():
            frame = reconcile_candidates(counts)
            for _, r in frame.iterrows():
                item = {"timeframe": tf}
                item.update(r.to_dict())
                recon_rows.append(item)
        recon_path = out / "candidate_reconciliation.csv"
        pd.DataFrame(recon_rows).to_csv(recon_path, index=False)

        # Phase 10 -- top N lost trades for chosen timeframe.
        lost_report = explain_lost_trades(
            self.ledger, timeframe=self.lost_trade_tf, limit=self.lost_trade_limit,
        )
        lost_path = out / ("lost_trades_%s.txt" % self.lost_trade_tf.lower())
        lost_path.write_text(lost_report)

        # Phase 11 -- health dashboard.
        dashboard = render_health_dashboard(self.ledger, matrix, integrity_flags)
        health_path = out / "pipeline_health.txt"
        health_path.write_text(dashboard)

        # Machine-readable aggregated audit JSON.
        audit_path = self._write_audit_json(
            out / "pipeline_audit.json",
            matrix=matrix,
            integrity_flags=integrity_flags or {},
            gap_warnings=gap_warnings,
            dashboard=dashboard,
        )

        if verbose:
            print(dashboard)
            if gap_warnings:
                print("\nSurvival gap warnings:")
                for w in gap_warnings:
                    print("  - " + w)

        # ---- Phase 4/7/9/11 spec-named aliases -------------------------
        # Duplicate the CSV content under the exact filenames required by
        # the implementation prompt so both naming conventions coexist.
        decisions_spec_path = write_candidate_decisions_spec(
            self.decisions, out / "candidate_decisions.csv",
        )  # spec header (overwrites the legacy header for this file)
        feature_drift_report_path = out / "feature_drift_report.csv"
        pd.DataFrame(drift_rows).to_csv(feature_drift_report_path, index=False)
        probability_report_path = out / "probability_report.csv"
        pd.DataFrame(prob_rows).to_csv(probability_report_path, index=False)
        candidate_integrity_path = write_candidate_integrity_csv(
            self.ledger, path=out / "candidate_integrity.csv",
        )

        # ---- Phase 12/13 top-100 rejected per timeframe ----------------
        top100_paths: Dict[str, str] = {}
        tfs = self.ledger.timeframes() or ["M15", "M5"]
        for tf in sorted(set([t.upper() for t in tfs] + ["M15", "M5"])):
            p = write_top_n_rejected(
                self.ledger, tf, out / ("top100_rejected_%s.csv" % tf), limit=100,
            )
            top100_paths[tf] = str(p)

        if verbose:
            print(dashboard)
            if gap_warnings:
                print("\nSurvival gap warnings:")
                for w in gap_warnings:
                    print("  - " + w)

        return {
            "run_id": self.run_id,
            # Legacy names (kept for backwards compat)
            "decisions_csv": str(decisions_spec_path),
            "survival_csv": str(survival_path),
            "hmm_txt": str(hmm_path),
            "probability_csv": str(prob_path),
            "probability_histograms_png": str(hist_path) if hist_path else None,
            "session_audit_csv": str(session_path),
            "feature_drift_csv": str(drift_path),
            "reconciliation_csv": str(recon_path),
            "lost_trades_txt": str(lost_path),
            "pipeline_health_txt": str(health_path),
            "pipeline_audit_json": str(audit_path),
            "survival_gap_warnings": gap_warnings,
            # Spec-mandated names (Phases 4/7/9/11/12/13)
            "candidate_decisions_csv":  str(decisions_spec_path),
            "feature_drift_report_csv": str(feature_drift_report_path),
            "probability_report_csv":   str(probability_report_path),
            "candidate_integrity_csv":  str(candidate_integrity_path),
            "top100_rejected_paths":    top100_paths,
        }

    # ---- Aggregated JSON audit artifact (Phase 5 spec-shape) ------------
    def _write_audit_json(
        self,
        path: os.PathLike,
        matrix: pd.DataFrame,
        integrity_flags: Dict[str, str],
        gap_warnings: List[str],
        dashboard: str,
    ) -> Path:
        # Ensure both spec timeframes always appear in the JSON, even if
        # the ledger only saw one -- otherwise downstream consumers may
        # fail their key lookups.
        tfs = sorted(set([tf.upper() for tf in self.ledger.timeframes()] + ["M15", "M5"]))

        # Phase 5 requires exactly these keys per timeframe.
        spec_timeframes: Dict[str, Dict[str, int]] = {}
        # Extended per-tf details (kept alongside so we don't lose fidelity).
        per_tf_summary: Dict[str, Any] = {}

        for tf in tfs:
            traces = self.ledger.all_for_tf(tf)
            gen = len(traces)
            session_pass = sum(1 for t in traces if t.session_pass)
            tbm_pass = sum(1 for t in traces if t.tbm_pass)
            hmm_pass = sum(1 for t in traces if t.hmm_pass)
            threshold_pass = sum(1 for t in traces if t.threshold_pass)
            execd = sum(1 for t in traces if t.executed)

            spec_timeframes[tf] = {
                "generated":      int(gen),
                "session_pass":   int(session_pass),
                "tbm_pass":       int(tbm_pass),
                "hmm_pass":       int(hmm_pass),
                "threshold_pass": int(threshold_pass),
                "executed":       int(execd),
            }

            bottleneck_stage, bottleneck_drop = bottleneck_for_tf(matrix, tf)
            per_tf_summary[tf] = {
                "generated": gen,
                "executed": execd,
                "survival_pct": round((execd / gen * 100.0) if gen else 0.0, 4),
                "threshold": self.thresholds.get(tf),
                "model_hash": self.model_hashes.get(tf),
                "bottleneck_stage": bottleneck_stage,
                "bottleneck_drop_pct": round(bottleneck_drop, 4),
                "hmm_snapshot": self.hmm_snapshots[tf].to_dict() if tf in self.hmm_snapshots else None,
                "probability_summary": self.probability_summaries.get(tf, {}),
                "feature_drift": (
                    self.feature_drift_frames[tf].to_dict(orient="records")
                    if tf in self.feature_drift_frames else []
                ),
                "reconciliation": self.reconciliation_counts.get(tf, {}),
            }

        payload = {
            # ---- Phase 5 canonical keys ----
            "experiment_id":   self.run_id,
            "timestamp":       pd.Timestamp.utcnow().isoformat(),
            "timeframes":      spec_timeframes,
            # ---- Extended fields (kept for full audit fidelity) ----
            "run_id":          self.run_id,
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "integrity_flags": integrity_flags,
            "survival_matrix": matrix.to_dict(orient="records"),
            "survival_gap_warnings": gap_warnings,
            "per_timeframe": per_tf_summary,
            "total_candidates": len(self.ledger),
            "dashboard": dashboard,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return p


__all__ = [
    "CandidateTrace",
    "CandidateLedger",
    "DecisionLog",
    "HMMSnapshot",
    "PipelineObservability",
    "STAGE_ORDER",
    "build_stage_survival_matrix",
    "bottleneck_for_tf",
    "compute_hmm_snapshot",
    "attach_pass_rates_by_state",
    "format_hmm_report",
    "probability_summary",
    "plot_probability_histograms",
    "feature_drift_report",
    "session_audit_frame",
    "reconcile_candidates",
    "explain_lost_trades",
    "render_health_dashboard",
]


# =========================================================================
# PHASE 3 -- PipelineLogger (thin facade around DecisionLog + CandidateLedger)
# =========================================================================
class PipelineLogger:
    """Unified event logger.

    Every pipeline stage calls `logger.log(candidate_id, timeframe, stage,
    decision, details=None)`. Under the hood this both:
      * appends a row to the DecisionLog (rendered as candidate_decisions.csv),
      * updates the corresponding CandidateTrace flag in the CandidateLedger.

    `decision` is a case-insensitive string; "PASS"/"ACCEPT"/"OK"/"EXECUTED"
    map to passed=True, everything else to passed=False.
    """

    _PASS_TOKENS = {"PASS", "ACCEPT", "ACCEPTED", "OK", "EXECUTED", "TRUE", "YES"}

    def __init__(self, ledger: "CandidateLedger", decisions: "DecisionLog") -> None:
        self.ledger = ledger
        self.decisions = decisions

    def log(
        self,
        candidate_id: Any,
        timeframe: str,
        stage: str,
        decision: Any,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = dict(details or {})
        decision_str = str(decision).upper()
        passed = decision_str in self._PASS_TOKENS or decision_str == "TRUE"
        reason = str(details.pop("reason", "") or "")
        if not passed and not reason:
            reason = "Rejected at %s" % stage
        strategy = str(details.pop("strategy", "") or "")
        timestamp = details.pop("timestamp", None)

        # Ensure the candidate exists in the ledger before we mark a stage.
        tf_u = str(timeframe).upper()
        try:
            cid_i = int(candidate_id)
        except Exception:
            cid_i = hash(str(candidate_id)) & 0x7fffffff
        tr = self.ledger.get(cid_i, tf_u)
        if tr is None:
            tr = self.ledger.add(CandidateTrace(
                candidate_id=cid_i,
                timeframe=tf_u,
                timestamp=pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.utcnow(),
                strategy=strategy or "unknown",
            ))

        # Update stage-specific fields from `details` before recording.
        if "session_name" in details:
            tr.session_name = str(details["session_name"])
        if "hmm_state" in details:
            try: tr.hmm_state = int(details["hmm_state"])
            except Exception: pass
        if "hmm_probability" in details:
            try: tr.hmm_probability = float(details["hmm_probability"])
            except Exception: pass
        if "xgb_probability" in details:
            try: tr.xgb_probability = float(details["xgb_probability"])
            except Exception: pass
        if "probability" in details and tr.xgb_probability is None:
            try: tr.xgb_probability = float(details["probability"])
            except Exception: pass
        if "threshold" in details:
            try: tr.threshold = float(details["threshold"])
            except Exception: pass

        # Record in the ledger's stage tracker (handles rejection_stage/reason).
        try:
            self.ledger.record_stage(candidate_id, tf_u, stage, passed=passed, reason=reason)
        except ValueError:
            # record_stage requires a non-empty reason for failures.
            self.ledger.record_stage(candidate_id, tf_u, stage, passed=False,
                                    reason=reason or ("Rejected at %s" % stage))

        # Append to the decision log.
        self.decisions.append(
            candidate_id=int(candidate_id) if isinstance(candidate_id, (int, np.integer)) else
                (hash(str(candidate_id)) & 0x7fffffff),
            timeframe=tf_u,
            stage=stage,
            decision="PASS" if passed else "REJECT",
            reason=reason if not passed else "",
            timestamp=pd.Timestamp(timestamp) if timestamp is not None else
                     pd.Timestamp(tr.timestamp) if tr.timestamp is not None else None,
            strategy=strategy or tr.strategy,
        )

        # Append to the trace's own history for full lifecycle replay.
        tr.history.append({
            "stage": stage,
            "decision": "PASS" if passed else "REJECT",
            "reason": reason,
            "details": details,
        })


# =========================================================================
# PHASE 14 -- Model Consistency (Training/Evaluation/Export UUID)
# =========================================================================
import uuid as _uuid


class ModelUUIDTracker:
    """Records the UUID assigned at each model checkpoint and verifies equality.

    Usage:
        tracker = ModelUUIDTracker()
        model_uuid = tracker.mint_training_uuid("M15")
        # ... after evaluation ...
        tracker.record_evaluation_uuid("M15", model_uuid)
        # ... after export ...
        tracker.record_export_uuid("M15", model_uuid)
        report = tracker.verify_all()  # -> {"M15": {"status": "PASS", ...}, ...}
    """

    def __init__(self) -> None:
        self._by_tf: Dict[str, Dict[str, Optional[str]]] = {}

    def _entry(self, tf: str) -> Dict[str, Optional[str]]:
        return self._by_tf.setdefault(str(tf).upper(), {
            "training": None, "evaluation": None, "export": None,
        })

    def mint_training_uuid(self, tf: str) -> str:
        u = str(_uuid.uuid4())
        self._entry(tf)["training"] = u
        return u

    def record_training_uuid(self, tf: str, u: str) -> None:
        self._entry(tf)["training"] = str(u)

    def record_evaluation_uuid(self, tf: str, u: str) -> None:
        self._entry(tf)["evaluation"] = str(u)

    def record_export_uuid(self, tf: str, u: str) -> None:
        self._entry(tf)["export"] = str(u)

    def verify(self, tf: str) -> Dict[str, Any]:
        e = self._by_tf.get(str(tf).upper())
        if not e:
            return {"status": "MISSING", "tf": tf, "uuids": {}}
        seen = [v for v in e.values() if v is not None]
        all_present = all(v is not None for v in e.values())
        all_equal = all_present and len(set(seen)) == 1
        return {
            "status": "PASS" if all_equal else ("PARTIAL" if seen else "MISSING"),
            "tf": tf,
            "uuids": dict(e),
            "all_present": all_present,
            "all_equal": all_equal,
        }

    def verify_all(self) -> Dict[str, Dict[str, Any]]:
        return {tf: self.verify(tf) for tf in self._by_tf}


# =========================================================================
# PHASE 15 -- Cross-Notebook Manifest
# =========================================================================
import hashlib as _hashlib


class PipelineManifest:
    """Read/write the shared manifest used to guarantee Strategy Tester and
    Explorer are looking at the same pipeline artifacts.

    Strategy Tester writes the manifest at the end of its run.
    Explorer validates the manifest at the beginning of its run and aborts
    if any hash does not match its own recomputed value.
    """

    KEYS: Tuple[str, ...] = (
        "feature_hash", "session_filter_hash", "strategy_hash",
        "candidate_hash", "model_hash", "pipeline_version",
    )

    def __init__(self, path: os.PathLike = "reports/pipeline_manifest.json") -> None:
        self.path = Path(path)
        self.data: Dict[str, Any] = {}

    # ---- Hash helpers ---------------------------------------------------
    @staticmethod
    def hash_object(obj: Any) -> str:
        """Deterministic hash of a JSON-serialisable object.

        Falls back to str(obj) for non-serialisable values so we don't
        raise inside a manifest write.
        """
        try:
            payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        except Exception:
            payload = repr(obj).encode("utf-8")
        return _hashlib.sha256(payload).hexdigest()

    @staticmethod
    def hash_dataframe(df: "pd.DataFrame") -> str:
        try:
            # Use pandas' own hash to be robust to numeric noise.
            from pandas.util import hash_pandas_object
            h = hash_pandas_object(df, index=True).values.tobytes()
        except Exception:
            h = df.to_csv(index=True).encode("utf-8")
        return _hashlib.sha256(h).hexdigest()

    @staticmethod
    def hash_file(path: os.PathLike) -> str:
        p = Path(path)
        if not p.exists():
            return ""
        h = _hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # ---- Write ----------------------------------------------------------
    def write(
        self,
        *,
        feature_hash: str,
        session_filter_hash: str,
        strategy_hash: str,
        candidate_hash: str,
        model_hash: str,
        pipeline_version: str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Path:
        self.data = {
            "feature_hash":        str(feature_hash),
            "session_filter_hash": str(session_filter_hash),
            "strategy_hash":       str(strategy_hash),
            "candidate_hash":      str(candidate_hash),
            "model_hash":          str(model_hash),
            "pipeline_version":    str(pipeline_version),
            "written_at":          pd.Timestamp.utcnow().isoformat(),
        }
        if extras:
            self.data["extras"] = extras
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)
        return self.path

    # ---- Read ----------------------------------------------------------
    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError("Manifest not found at %s" % self.path)
        with open(self.path, "r", encoding="utf-8") as fh:
            self.data = json.load(fh)
        return self.data

    # ---- Validate -------------------------------------------------------
    def validate(self, expected: Dict[str, str], *, strict: bool = True) -> Dict[str, Any]:
        """Compare loaded manifest against locally-recomputed hashes.

        `expected` is a dict with any subset of self.KEYS. Only the keys
        provided are checked. Returns {"status": "PASS"|"FAIL", "mismatches": [...]}.
        Raises RuntimeError when strict=True and any hash mismatches.
        """
        if not self.data:
            self.load()
        mismatches = []
        for k, exp in expected.items():
            got = self.data.get(k, None)
            if got != exp:
                mismatches.append({"key": k, "manifest": got, "local": exp})
        status = "PASS" if not mismatches else "FAIL"
        report = {"status": status, "mismatches": mismatches, "checked": list(expected.keys())}
        if strict and mismatches:
            raise RuntimeError("Pipeline manifest hash mismatch: %s" % mismatches)
        return report


# =========================================================================
# PHASES 12 & 13 -- Top 100 rejected trades (per timeframe)
# =========================================================================
def _decision_word(passed: Optional[bool]) -> str:
    if passed is None:
        return "N/A"
    return "PASS" if bool(passed) else "REJECT"


def top_n_rejected_frame(ledger: "CandidateLedger", timeframe: str, limit: int = 100) -> "pd.DataFrame":
    """Build the Phase 12/13 top-N rejected schema for the given timeframe."""
    tf_u = str(timeframe).upper()
    rows: List[Dict[str, Any]] = []
    for tr in ledger.list_by_tf(tf_u):
        if tr.executed:
            continue
        rows.append({
            "Candidate ID":         tr.candidate_id,
            "Timestamp":            pd.Timestamp(tr.timestamp).isoformat() if tr.timestamp is not None else "",
            "Strategy":             tr.strategy,
            "Session Decision":     _decision_word(tr.session_pass),
            "TBM Decision":         _decision_word(tr.tbm_pass),
            "HMM State":            tr.hmm_state if tr.hmm_state is not None else "",
            "HMM Decision":         _decision_word(tr.hmm_pass),
            "XGBoost Probability":  tr.xgb_probability if tr.xgb_probability is not None else "",
            "Threshold":            tr.threshold if tr.threshold is not None else "",
            "Risk Decision":        _decision_word(tr.risk_pass),
            "Final Rejection Stage": tr.rejection_stage or "",
            "Rejection Reason":     tr.rejection_reason or "",
        })
    if not rows:
        # Return empty frame with the mandated columns so downstream
        # consumers never see a missing file.
        return pd.DataFrame(columns=[
            "Candidate ID", "Timestamp", "Strategy", "Session Decision",
            "TBM Decision", "HMM State", "HMM Decision", "XGBoost Probability",
            "Threshold", "Risk Decision", "Final Rejection Stage", "Rejection Reason",
        ])
    df = pd.DataFrame(rows)
    # Preserve original rejection order (first-rejected first).
    return df.head(int(limit)).reset_index(drop=True)


def write_top_n_rejected(ledger: "CandidateLedger", timeframe: str, path: os.PathLike, limit: int = 100) -> Path:
    df = top_n_rejected_frame(ledger, timeframe, limit=limit)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


# =========================================================================
# Phase 4 -- CSV export with the spec column names
# =========================================================================
def write_candidate_decisions_spec(decisions: "DecisionLog", path: os.PathLike) -> Path:
    """Write candidate_decisions.csv with the exact Phase 4 header:
         Candidate, TF, Timestamp, Stage, Decision, Reason
    Every REJECT row is guaranteed to have a non-blank Reason.
    """
    df = decisions.as_frame()
    if df.empty:
        out = pd.DataFrame(columns=["Candidate", "TF", "Timestamp", "Stage", "Decision", "Reason"])
    else:
        out = pd.DataFrame({
            "Candidate": df["candidate_id"],
            "TF":        df["timeframe"],
            "Timestamp": df["timestamp"],
            "Stage":     df["stage"],
            "Decision":  df["decision"],
            "Reason":    df["reason"].fillna("").astype(str),
        })
        # Enforce Phase 4 invariant: rejection rows must have a reason.
        mask = (out["Decision"].str.upper() == "REJECT") & (out["Reason"].str.len() == 0)
        if mask.any():
            out.loc[mask, "Reason"] = "Rejected at " + out.loc[mask, "Stage"].astype(str)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return p


# =========================================================================
# Phase 11 -- candidate_integrity.csv with the spec columns
# =========================================================================
def write_candidate_integrity_csv(
    ledger: "CandidateLedger",
    tester_ids: Optional[Sequence[Any]] = None,
    explorer_ids: Optional[Sequence[Any]] = None,
    path: os.PathLike = "reports/observability/candidate_integrity.csv",
) -> Path:
    """Emit the Phase 11 candidate integrity CSV.

    Columns: Candidate ID | Strategy Tester Status | Explorer Status | Final Status

    If `tester_ids` / `explorer_ids` are None, the ledger is treated as the
    source of truth for both (i.e. single-notebook self-check).
    """
    ledger_ids = {tr.candidate_id for tr in ledger.list_all()}
    t_set = set(tester_ids) if tester_ids is not None else ledger_ids
    e_set = set(explorer_ids) if explorer_ids is not None else ledger_ids
    all_ids = sorted(ledger_ids | t_set | e_set, key=lambda x: str(x))
    rows = []
    for cid in all_ids:
        in_t = cid in t_set
        in_e = cid in e_set
        tr = None
        try:
            # Attempt to find any trace with this id, across timeframes.
            for _tr in ledger.list_all():
                if _tr.candidate_id == cid:
                    tr = _tr; break
        except Exception:
            pass
        if tr is not None and tr.executed:
            final = "EXECUTED"
        elif tr is not None and tr.rejection_stage:
            final = "REJECTED @ %s" % tr.rejection_stage
        elif not in_t or not in_e:
            final = "MISSING (%s%s)" % (
                "tester" if not in_t else "", "explorer" if not in_e else "",
            )
        else:
            final = "UNKNOWN"
        rows.append({
            "Candidate ID":            cid,
            "Strategy Tester Status":  "PRESENT" if in_t else "MISSING",
            "Explorer Status":         "PRESENT" if in_e else "MISSING",
            "Final Status":            final,
        })
    df = pd.DataFrame(rows, columns=["Candidate ID", "Strategy Tester Status",
                                     "Explorer Status", "Final Status"])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


# =========================================================================
# Update __all__
# =========================================================================
try:
    __all__.extend([
        "PipelineLogger",
        "ModelUUIDTracker",
        "PipelineManifest",
        "top_n_rejected_frame",
        "write_top_n_rejected",
        "write_candidate_decisions_spec",
        "write_candidate_integrity_csv",
        "STAGE_DISPLAY_NAMES",
    ])
except NameError:
    pass
