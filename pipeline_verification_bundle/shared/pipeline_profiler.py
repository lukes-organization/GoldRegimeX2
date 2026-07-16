"""shared/pipeline_profiler.py  -- Observability Reconstruction V2 (Phases 1-5, 10-12).

Single, shared, event-based pipeline profiler used by BOTH notebooks
(Strategy Tester + Explorer), the Validator, Backtester and Live Trader.

Design contract (do NOT reintroduce the old behaviour):
  * Every pipeline stage emits ONE event per candidate at the decision site
    (PASS or REJECT), via `record(...)`.
  * There are NO stage aliases, NO renaming, NO mapping tables, NO post-hoc
    hydration, and NO set-union count preservation.
  * Stage survival counts are ALWAYS computed by filtering the append-only
    ledger -- never reconstructed afterwards.
  * Survival must be monotonically non-increasing across PIPELINE_STAGES;
    a violation raises PipelineIntegrityError.
  * Diagnostics are dataset-scoped: create one profiler with dataset="OOS"
    and (optionally) one with dataset="IS". Only the OOS profiler produces
    the canonical lifecycle reports.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical stage vocabulary (Phase 1). No aliases anywhere in the codebase.
# ---------------------------------------------------------------------------
PIPELINE_STAGES: List[str] = [
    "GENERATED",
    "FEATURE_ENGINEERING",
    "SESSION",
    "TBM",
    "HMM",
    "PROBABILITY",
    "RISK",
    "EXECUTED",
]
_STAGE_INDEX = {s: i for i, s in enumerate(PIPELINE_STAGES)}

VALID_DECISIONS = ("PASS", "REJECT")
VALID_DATASETS = ("OOS", "IS")


class PipelineIntegrityError(RuntimeError):
    """Raised when the recorded ledger violates a hard observability invariant."""


@dataclass
class PipelineEvent:
    candidate_id: str
    timeframe: str
    stage: str
    decision: str
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in _STAGE_INDEX:
            raise PipelineIntegrityError(
                "Unknown stage %r; allowed: %s" % (self.stage, PIPELINE_STAGES)
            )
        if self.decision not in VALID_DECISIONS:
            raise PipelineIntegrityError(
                "Unknown decision %r; allowed: %s" % (self.decision, VALID_DECISIONS)
            )

    def as_row(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "timeframe": self.timeframe,
            "stage": self.stage,
            "decision": self.decision,
            "reason": self.reason,
            "metadata": json.dumps(self.metadata, sort_keys=True, default=str),
        }


def get_git_sha(default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip() or default
    except Exception:
        return default


def build_run_meta(dataset: str, timeframe: str, run_uuid: Optional[str] = None) -> Dict[str, str]:
    if dataset not in VALID_DATASETS:
        raise PipelineIntegrityError("dataset must be one of %s" % (VALID_DATASETS,))
    return {
        "run_uuid": run_uuid or str(uuid.uuid4()),
        "git_sha": get_git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "timeframe": str(timeframe).upper(),
    }


class PipelineProfiler:
    """Append-only, event-based profiler. One instance per dataset scope."""

    def __init__(self, dataset: str, run_uuid: Optional[str] = None):
        if dataset not in VALID_DATASETS:
            raise PipelineIntegrityError("dataset must be one of %s" % (VALID_DATASETS,))
        self.dataset = dataset
        self.run_uuid = run_uuid or str(uuid.uuid4())
        self._events: List[PipelineEvent] = []          # the ledger (Phase 2)
        self._pre_candidate: List[Dict[str, Any]] = []  # Phase 11

    # -- Phase 4: record at the decision site ------------------------------
    def record(
        self,
        candidate_id: str,
        timeframe: str,
        stage: str,
        decision: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._events.append(
            PipelineEvent(
                candidate_id=str(candidate_id),
                timeframe=str(timeframe).upper(),
                stage=stage,
                decision=decision,
                reason=reason,
                metadata=metadata or {},
            )
        )

    def record_pass(self, candidate_id, timeframe, stage, metadata=None) -> None:
        self.record(candidate_id, timeframe, stage, "PASS", None, metadata)

    def record_reject(self, candidate_id, timeframe, stage, reason, metadata=None) -> None:
        self.record(candidate_id, timeframe, stage, "REJECT", reason, metadata)

    # -- Phase 11: pre-candidate (row-level) losses ------------------------
    def record_pre_candidate(self, reason: str, count: int, timeframe: str, stage: str = "FEATURE_ENGINEERING") -> None:
        self._pre_candidate.append(
            {
                "timeframe": str(timeframe).upper(),
                "stage": stage,
                "reason": reason,
                "count": int(count),
            }
        )

    # -- Ledger access -----------------------------------------------------
    def ledger_frame(self) -> pd.DataFrame:
        if not self._events:
            return pd.DataFrame(columns=["candidate_id", "timeframe", "stage", "decision", "reason", "metadata"])
        return pd.DataFrame([e.as_row() for e in self._events])

    def pre_candidate_frame(self) -> pd.DataFrame:
        if not self._pre_candidate:
            return pd.DataFrame(columns=["timeframe", "stage", "reason", "count"])
        return pd.DataFrame(self._pre_candidate)

    # -- Phase 3: real stage survival, computed from the ledger ------------
    def _passed_ids(self, timeframe: str, stage: str) -> set:
        tf = str(timeframe).upper()
        return {
            e.candidate_id
            for e in self._events
            if e.timeframe == tf and e.stage == stage and e.decision == "PASS"
        }

    def _reached_ids(self, timeframe: str, stage: str) -> set:
        tf = str(timeframe).upper()
        return {
            e.candidate_id
            for e in self._events
            if e.timeframe == tf and e.stage == stage
        }

    def stage_counts(self, timeframe: str) -> Dict[str, int]:
        return {stage: len(self._passed_ids(timeframe, stage)) for stage in PIPELINE_STAGES}

    def assert_monotonic(self, timeframe: str) -> None:
        counts = self.stage_counts(timeframe)
        for i in range(len(PIPELINE_STAGES) - 1):
            a, b = PIPELINE_STAGES[i], PIPELINE_STAGES[i + 1]
            if counts[b] > counts[a]:
                raise PipelineIntegrityError(
                    "[%s/%s] survival not monotonic: %s=%d < %s=%d"
                    % (self.dataset, timeframe, a, counts[a], b, counts[b])
                )
        # A candidate that PASSED stage i but has NO event at stage i+1 was
        # lost without logging -- forbidden by the decision-site contract.
        for i in range(len(PIPELINE_STAGES) - 1):
            a, b = PIPELINE_STAGES[i], PIPELINE_STAGES[i + 1]
            passed_a = self._passed_ids(timeframe, a)
            reached_b = self._reached_ids(timeframe, b)
            silent = passed_a - reached_b
            if silent:
                raise PipelineIntegrityError(
                    "[%s/%s] %d candidate(s) passed %s but emitted no event at %s "
                    "(candidate lost without logging): e.g. %s"
                    % (self.dataset, timeframe, len(silent), a, b, sorted(silent)[:3])
                )

    def survival_frame(self, timeframe: str) -> pd.DataFrame:
        counts = self.stage_counts(timeframe)
        base = counts[PIPELINE_STAGES[0]] or 1
        rows = []
        prev = None
        for stage in PIPELINE_STAGES:
            c = counts[stage]
            rows.append(
                {
                    "dataset": self.dataset,
                    "timeframe": str(timeframe).upper(),
                    "stage": stage,
                    "count": c,
                    "pct_of_generated": round(100.0 * c / base, 2),
                    "lost_from_prev": 0 if prev is None else (prev - c),
                }
            )
            prev = c
        return pd.DataFrame(rows)

    def lost_trades_frame(self, timeframe: str) -> pd.DataFrame:
        tf = str(timeframe).upper()
        rows = []
        for e in self._events:
            if e.timeframe == tf and e.decision == "REJECT":
                rows.append(
                    {
                        "dataset": self.dataset,
                        "timeframe": tf,
                        "candidate_id": e.candidate_id,
                        "rejected_at_stage": e.stage,
                        "reason": e.reason,
                    }
                )
        cols = ["dataset", "timeframe", "candidate_id", "rejected_at_stage", "reason"]
        return pd.DataFrame(rows, columns=cols)

    def waterfall_text(self, timeframe: str) -> str:
        sf = self.survival_frame(timeframe)
        lines = ["DATASET  %s" % self.dataset, "TIMEFRAME  %s" % str(timeframe).upper(), ""]
        for _, r in sf.iterrows():
            bar = "#" * int(r["pct_of_generated"] / 2.0)
            lines.append(
                "%-20s %8d  %6.2f%%  -%d  %s"
                % (r["stage"], r["count"], r["pct_of_generated"], r["lost_from_prev"], bar)
            )
        return "\n".join(lines)

    def candidate_ids_unique_per_stage(self, timeframe: str) -> bool:
        """Exactly one event per (candidate, stage) -- no double logging."""
        tf = str(timeframe).upper()
        seen = set()
        for e in self._events:
            if e.timeframe != tf:
                continue
            key = (e.candidate_id, e.stage)
            if key in seen:
                return False
            seen.add(key)
        return True


# ---------------------------------------------------------------------------
# Phase 10: report regeneration with mandatory run metadata.
# ---------------------------------------------------------------------------
def purge_reports_dir(outdir: str) -> List[str]:
    """Delete everything under outdir except .gitkeep. Returns removed names."""
    os.makedirs(outdir, exist_ok=True)
    removed = []
    for name in os.listdir(outdir):
        if name == ".gitkeep":
            continue
        path = os.path.join(outdir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(name)
        elif os.path.isdir(path):
            for root, _dirs, files in os.walk(path, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                os.rmdir(root)
            removed.append(name + "/")
    gitkeep = os.path.join(outdir, ".gitkeep")
    if not os.path.exists(gitkeep):
        open(gitkeep, "w").close()
    return removed


def _meta_header(meta: Dict[str, str]) -> str:
    return "# " + " | ".join("%s=%s" % (k, meta[k]) for k in ("run_uuid", "git_sha", "timestamp", "dataset", "timeframe"))


def _write_csv_with_meta(df: pd.DataFrame, path: str, meta: Dict[str, str]) -> None:
    with open(path, "w", newline="") as fh:
        fh.write(_meta_header(meta) + "\n")
        df.to_csv(fh, index=False)


def write_oos_reports(profiler: PipelineProfiler, timeframes: List[str], outdir: str,
                      purge: bool = True) -> Dict[str, str]:
    """Write the canonical lifecycle reports. OOS profiler ONLY (Phase 5)."""
    if profiler.dataset != "OOS":
        raise PipelineIntegrityError(
            "Canonical lifecycle reports must come from the OOS profiler, got dataset=%s"
            % profiler.dataset
        )
    if purge:
        purge_reports_dir(outdir)

    written: Dict[str, str] = {}
    survival_all, lost_all, health_lines, waterfalls = [], [], [], []
    for tf in timeframes:
        profiler.assert_monotonic(tf)  # Phase 3 hard gate
        survival_all.append(profiler.survival_frame(tf))
        lost_all.append(profiler.lost_trades_frame(tf))
        waterfalls.append(profiler.waterfall_text(tf))
        counts = profiler.stage_counts(tf)
        gen, execd = counts["GENERATED"], counts["EXECUTED"]
        surv = round(100.0 * execd / gen, 2) if gen else 0.0
        health_lines += [
            "%s Generated: %d" % (tf, gen),
            "%s Executed:  %d" % (tf, execd),
            "%s Survival:  %.2f%%" % (tf, surv),
            "",
        ]

    meta = build_run_meta("OOS", "+".join(str(t).upper() for t in timeframes), run_uuid=profiler.run_uuid)

    p = os.path.join(outdir, "survival_analysis.csv")
    _write_csv_with_meta(pd.concat(survival_all, ignore_index=True), p, meta); written["survival_analysis.csv"] = p
    p = os.path.join(outdir, "lost_trades.csv")
    _write_csv_with_meta(pd.concat(lost_all, ignore_index=True), p, meta); written["lost_trades.csv"] = p
    p = os.path.join(outdir, "PreCandidateLosses.csv")
    _write_csv_with_meta(profiler.pre_candidate_frame(), p, meta); written["PreCandidateLosses.csv"] = p
    p = os.path.join(outdir, "waterfall.txt")
    with open(p, "w") as fh:
        fh.write(_meta_header(meta) + "\n\n" + "\n\n".join(waterfalls) + "\n")
    written["waterfall.txt"] = p
    p = os.path.join(outdir, "pipeline_health.txt")
    with open(p, "w") as fh:
        fh.write(_meta_header(meta) + "\n\n" + "====== PIPELINE HEALTH ======\n\n" + "\n".join(health_lines))
    written["pipeline_health.txt"] = p
    return written


def write_is_debug_report(profiler: PipelineProfiler, timeframes: List[str], outdir: str) -> str:
    """IS profiler exported separately for debugging only (Phase 5)."""
    if profiler.dataset != "IS":
        raise PipelineIntegrityError("write_is_debug_report requires dataset=IS")
    os.makedirs(outdir, exist_ok=True)
    meta = build_run_meta("IS", "+".join(str(t).upper() for t in timeframes), run_uuid=profiler.run_uuid)
    frames = [profiler.survival_frame(tf) for tf in timeframes]
    path = os.path.join(outdir, "survival_analysis_IS_debug.csv")
    _write_csv_with_meta(pd.concat(frames, ignore_index=True), path, meta)
    return path


# ---------------------------------------------------------------------------
# Shared candidate identity + record types (Phase 12).
# Previously duplicated as `class CandidateTrade` / `make_candidate_id` in BOTH
# notebooks; centralised here so IDs generated independently always match.
# ---------------------------------------------------------------------------
import hashlib as _hashlib
import json as _json
from enum import Enum


@dataclass
class CandidateTrade:
    candidate_id: str
    timeframe: str
    timestamp: Any
    direction: str  # "BUY" or "SELL"
    entry_price: float
    stop_price: float
    target_price: float
    strategy_name: str
    parameter_set_id: str
    hmm_state: Optional[int] = None
    xgb_probability: Optional[float] = None
    accepted: bool = False
    rejection_reason: Optional[str] = None


def make_candidate_id(timeframe: str, timestamp, direction: str) -> str:
    """Deterministic candidate id: sha256('{tf}_{ts}_{direction}')[:16].

    Identical formula for Strategy Tester and Explorer so independently
    generated ids match for the same (timeframe, timestamp, direction).
    """
    key = "%s_%s_%s" % (timeframe, timestamp, direction)
    return _hashlib.sha256(key.encode()).hexdigest()[:16]


def parameter_set_id(params: dict) -> str:
    blob = _json.dumps(params, sort_keys=True, default=str)
    return _hashlib.sha256(blob.encode()).hexdigest()[:12]


class RejectionReason(str, Enum):
    LOW_PROBABILITY = "LOW_PROBABILITY"
    SESSION = "SESSION"
    ATR = "ATR"
    TBM = "TBM"
    DUPLICATE = "DUPLICATE"
    MAX_EXPOSURE = "MAX_EXPOSURE"
    INVALID_FEATURE = "INVALID_FEATURE"
    NO_TRADE_REGIME = "NO_TRADE_REGIME"
    UNKNOWN = "UNKNOWN"
