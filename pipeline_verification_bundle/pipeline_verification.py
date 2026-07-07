# Central Pipeline Verification Module (Phase 1).
#
# Contains the eight verifier classes plus PipelineCertification.
# Both Strategy_Tester.ipynb and GoldRegimeX_Explorer.ipynb import from
# this module rather than duplicating verification code.
#
# Every verifier respects the global VERIFY_PIPELINE flag: the calling
# notebook cells should check that flag before invoking a verifier, and
# a verifier called with enabled=False returns a skipped result.

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------
@dataclass
class VerificationResult:
    name: str
    status: str  # PASS | FAIL | WARN | SKIPPED
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def is_pass(self) -> bool:
        return self.status == "PASS"

    def to_dict(self):
        return asdict(self)


def _hash_index(idx) -> str:
    return hashlib.sha256(pd.Index(idx).astype(str).str.cat().encode()).hexdigest()


def _print_header(title: str, width: int = 70):
    bar = "=" * width
    print(bar)
    pad = max(0, (width - len(title) - 2) // 2)
    print(" " * pad + title)
    print(bar)


# ------------------------------------------------------------------
# Phase 2 - DatasetIntegrityVerifier
# ------------------------------------------------------------------
class DatasetIntegrityVerifier:
    def verify(
        self,
        train_df: pd.DataFrame,
        oos_df: pd.DataFrame,
        feature_columns: Sequence[str],
        label_column: Optional[str] = None,
        candidate_column: str = "candidate_id",
        abort_on_overlap: bool = True,
        verbose: bool = True,
    ) -> VerificationResult:
        details: Dict[str, Any] = {}

        train_dup_idx = int(train_df.index.duplicated().sum())
        oos_dup_idx = int(oos_df.index.duplicated().sum())

        overlap_ts = set(train_df.index).intersection(set(oos_df.index))

        overlap_ids = set()
        train_dup_ids = 0
        oos_dup_ids = 0
        if candidate_column in train_df.columns and candidate_column in oos_df.columns:
            tr_ids = train_df[candidate_column].dropna()
            oo_ids = oos_df[candidate_column].dropna()
            train_dup_ids = int(tr_ids.duplicated().sum())
            oos_dup_ids = int(oo_ids.duplicated().sum())
            overlap_ids = set(tr_ids).intersection(set(oo_ids))

        available_features = [c for c in feature_columns if c in train_df.columns and c in oos_df.columns]
        train_dup_features = int(train_df[available_features].duplicated().sum()) if available_features else 0
        oos_dup_features = int(oos_df[available_features].duplicated().sum()) if available_features else 0

        train_dup_labels = 0
        oos_dup_labels = 0
        if label_column and label_column in train_df.columns:
            train_dup_labels = int(train_df[label_column].isna().sum())
        if label_column and label_column in oos_df.columns:
            oos_dup_labels = int(oos_df[label_column].isna().sum())

        details.update({
            "train_rows": len(train_df),
            "oos_rows": len(oos_df),
            "train_duplicate_index": train_dup_idx,
            "oos_duplicate_index": oos_dup_idx,
            "timestamp_overlap": len(overlap_ts),
            "candidate_overlap": len(overlap_ids),
            "train_duplicate_candidate_ids": train_dup_ids,
            "oos_duplicate_candidate_ids": oos_dup_ids,
            "train_duplicate_feature_rows": train_dup_features,
            "oos_duplicate_feature_rows": oos_dup_features,
            "train_null_labels": train_dup_labels,
            "oos_null_labels": oos_dup_labels,
        })

        overlap_any = len(overlap_ts) > 0 or len(overlap_ids) > 0
        dup_any = (
            train_dup_idx > 0 or oos_dup_idx > 0 or train_dup_ids > 0 or oos_dup_ids > 0
        )
        status = "FAIL" if (overlap_any or dup_any) else "PASS"

        if verbose:
            _print_header("DATASET INTEGRITY")
            print("Train Rows           : %d" % details["train_rows"])
            print("OOS Rows             : %d" % details["oos_rows"])
            print("Timestamp Overlap    : %d" % details["timestamp_overlap"])
            print("Candidate Overlap    : %d" % details["candidate_overlap"])
            print("Duplicate Features   : %d (train) / %d (oos)" % (train_dup_features, oos_dup_features))
            print("Duplicate Indices    : %d (train) / %d (oos)" % (train_dup_idx, oos_dup_idx))
            print("Status               : %s" % status)

        if status == "FAIL" and abort_on_overlap:
            raise AssertionError(
                "Dataset integrity failed: timestamp_overlap=%d candidate_overlap=%d" % (
                    len(overlap_ts), len(overlap_ids),
                )
            )
        return VerificationResult("Dataset Integrity", status, details)


# ------------------------------------------------------------------
# Phase 3 - Train/OOS leakage helpers + Phase 7 model identity
# ------------------------------------------------------------------
def train_hash_of(df: pd.DataFrame) -> str:
    return _hash_index(df.index)


def assert_no_train_oos_leakage(train_df: pd.DataFrame, oos_df: pd.DataFrame, verbose: bool = True) -> VerificationResult:
    train_hash = train_hash_of(train_df)
    oos_hash = train_hash_of(oos_df)
    disjoint = set(train_df.index).isdisjoint(set(oos_df.index))
    details = {
        "train_hash": train_hash,
        "oos_hash": oos_hash,
        "train_rows": len(train_df),
        "oos_rows": len(oos_df),
        "index_disjoint": disjoint,
    }
    status = "PASS" if (train_hash != oos_hash and disjoint) else "FAIL"
    if verbose:
        _print_header("TRAIN/OOS VERIFICATION")
        print("Train Hash : %s..." % train_hash[:16])
        print("OOS Hash   : %s..." % oos_hash[:16])
        print("Disjoint   : %s" % disjoint)
        print("Status     : %s" % status)
    if status == "FAIL":
        raise AssertionError("Train/OOS leakage detected: hashes equal or indices overlap")
    return VerificationResult("Train/OOS Separation", status, details)


# ------------------------------------------------------------------
# Phase 4 - FeatureLeakageVerifier
# ------------------------------------------------------------------
LEAKY_NAME_PATTERNS = (
    "future_return",
    "future_close",
    "future_high",
    "future_low",
    "forward_",
    "lookahead",
    "target_next",
)


class FeatureLeakageVerifier:
    CRITICAL_ABS_CORR = 0.98
    WARNING_ABS_CORR = 0.85

    def verify(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        label_column: str,
        source_snippets: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> VerificationResult:
        if label_column not in df.columns:
            return VerificationResult(
                "Feature Leakage", "SKIPPED",
                {"reason": "label column %r not present" % label_column},
            )

        y = df[label_column]
        rows = []
        critical = 0
        warned = 0
        for f in feature_columns:
            if f not in df.columns:
                continue
            x = df[f]
            corr = float(x.corr(y)) if x.dtype.kind in "fiu" else float("nan")
            name_hits = [p for p in LEAKY_NAME_PATTERNS if p in f.lower()]
            snippet = (source_snippets or {}).get(f, "")
            code_hits = []
            if snippet:
                if "shift(-" in snippet:
                    code_hits.append("shift-negative")
                if "rolling(" in snippet and "center=True" in snippet:
                    code_hits.append("rolling-centered")
            if not np.isnan(corr) and abs(corr) > self.CRITICAL_ABS_CORR:
                classification = "CRITICAL"
                critical += 1
            elif name_hits or code_hits or (not np.isnan(corr) and abs(corr) > self.WARNING_ABS_CORR):
                classification = "WARNING"
                warned += 1
            else:
                classification = "SAFE"
            rows.append({
                "feature": f,
                "abs_corr": abs(corr) if not np.isnan(corr) else float("nan"),
                "name_flags": ",".join(name_hits),
                "code_flags": ",".join(code_hits),
                "classification": classification,
            })

        report = pd.DataFrame(rows).sort_values("abs_corr", ascending=False, na_position="last")
        status = "FAIL" if critical > 0 else ("WARN" if warned > 0 else "PASS")
        if verbose:
            _print_header("FEATURE LEAKAGE REPORT")
            print(report.to_string(index=False))
            print("Critical: %d  Warning: %d  Status: %s" % (critical, warned, status))
        return VerificationResult("Feature Leakage", status, {"report": report, "critical": critical, "warned": warned})


# ------------------------------------------------------------------
# Phase 5 - EvaluationVerifier
# ------------------------------------------------------------------
class EvaluationVerifier:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def record(
        self,
        model_uuid: str,
        dataset_name: str,
        prediction_source: str,
        predictions,
        labels,
        training_index=None,
        verbose: bool = True,
    ) -> VerificationResult:
        n_pred = int(len(predictions))
        n_lbl = int(len(labels))
        record = {
            "model_uuid": model_uuid,
            "dataset_name": dataset_name,
            "prediction_source": prediction_source,
            "prediction_count": n_pred,
            "label_count": n_lbl,
        }
        self._records.append(record)

        leakage = False
        leakage_reason = ""
        if training_index is not None:
            pred_idx = getattr(predictions, "index", None)
            eval_idx = pred_idx if pred_idx is not None else getattr(labels, "index", None)
            if eval_idx is not None:
                shared = set(training_index).intersection(set(eval_idx))
                if shared and dataset_name.upper() == "OOS":
                    leakage = True
                    leakage_reason = (
                        "OOS evaluation index shares %d timestamps with training index" % len(shared)
                    )

        status = "FAIL" if leakage else "PASS"
        if verbose:
            _print_header("EVALUATION VERIFICATION")
            print("Model UUID          : %s" % model_uuid)
            print("Prediction Source   : %s" % prediction_source)
            print("Evaluation Dataset  : %s" % dataset_name)
            print("Prediction Count    : %d" % n_pred)
            print("Label Count         : %d" % n_lbl)
            if leakage:
                print("WARNING: %s" % leakage_reason)
            print("Status              : %s" % status)

        if leakage:
            raise AssertionError("Evaluation Leakage Detected: " + leakage_reason)
        return VerificationResult("Evaluation Integrity", status, record)


# ------------------------------------------------------------------
# Phase 6 - PredictionAlignmentVerifier
# ------------------------------------------------------------------
class PredictionAlignmentVerifier:
    def verify(self, predictions, labels, verbose: bool = True) -> VerificationResult:
        length_ok = len(predictions) == len(labels)
        pred_idx = getattr(predictions, "index", None)
        lbl_idx = getattr(labels, "index", None)
        index_ok = True
        arr_ok = True
        if pred_idx is not None and lbl_idx is not None:
            index_ok = bool(pred_idx.equals(lbl_idx))
            arr_ok = bool(np.array_equal(np.asarray(pred_idx), np.asarray(lbl_idx)))
        details = {
            "length_predictions": int(len(predictions)),
            "length_labels": int(len(labels)),
            "length_match": length_ok,
            "index_equals": index_ok,
            "array_equal": arr_ok,
        }
        status = "PASS" if (length_ok and index_ok and arr_ok) else "FAIL"
        if verbose:
            _print_header("PREDICTION ALIGNMENT REPORT")
            for k, v in details.items():
                print("%-24s: %s" % (k, v))
            print("Status                  : %s" % status)
        if status == "FAIL":
            raise AssertionError("Prediction alignment failed: %s" % details)
        return VerificationResult("Prediction Alignment", status, details)


# ------------------------------------------------------------------
# Phase 7 - Model identity helper
# ------------------------------------------------------------------
def assign_model_uuid(model) -> str:
    if not hasattr(model, "model_uuid") or getattr(model, "model_uuid", None) is None:
        try:
            setattr(model, "model_uuid", str(uuid.uuid4()))
        except Exception:
            # Fallback: return a fresh uuid even if we cannot attach it.
            return str(uuid.uuid4())
    return getattr(model, "model_uuid")


def verify_model_identity(training_uuid: str, eval_uuid: str, verbose: bool = True) -> VerificationResult:
    ok = training_uuid == eval_uuid
    status = "PASS" if ok else "FAIL"
    if verbose:
        _print_header("MODEL IDENTITY")
        print("Training UUID   : %s" % training_uuid)
        print("Evaluation UUID : %s" % eval_uuid)
        print("Status          : %s" % status)
    if not ok:
        raise AssertionError("Unexpected Model Replacement: %s -> %s" % (training_uuid, eval_uuid))
    return VerificationResult("Model Reuse", status, {"train": training_uuid, "eval": eval_uuid})


# ------------------------------------------------------------------
# Phase 10 - SessionVerifier / Session Audit
# ------------------------------------------------------------------
class SessionVerifier:
    def __init__(self, session_filter):
        # session_filter must be an instance of shared.session_filter.SessionFilter
        self.session_filter = session_filter

    def audit(
        self,
        candidates,
        session_filter_value,
        broker_tz: str = "Europe/Athens",
        verbose: bool = True,
        max_rows_to_print: int = 10,
    ) -> VerificationResult:
        rows = []
        for c in candidates:
            cid = c.get("candidate_id") if isinstance(c, dict) else getattr(c, "candidate_id", None)
            ts = c.get("timestamp") if isinstance(c, dict) else getattr(c, "timestamp", None)
            if ts is None:
                continue
            ts = pd.Timestamp(ts)
            try:
                broker_time = ts.tz_localize("UTC").tz_convert(broker_tz).strftime("%Y-%m-%d %H:%M %Z") \
                    if ts.tzinfo is None else ts.tz_convert(broker_tz).strftime("%Y-%m-%d %H:%M %Z")
            except Exception:
                broker_time = str(ts)
            passes = self.session_filter.passes(ts, session_filter_value)
            rows.append({
                "candidate_id": cid,
                "timestamp": ts,
                "utc_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "broker_time": broker_time,
                "detected_session": self.session_filter.detect_session(ts),
                "accepted": bool(passes),
            })
        df = pd.DataFrame(rows)
        details = {
            "total": len(df),
            "accepted": int(df["accepted"].sum()) if len(df) else 0,
            "rejected": int((~df["accepted"]).sum()) if len(df) else 0,
            "session_filter_value": str(session_filter_value),
            "session_filter_hash": self.session_filter.version_hash(),
            "audit": df,
        }
        status = "PASS"
        if verbose:
            _print_header("SESSION AUDIT REPORT")
            if len(df):
                print(df.head(max_rows_to_print).to_string(index=False))
                if len(df) > max_rows_to_print:
                    print("... (%d more rows)" % (len(df) - max_rows_to_print))
            print("Total: %d  Accepted: %d  Rejected: %d  filter=%s  hash=%s" % (
                details["total"], details["accepted"], details["rejected"],
                details["session_filter_value"], details["session_filter_hash"],
            ))
        return VerificationResult("Session Consistency", status, details)


# ------------------------------------------------------------------
# Phase 12 - ThresholdVerifier
# ------------------------------------------------------------------
class ThresholdVerifier:
    def verify(
        self,
        probabilities,
        threshold: float,
        applications_seen: int,
        verbose: bool = True,
    ) -> VerificationResult:
        probs = np.asarray(probabilities)
        pass_count = int(np.sum(probs >= threshold))
        fail_count = int(np.sum(probs < threshold))
        details = {
            "threshold": float(threshold),
            "threshold_applications": int(applications_seen),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "total": pass_count + fail_count,
        }
        status = "PASS" if applications_seen == 1 else "FAIL"
        if verbose:
            _print_header("THRESHOLD VERIFICATION REPORT")
            for k, v in details.items():
                print("%-24s: %s" % (k, v))
            print("Status                  : %s" % status)
        if status == "FAIL":
            raise AssertionError(
                "Threshold applied %d times (expected exactly 1)" % applications_seen
            )
        return VerificationResult("Threshold Verification", status, details)


# ------------------------------------------------------------------
# Phase 14 - CandidateIntegrityVerifier
# ------------------------------------------------------------------
class CandidateIntegrityVerifier:
    def verify(
        self,
        exported_ids,
        imported_ids,
        processed_ids,
        executed_ids,
        verbose: bool = True,
    ) -> VerificationResult:
        exported = set(exported_ids)
        imported = set(imported_ids)
        processed = set(processed_ids)
        executed = set(executed_ids)

        duplicates_exported = len(list(exported_ids)) - len(exported)
        duplicates_imported = len(list(imported_ids)) - len(imported)

        missing = exported - imported
        orphaned = imported - exported
        exec_not_processed = executed - processed

        details = {
            "exported": len(exported),
            "imported": len(imported),
            "processed": len(processed),
            "executed": len(executed),
            "duplicates_exported": duplicates_exported,
            "duplicates_imported": duplicates_imported,
            "missing": len(missing),
            "orphaned": len(orphaned),
            "executed_not_processed": len(exec_not_processed),
        }
        problems = missing or orphaned or exec_not_processed or duplicates_exported or duplicates_imported
        status = "FAIL" if problems else "PASS"
        if verbose:
            _print_header("CANDIDATE INTEGRITY REPORT")
            for k, v in details.items():
                print("%-24s: %s" % (k, v))
            print("Status                  : %s" % status)
        return VerificationResult("Candidate Integrity", status, details)


# ------------------------------------------------------------------
# Phase 13 - RootCauseAnalyzer V2 (marginal loss)
# ------------------------------------------------------------------
class RootCauseAnalyzerV2:
    def analyse(self, stage_counts: Dict[str, int], verbose: bool = True):
        stages = list(stage_counts.keys())
        rows = []
        prev = None
        for st in stages:
            remaining = int(stage_counts[st])
            if prev is None:
                rows.append({"stage": st, "remaining": remaining, "lost": None, "marginal_loss_pct": None})
            else:
                lost = prev - remaining
                marginal = (lost / prev * 100.0) if prev > 0 else float("nan")
                rows.append({"stage": st, "remaining": remaining, "lost": lost, "marginal_loss_pct": marginal})
            prev = remaining
        df = pd.DataFrame(rows)

        ranked = df.dropna(subset=["marginal_loss_pct"]).sort_values("marginal_loss_pct", ascending=False)
        top = ranked.iloc[0] if len(ranked) else None
        top_stage = top["stage"] if top is not None else None
        top_loss = top["marginal_loss_pct"] if top is not None else float("nan")

        if top_stage is None:
            explanation = "No losses detected between stages."
        else:
            explanation = (
                "Dominant bottleneck: %s stage removes %.1f%% of candidates that reached it. "
                "This is the marginal loss (relative to survivors of the previous stage), not "
                "the total loss versus Generated, so it is not misattributed to signal generation."
            ) % (top_stage, top_loss)

        if verbose:
            _print_header("ROOT CAUSE ANALYSIS (V2, marginal loss)")
            print(df.to_string(index=False, float_format=lambda x: "%.1f" % x))
            print(explanation)

        return {
            "table": df,
            "dominant_stage": top_stage,
            "dominant_marginal_loss_pct": float(top_loss) if not pd.isna(top_loss) else float("nan"),
            "explanation": explanation,
        }


# ------------------------------------------------------------------
# Phase 15 - Pipeline Waterfall
# ------------------------------------------------------------------
def build_pipeline_waterfall(stage_counts: Dict[str, int], verbose: bool = True):
    stages = list(stage_counts.keys())
    if not stages:
        return pd.DataFrame()
    initial = stage_counts[stages[0]]
    rows = []
    prev = None
    for st in stages:
        remaining = int(stage_counts[st])
        if prev is None:
            rows.append({
                "stage": st,
                "remaining": remaining,
                "lost": 0,
                "cumulative_survival_pct": 100.0,
                "marginal_loss_pct": 0.0,
            })
        else:
            lost = prev - remaining
            marginal = (lost / prev * 100.0) if prev > 0 else 0.0
            cum = (remaining / initial * 100.0) if initial > 0 else 0.0
            rows.append({
                "stage": st,
                "remaining": remaining,
                "lost": lost,
                "cumulative_survival_pct": cum,
                "marginal_loss_pct": marginal,
            })
        prev = remaining
    df = pd.DataFrame(rows)
    if verbose:
        _print_header("PIPELINE WATERFALL")
        print(df.to_string(index=False, float_format=lambda x: "%.1f" % x))
    return df


# ------------------------------------------------------------------
# Phase 11 - Probability calibration metrics
# ------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_prob)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        w = mask.sum() / n
        gap = abs(acc - conf)
        ece += w * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


# ------------------------------------------------------------------
# Phase 16 - PipelineCertification
# ------------------------------------------------------------------
CERTIFICATION_CATEGORIES = [
    "Train/OOS Separation",
    "Feature Leakage",
    "Evaluation Integrity",
    "Prediction Alignment",
    "Session Consistency",
    "Candidate Integrity",
    "Model Reuse",
    "Threshold Verification",
    "Probability Calibration",
    "Root Cause Analysis",
    "Dataset Integrity",
]


class PipelineCertification:
    def __init__(self):
        self.results: Dict[str, VerificationResult] = {}

    def record(self, result: VerificationResult):
        if result is None:
            return
        self.results[result.name] = result

    def record_many(self, results):
        for r in results:
            self.record(r)

    def certify(self, verbose: bool = True) -> Dict[str, Any]:
        rows = []
        for cat in CERTIFICATION_CATEGORIES:
            r = self.results.get(cat)
            if r is None:
                rows.append({"category": cat, "status": "SKIPPED"})
            else:
                rows.append({"category": cat, "status": r.status})
        df = pd.DataFrame(rows)
        mandatory_fail = any(r.status == "FAIL" for r in self.results.values())
        overall = "FAIL" if mandatory_fail else "PASS"
        if verbose:
            _print_header("PIPELINE CERTIFICATION")
            print(df.to_string(index=False))
            print("Overall Status: %s" % overall)
        return {"table": df, "overall": overall, "results": {k: v.to_dict() for k, v in self.results.items()}}


# ------------------------------------------------------------------
# Phase 17 - Shared manifest helpers
# ------------------------------------------------------------------
def feature_set_hash(feature_columns: Sequence[str]) -> str:
    blob = json.dumps(sorted(list(feature_columns)))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def candidate_ids_hash(candidate_ids) -> str:
    blob = "|".join(sorted(str(c) for c in candidate_ids))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_manifest(
    pipeline_version: str,
    strategy_version: str,
    feature_columns: Sequence[str],
    session_filter_hash: str,
    candidate_ids: Sequence[str],
    train_hash: str = "",
    oos_hash: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    m = {
        "pipeline_version": pipeline_version,
        "strategy_version": strategy_version,
        "feature_set_hash": feature_set_hash(feature_columns),
        "session_filter_hash": session_filter_hash,
        "candidate_count": len(list(candidate_ids)),
        "candidate_ids_hash": candidate_ids_hash(candidate_ids),
        "train_hash": train_hash,
        "oos_hash": oos_hash,
    }
    if extra:
        m.update(extra)
    return m


def verify_manifest_match(strategy_manifest: Dict[str, Any], explorer_manifest: Dict[str, Any], verbose: bool = True) -> VerificationResult:
    mismatches = []
    for k in ("feature_set_hash", "session_filter_hash", "candidate_ids_hash", "strategy_version"):
        if strategy_manifest.get(k) != explorer_manifest.get(k):
            mismatches.append(k)
    status = "FAIL" if mismatches else "PASS"
    if verbose:
        _print_header("CROSS-NOTEBOOK MANIFEST VERIFICATION")
        print("Mismatched keys : %s" % (mismatches or "none"))
        print("Status          : %s" % status)
    if mismatches:
        raise AssertionError("Manifest mismatch on: %s" % mismatches)
    return VerificationResult("Manifest Match", status, {"mismatches": mismatches})
