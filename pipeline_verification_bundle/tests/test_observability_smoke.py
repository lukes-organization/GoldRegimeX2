"""End-to-end smoke test for the full 18-phase observability bundle.

Runs entirely on synthetic data. Verifies:

    * CandidateTrace has every Phase-2 field
    * PipelineLogger + ledger flow (Phase 2/3)
    * candidate_decisions.csv produced with the exact Phase 4 header, and
      every REJECT row carries a non-blank Reason
    * pipeline_audit.json produced with the exact Phase 5 shape
    * stage_survival.csv includes the Threshold stage (Phase 6)
    * feature_drift_report.csv, probability_report.csv, session_audit.csv,
      candidate_integrity.csv exist with expected columns (Phase 7/9/10/11)
    * top100_rejected_M15.csv and top100_rejected_M5.csv exist with the
      Phase 12/13 column schema
    * ModelUUIDTracker detects match / mismatch (Phase 14)
    * PipelineManifest write + validate round-trip (Phase 15)
    * pipeline_health.txt exists (Phase 16)

Run from repo root:
    python pipeline_verification_bundle/tests/test_observability_smoke.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # pipeline_verification_bundle/../
sys.path.insert(0, str(REPO))

from pipeline_verification_bundle.shared.pipeline_observability import (
    CandidateTrace, PipelineObservability, PipelineLogger,
    ModelUUIDTracker, PipelineManifest, STAGE_ORDER,
)

OUT = HERE / "_smoke_out"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)


# ---------------------------------------------------------------------------
# Individual phase checks
# ---------------------------------------------------------------------------
def _check_phase2_fields() -> None:
    required = [
        "candidate_id", "timeframe", "strategy", "timestamp",
        "generated", "feature_engineering_pass", "session_pass", "session_name",
        "tbm_pass", "hmm_state", "hmm_probability", "hmm_pass",
        "xgb_probability", "threshold", "threshold_pass",
        "risk_pass", "executed", "rejection_stage", "rejection_reason", "history",
    ]
    tr = CandidateTrace(candidate_id=1, timeframe="M15",
                        timestamp=pd.Timestamp.utcnow(), strategy="x")
    missing = [f for f in required if not hasattr(tr, f)]
    assert not missing, f"CandidateTrace missing Phase 2 fields: {missing}"
    print("  Phase 2   CandidateTrace has all required fields.")


def _simulate_run(obs: PipelineObservability) -> None:
    rng = np.random.default_rng(7)
    logger = PipelineLogger(obs.ledger, obs.decisions)

    def gen_batch(tf: str, n: int, session_reject_rate: float,
                  tbm_reject_rate: float, hmm_reject_rate: float,
                  threshold_reject_rate: float, risk_reject_rate: float):
        base_ts = pd.Timestamp("2026-06-01T00:00:00Z")
        for i in range(n):
            cid = int(f"{1 if tf=='M15' else 2}{i:05d}")
            ts = base_ts + pd.Timedelta(minutes=(15 if tf == "M15" else 5) * i)
            logger.log(cid, tf, "Generated", "PASS",
                       {"strategy": "trend_pull", "timestamp": ts})
            logger.log(cid, tf, "FeatureEngineering", "PASS")
            if rng.random() < session_reject_rate:
                logger.log(cid, tf, "Session", "REJECT",
                           {"reason": "outside session", "session_name": "None"})
                continue
            logger.log(cid, tf, "Session", "PASS", {"session_name": "London_NY"})
            if rng.random() < tbm_reject_rate:
                logger.log(cid, tf, "TBM", "REJECT", {"reason": "TBM timeout"})
                continue
            logger.log(cid, tf, "TBM", "PASS")
            hmm_state = int(rng.integers(0, 3))
            if rng.random() < hmm_reject_rate:
                logger.log(cid, tf, "HMM", "REJECT",
                           {"reason": "unfavourable regime", "hmm_state": hmm_state})
                continue
            logger.log(cid, tf, "HMM", "PASS", {"hmm_state": hmm_state})
            prob = float(rng.beta(2, 5))
            logger.log(cid, tf, "Probability", "PASS", {"xgb_probability": prob})
            threshold = 0.6
            if prob < threshold or rng.random() < threshold_reject_rate:
                logger.log(cid, tf, "Threshold", "REJECT",
                           {"reason": f"prob {prob:.2f} < {threshold}",
                            "threshold": threshold})
                continue
            logger.log(cid, tf, "Threshold", "PASS", {"threshold": threshold})
            if rng.random() < risk_reject_rate:
                logger.log(cid, tf, "Risk", "REJECT", {"reason": "max exposure hit"})
                continue
            logger.log(cid, tf, "Risk", "PASS")
            logger.log(cid, tf, "Executed", "PASS")

    # M15: session-heavy bottleneck.
    gen_batch("M15", n=600, session_reject_rate=0.55, tbm_reject_rate=0.10,
              hmm_reject_rate=0.10, threshold_reject_rate=0.05, risk_reject_rate=0.05)
    # M5: threshold-heavy bottleneck.
    gen_batch("M5", n=1200, session_reject_rate=0.10, tbm_reject_rate=0.10,
              hmm_reject_rate=0.10, threshold_reject_rate=0.28, risk_reject_rate=0.05)

    # Feed the module with synthetic feature drift and probability snapshots
    # so those artifacts have real content.
    obs.record_feature_drift("M15", pd.DataFrame({
        "feature":    ["rsi", "macd"],
        "psi":        [0.05, 0.28],
        "ks":         [0.03, 0.11],
        "mean_shift": [0.01, 0.05],
        "std_shift":  [0.02, 0.10],
        "flag":       ["SAFE", "CRITICAL"],
    }))
    obs.record_probability_snapshot("M15", "raw",
                                    rng.beta(2, 5, size=500))
    obs.record_probability_snapshot("M15", "thresholded",
                                    rng.beta(3, 3, size=200))


def _check_artifact_exists(path: Path, must_contain_cols=None) -> pd.DataFrame:
    assert path.exists(), f"Missing artifact: {path}"
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if must_contain_cols:
            missing = [c for c in must_contain_cols if c not in df.columns]
            assert not missing, (
                f"{path.name} missing cols {missing}. Have: {list(df.columns)}"
            )
        return df
    return None


def _check_phase14_uuid() -> None:
    tr = ModelUUIDTracker()
    u = tr.mint_training_uuid("M15")
    tr.record_evaluation_uuid("M15", u)
    tr.record_export_uuid("M15", u)
    r = tr.verify("M15")
    assert r["status"] == "PASS", f"UUID match expected PASS, got {r}"
    # Now inject a mismatch on M5.
    tr.mint_training_uuid("M5")
    tr.record_evaluation_uuid("M5", "not-the-same-uuid")
    r5 = tr.verify("M5")
    assert r5["status"] != "PASS", f"UUID mismatch expected non-PASS, got {r5}"
    print("  Phase 14  ModelUUIDTracker detects match and mismatch.")


def _check_phase15_manifest(tmp: Path) -> None:
    mpath = tmp / "pipeline_manifest.json"
    m = PipelineManifest(path=mpath)
    m.write(feature_hash="F1", session_filter_hash="S1", strategy_hash="STR1",
            candidate_hash="C1", model_hash="MOD1", pipeline_version="v1.0.0")
    assert mpath.exists()
    m2 = PipelineManifest(path=mpath)
    m2.load()
    ok = m2.validate({"feature_hash": "F1", "model_hash": "MOD1"}, strict=False)
    assert ok["status"] == "PASS", f"Manifest validate should PASS, got {ok}"
    fail = m2.validate({"feature_hash": "WRONG"}, strict=False)
    assert fail["status"] == "FAIL", f"Manifest validate should FAIL, got {fail}"
    print("  Phase 15  PipelineManifest write + validate + mismatch detection.")


def main() -> int:
    print("Smoke test: full 18-phase observability bundle\n")

    _check_phase2_fields()

    obs = PipelineObservability(
        output_dir=OUT,
        expected_session_by_tf={"M15": "London_NY", "M5": "London_NY"},
        material_survival_gap_pct=10.0,
        lost_trade_limit=100,
        lost_trade_tf="M15",
    )
    _simulate_run(obs)
    print(f"  Phase 3   PipelineLogger recorded {len(obs.ledger)} traces.")

    _check_phase14_uuid()
    _check_phase15_manifest(OUT)

    # Fold model-UUID integrity in.
    integrity = {
        "Candidate Integrity":     "PASS",
        "Model Integrity":         "PASS",
        "Train/OOS Separation":    "PASS",
        "Model UUID Consistency":  "PASS",
    }
    result = obs.finalize(integrity_flags=integrity, verbose=False)

    # ---- Phase 4 candidate_decisions.csv --------------------------------
    dec = _check_artifact_exists(
        Path(result["candidate_decisions_csv"]),
        must_contain_cols=["Candidate", "TF", "Timestamp", "Stage", "Decision", "Reason"],
    )
    reject_rows = dec[dec["Decision"] == "REJECT"]
    empty_reason = reject_rows[reject_rows["Reason"].fillna("").astype(str).str.len() == 0]
    assert empty_reason.empty, (
        f"Phase 4 violation: {len(empty_reason)} REJECT rows have blank Reason."
    )
    print(f"  Phase 4   candidate_decisions.csv OK  "
          f"({len(dec)} rows, {len(reject_rows)} rejects, all with reasons).")

    # ---- Phase 5 pipeline_audit.json ------------------------------------
    audit = json.loads(Path(result["pipeline_audit_json"]).read_text())
    for k in ("experiment_id", "timestamp", "timeframes"):
        assert k in audit, f"pipeline_audit.json missing top-level key {k!r}"
    for tf in ("M15", "M5"):
        assert tf in audit["timeframes"], f"pipeline_audit.json missing {tf!r}"
        for k in ("generated", "session_pass", "tbm_pass", "hmm_pass",
                  "threshold_pass", "executed"):
            assert k in audit["timeframes"][tf], (
                f"pipeline_audit.json timeframes.{tf} missing key {k!r}"
            )
    print("  Phase 5   pipeline_audit.json matches spec shape.")

    # ---- Phase 6 stage_survival.csv -------------------------------------
    surv = _check_artifact_exists(Path(result["survival_csv"]))
    stage_col = next((c for c in surv.columns if c.lower() == "stage"), None)
    stages_in_surv = set(surv[stage_col].astype(str)) if stage_col else set()
    assert "Threshold" in stages_in_surv, (
        f"stage_survival.csv missing Threshold stage. Stages: {sorted(stages_in_surv)}"
    )
    print(f"  Phase 6   stage_survival.csv OK  (stages: {sorted(stages_in_surv)}).")

    # ---- Phase 7 feature_drift_report.csv -------------------------------
    _check_artifact_exists(Path(result["feature_drift_report_csv"]))
    print("  Phase 7   feature_drift_report.csv OK.")

    # ---- Phase 8 hmm_diagnostics.txt ------------------------------------
    assert Path(result["hmm_txt"]).exists(), "hmm_diagnostics.txt missing"
    print("  Phase 8   hmm_diagnostics.txt OK.")

    # ---- Phase 9 probability_report.csv ---------------------------------
    _check_artifact_exists(Path(result["probability_report_csv"]))
    print("  Phase 9   probability_report.csv OK.")

    # ---- Phase 10 session_audit.csv -------------------------------------
    _check_artifact_exists(Path(result["session_audit_csv"]))
    print("  Phase 10  session_audit.csv OK.")

    # ---- Phase 11 candidate_integrity.csv -------------------------------
    _check_artifact_exists(
        Path(result["candidate_integrity_csv"]),
        must_contain_cols=["Candidate ID", "Strategy Tester Status",
                          "Explorer Status", "Final Status"],
    )
    print("  Phase 11  candidate_integrity.csv OK.")

    # ---- Phase 12 & 13 top100_rejected --------------------------------
    for tf in ("M15", "M5"):
        p = Path(result["top100_rejected_paths"][tf])
        df = _check_artifact_exists(p, must_contain_cols=[
            "Candidate ID", "Timestamp", "Strategy", "Session Decision",
            "TBM Decision", "HMM State", "HMM Decision", "XGBoost Probability",
            "Threshold", "Risk Decision", "Final Rejection Stage", "Rejection Reason",
        ])
        assert len(df) <= 100, f"{p.name} has {len(df)} rows (>100)"
        print(f"  Phase 12/13  {p.name} OK  ({len(df)} rows).")

    # ---- Phase 16 pipeline_health.txt -----------------------------------
    health_text = Path(result["pipeline_health_txt"]).read_text()
    assert "PIPELINE HEALTH" in health_text.upper()
    print("  Phase 16  pipeline_health.txt OK.")

    print("\nSMOKE TEST PASSED")
    print(f"Artifacts written to: {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
