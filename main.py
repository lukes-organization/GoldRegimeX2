import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 output on Windows so emoji/em-dash in log lines don't crash the terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env for Telegram credentials (silently ignored if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.logger import setup_logger, reconfigure_for_tf
from src.optimizer import (
    run_optimization, get_best_params, _score_result as _calc_score,
    extract_consensus_params, run_wfa as optimizer_run_wfa,
    WFO_PARAMS, WFO_PARAMS_FAST,
    run_optimization_stage1,
    CPCV_N_BLOCKS, CPCV_K_TEST, _N_PATHS,
    resolve_n_states as _optimizer_resolve_n_states,
)
from src.risk_manager import AdaptiveRiskManager
from src.strategy_backtest import backtest_tf, load_live_bundle

logger = setup_logger("main")


def _resolve_balance(args) -> float:
    """--balance takes precedence over --min_cap."""
    return args.balance if args.balance is not None else args.min_cap


def resolve_n_states(tf: str, params: dict) -> int:
    """Return the canonical n_states for *tf* in both optimize and train paths.

    The canonical regime contract mandates exactly 3 states for ALL timeframes:
    TREND (0), MEAN_REVERSION (1), VOLATILITY_SHOCK (2).
    Passing any other value raises immediately so mis-configured runs fail fast.
    """
    requested = int(params.get("n_states", 3))
    if requested != 3:
        raise ValueError(
            f"n_states must be 3 for all TFs (canonical regime contract). "
            f"Received {requested} for {tf}. "
            f"Supported states: TREND(0), MEAN_REVERSION(1), VOLATILITY_SHOCK(2)."
        )
    return 3


def _extract_cpcv_metrics(cpcv_json: dict) -> dict:
    """Extract key CPCV metrics from unified schema or legacy flat payload."""
    agg = cpcv_json.get("cpcv_aggregate_stats") if isinstance(cpcv_json, dict) else None
    if isinstance(agg, dict):
        return {
            "score": float(agg.get("cpcv_score", cpcv_json.get("final_score_and_penalties", {}).get("final_score", 0.0))),
            "n_valid_paths": int(agg.get("n_valid_paths", 0)),
            "median_sharpe": float(agg.get("median_sharpe", 0.0)),
            "median_trades": int(agg.get("median_trades", 0)),
            "median_win_rate": float(agg.get("median_win_rate", 0.0)),
            "median_drawdown": float(agg.get("median_drawdown", 0.0)),
            "median_return": float(agg.get("median_return", 0.0)),
            "std_sharpe": float(agg.get("std_sharpe", 0.0)),
        }
    return {
        "score": float(cpcv_json.get("cpcv_score", 0.0)),
        "n_valid_paths": int(cpcv_json.get("n_valid_paths", 0)),
        "median_sharpe": float(cpcv_json.get("median_sharpe", 0.0)),
        "median_trades": int(cpcv_json.get("median_trades", 0)),
        "median_win_rate": float(cpcv_json.get("median_win_rate", 0.0)),
        "median_drawdown": float(cpcv_json.get("median_drawdown", 0.0)),
        "median_return": float(cpcv_json.get("median_return", 0.0)),
        "std_sharpe": float(cpcv_json.get("std_sharpe", 0.0)),
    }


_M5_EXPIRY_HOURS   = 120   # 5 days

# Staleness thresholds per TF (days).  If the saved model is older than this
# the live gate aborts with a warning and Telegram alert.
# M5 is tightest (14d) — microstructure regimes shift weekly.
# H1/M15 are more stable but should still re-optimise monthly.
_MODEL_STALE_DAYS  = {"M5": 14, "M15": 30, "H1": 30}


def _m5_meta_path(broker: str) -> Path:
    return Path(f"models/m5_meta_{broker}.json")


def _check_m5_readiness(tf: str, broker: str = "headway_cent") -> bool:
    """Return True if the M5 model is fresh enough for live trading.

    The M5 timeframe is sensitive to microstructure changes, so the model
    must have been optimised within the last 5 days.  A meta.json timestamp
    is written by cmd_optimize after each successful M5 optimization.
    """
    if tf.upper() != "M5":
        return True
    meta_path = _m5_meta_path(broker)
    if not meta_path.exists():
        # Legacy fallback: files created before per-broker naming used models/m5_meta.json
        _legacy = Path("models/m5_meta.json")
        if _legacy.exists():
            logger.info(
                "Migrating legacy m5_meta.json -> %s for broker=%s.", meta_path, broker
            )
            meta_path.write_text(_legacy.read_text())
        else:
            print(
                "\nERROR: M5 model meta-data not found.\n"
                "Run  python main.py --mode optimize --tf M5  before live trading."
            )
            return False
    meta     = json.loads(meta_path.read_text())
    age_h    = (time.time() - meta.get("timestamp", 0)) / 3600
    if age_h > _M5_EXPIRY_HOURS:
        print(
            f"\nWARNING: M5 model is {age_h:.0f} hours old (limit: {_M5_EXPIRY_HOURS}h / 5 days).\n"
            "REQUIRED ACTION: Run  python main.py --mode optimize --tf M5  to refresh before live.\n"
            "\n  Pre-flight checklist:\n"
            "  1. python main.py --mode sync_validate --period 3m --tf M5\n"
            "  2. python main.py --mode optimize --tf M5 --trials 500 --balance 15\n"
            "  3. python main.py --mode train --tf M5\n"
        )
        return False
    return True




def _check_model_staleness(tf: str, broker: str, args) -> None:
    """Abort live/demo start if the saved model exceeds the staleness threshold.

    Sends a Telegram alert and calls ``sys.exit(1)`` when the model is too old.
    Pass ``--skip_stale_check`` on the CLI to bypass this gate (e.g. for demo
    testing when you intentionally don't want to re-optimise).

    Does nothing for unknown TFs or when ``--skip_stale_check`` is set.
    """
    if getattr(args, "skip_stale_check", False):
        logger.info("Staleness gate bypassed (--skip_stale_check).")
        return

    from src.validator import check_model_age
    from src.notifier import send_telegram_msg

    max_age  = _MODEL_STALE_DAYS.get(tf.upper(), 30)
    age_days = check_model_age(tf=tf, broker=broker)

    if age_days <= max_age:
        logger.info(
            "Model freshness OK: %s/%s is %.1f days old (limit %d).",
            tf, broker, age_days, max_age,
        )
        return

    age_str = f"{age_days:.0f}" if age_days != float("inf") else "unknown (file missing)"
    msg = (
        f"⚠️ Market Drift/Staleness detected. Pausing trade loop — "
        f"[{tf}] model is {age_str} days old (limit: {max_age} days).\n"
        f"Re-optimise before going live:\n"
        f"  python main.py --mode optimize --tf {tf} --broker {broker} --trials 500\n"
        f"  python main.py --mode train    --tf {tf} --broker {broker}\n"
        f"Add --skip_stale_check to the live command to bypass this gate."
    )
    logger.warning("STALE MODEL [%s/%s]: %.0f days old — aborting.", tf, broker, age_days)
    send_telegram_msg(f"<b>{msg}</b>")
    print(f"\n{msg}")
    sys.exit(1)


def cmd_wfa(args):
    print('[RETIRED] --mode wfa relied on the legacy ML stack (processor/HMM/XGB/backtester), removed in the notebook-engine consolidation. Use --mode optimize (grid-search-plateau); see CONSOLIDATION.md.')
    raise SystemExit(2)


def cmd_process(args):
    print('[RETIRED] --mode process relied on the legacy ML stack (processor/HMM/XGB/backtester), removed in the notebook-engine consolidation. Use --mode optimize (grid-search-plateau); see CONSOLIDATION.md.')
    raise SystemExit(2)


def cmd_optimize(args):
    balance  = _resolve_balance(args)
    broker   = args.broker
    tfs      = [t.strip().upper() for t in args.tf.split(",")]
    wfo_mode = "fast" if getattr(args, "fast_wfo", False) else "standard"
    stage    = getattr(args, "stage", None)

    for tf in tfs:
        reconfigure_for_tf(tf)

        # Auto-sync raw CSV from MT5 first; silently skipped when MT5 is unavailable.
        try:
            from src.mt5_sync import ensure_data_updated
            ensure_data_updated(tf=tf, symbol="XAUUSD")
        except Exception as _sync_exc:
            logger.warning("Auto-sync skipped (%s). Continuing with existing data.", _sync_exc)

        logger.info(
            "Optimizing [%s] broker=%s balance=$%.0f trials=%d wfo_mode=%s stage=%s",
            tf, broker, balance, args.trials, wfo_mode, stage or "joint",
        )

        if stage is not None:
            print(
                f"[DEPRECATED] --stage {stage!r} is ignored. "
                "Running the unified grid-search-plateau pipeline."
            )

        # The notebook engine loads its own multi-asset panels and trains
        # M15 + M5 in a single pass, so no processed parquet is required here.
        study = run_optimization(
            df=None,
            tf=tf,
            broker=broker,
            account_size=balance,
            n_trials=args.trials,
            wfo_mode=wfo_mode,
            n_jobs=args.n_jobs,
        )

        print("")
        print(f"=== Best Result [{tf}] ===")
        print(f"Score:         {study.best_value:.3f}")
        print(f"Broker:        {broker}  |  Balance: ${balance:.0f}  |  WFO mode: {wfo_mode}")
        print("Best Params (grid-search-plateau center):")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        if tf == "M5":
            meta_path = _m5_meta_path(broker)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps({
                "timestamp":  time.time(),
                "tf":         "M5",
                "best_score": study.best_value,
            }))
            print(
                "Optimization Complete. The live model bundle has been exported. "
                "You are cleared for M5 Live Trading for the next 5 days."
            )


def cmd_train(args):
    balance = _resolve_balance(args)
    broker  = args.broker
    tf      = args.tf.upper()
    reconfigure_for_tf(tf)

    # With the notebook-engine consolidation, --mode optimize already trains AND
    # exports the single live model bundle (models + thresholds + base_params).
    # "train" now loads that exact bundle and reports a full-period backtest, so
    # the reported edge matches what will actually trade.
    try:
        bundle = load_live_bundle()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(
            f"ERROR: No live model bundle found for [{tf}]. "
            f"Run  python main.py --mode optimize --tf {tf} --broker {broker}  first."
        )
        sys.exit(1)

    try:
        params = get_best_params(balance=balance, broker=broker, tf=tf)
        logger.info("Grid-search-plateau params [%s/%s]: %s", tf, broker, params)
    except Exception:
        logger.warning("Could not read exported params for tf=%s broker=%s.", tf, broker)

    try:
        metrics = backtest_tf(tf, bundle=bundle)
    except Exception as exc:
        logger.error("Backtest via the notebook engine failed: %s", exc)
        sys.exit(1)

    arm = AdaptiveRiskManager(balance, broker=broker)

    cpcv_score = None
    _cpcv_json_path = Path(f"reports/cpcv_{tf.lower()}_{broker}.json")
    if _cpcv_json_path.exists():
        try:
            cpcv_score = _extract_cpcv_metrics(json.loads(_cpcv_json_path.read_text()))["score"]
        except Exception:
            cpcv_score = None

    print("")
    print(f"=== Training Results [{tf}] ===")
    print(f"Broker: {broker} | Balance: ${balance:.0f} | Tier: {'small' if arm.is_small_account else 'growth'}")
    print("Model bundle exported by --mode optimize (single-source live model).")
    print(
        f"Full-period Sharpe: {metrics.get('sharpe', 0.0):.3f}"
        f" | MaxDD: {metrics.get('max_drawdown', 0.0):.1f}%"
        f" | WR: {metrics.get('win_rate', 0.0) * 100:.1f}%"
        f" | Trades: {metrics.get('trade_count', 0)}"
    )
    if cpcv_score is not None:
        print(f"CPCV Validation Score (best trial): {cpcv_score:.3f}")
    print("")
    print("Live model bundle is ready. Run --mode sync_validate before going live.")


def cmd_extract_consensus(args):
    print('[RETIRED] --mode extract_consensus relied on the legacy ML stack (processor/HMM/XGB/backtester), removed in the notebook-engine consolidation. Use --mode optimize (grid-search-plateau); see CONSOLIDATION.md.')
    raise SystemExit(2)


def cmd_compare(args):
    print('[RETIRED] --mode compare relied on the legacy ML stack (processor/HMM/XGB/backtester), removed in the notebook-engine consolidation. Use --mode optimize (grid-search-plateau); see CONSOLIDATION.md.')
    raise SystemExit(2)


def cmd_export(args):
    print('[RETIRED] --mode export relied on the legacy ML stack (processor/HMM/XGB/backtester), removed in the notebook-engine consolidation. Use --mode optimize (grid-search-plateau); see CONSOLIDATION.md.')
    raise SystemExit(2)


def cmd_sync_validate(args):
    """Download recent MT5 bars then run the model validation gatekeeper."""
    from src.mt5_sync import sync_mt5_data
    from src.validator import run_validation, validate_strategy

    balance = _resolve_balance(args)
    tf      = args.tf.upper()
    reconfigure_for_tf(tf)

    logger.info("Syncing MT5 data [%s] period=%s ...", tf, args.period)
    try:
        df = sync_mt5_data(tf=tf, period=args.period)
        logger.info(
            "Sync complete: %d bars  %s -> %s",
            len(df), df.index.min(), df.index.max(),
        )
    except Exception as exc:
        logger.error("MT5 sync failed: %s", exc)
        sys.exit(1)

    try:
        result = run_validation(tf=tf, broker=args.broker, account_size=balance)
    except Exception as exc:
        logger.error("Validation error: %s", exc)
        sys.exit(1)

    # ── Deployment gate (Phase F) ─────────────────────────────────────────
    gate = validate_strategy(result, tf=tf)
    gate_status = gate["status"]
    gate_reason = gate["reason"]
    gate_details = gate["details"]

    print(f"\n=== Validation Result [{tf}] ===")
    _fdd     = result.get("max_dd", 0.0)
    _eff     = result.get("avg_efficiency", 0.0)
    _cost_e  = result.get("cost_efficiency", 0.0)
    _tr_pct  = result.get("total_return", 0.0)
    _payout  = f"{_tr_pct:+.1f}% (${balance * _tr_pct / 100.0:+,.2f})"
    print(
        f"  [{tf} LIVE] Score: {result.get('score', 0.0):.2f}"
        f" | RF: {result.get('recovery_factor', 0.0):.2f}"
        f" | PF: {result.get('profit_factor', 1.0):.2f}"
        f" | Payoff: ${result.get('expected_payoff', 0.0)*balance:.4f}"
        f" | MaxDD: {_fdd*100:.1f}% (Floating)"
    )
    print(
        f"  Efficiency: {_eff:.2f}x ATR/Spread"
        f" | CostEff: {_cost_e*100:.1f}%"
        f" | Total Payout: {_payout}"
    )
    if _eff < 1.2:
        print("  [!] WARNING: Low Market Efficiency — Spread is eating your edge.")
    if _cost_e < 0.50:
        print(f"  [!] WARNING: Broker is consuming >{(1-_cost_e)*100:.0f}% of gross profit.")
    print(f"  Sharpe: {result['sharpe']:.3f} | Trades: {result['n_trades']} | WR: {result['win_rate']*100:.1f}%")
    print(f"  Status: {result['status'].upper()}")
    print(f"  {result['message']}")

    # Gate result
    if gate_status == "fail":
        print(f"\n[DEPLOYMENT GATE] FAIL — reason: {gate_reason}")
        print(f"  Details: {gate_details}")
        print(
            "\nABORTING: Deployment gate failed. "
            "Retune with --mode optimize then --mode train before going live."
        )
        sys.exit(1)
    if gate_status == "warn":
        print(f"\n[DEPLOYMENT GATE] WARN — {gate_reason}  |  details: {gate_details}")
        print("  WARNING: Borderline performance — consider re-optimising before live trading.")
    else:
        print(f"\n[DEPLOYMENT GATE] PASS — all checks OK  |  details: {gate_details}")

    if result["status"] == "fail":
        print(
            "\nABORTING: Validation failed. "
            "Retune with --mode optimize then --mode train before going live."
        )
        sys.exit(1)
    if result["status"] == "warn":
        print("\nWARNING: Borderline performance — consider re-optimising before live trading.")


def cmd_demo(args):
    """Connect to MT5 demo account and start the live signal execution loop."""
    from src.mt5_trader import run_live_loop

    balance = _resolve_balance(args)
    tf      = args.tf.upper()

    if not _check_m5_readiness(tf, args.broker):
        sys.exit(1)

    _check_model_staleness(tf, args.broker, args)

    logger.info("Starting demo loop — TF=%s  broker=%s  balance=$%.0f",
        tf, args.broker, balance,
    )
    run_live_loop(tf=tf, broker=args.broker, account_size=balance,
                  profit_target=getattr(args, "profit_target", None))


def cmd_live(args):
    """Connect to MT5 live account and start the live signal execution loop."""
    from src.mt5_trader import run_live_loop

    balance = _resolve_balance(args)
    tf      = args.tf.upper()
    reconfigure_for_tf(tf)

    if not _check_m5_readiness(tf, args.broker):
        sys.exit(1)

    _check_model_staleness(tf, args.broker, args)

    if not args.yes:
        print("\n" + "=" * 60)
        print("  WARNING: LIVE ACCOUNT — real money is at risk.")
        print("  Ensure  --mode sync_validate  passed before continuing.")
        print("  Ensure the GoldRegimeX EA is removed from the XAUUSD chart.")
        confirm = input("  Type  YES  to confirm live trading: ").strip()
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)
        print("=" * 60 + "\n")

    logger.info(
        "Starting live loop — TF=%s  broker=%s  balance=$%.0f",
        tf, args.broker, balance,
    )
    run_live_loop(tf=tf, broker=args.broker, account_size=balance,
                  profit_target=getattr(args, "profit_target", None))


def cmd_report(args):
    print('[RETIRED] --mode report relied on the legacy ML stack (processor/HMM/XGB/backtester), removed in the notebook-engine consolidation. Use --mode optimize (grid-search-plateau); see CONSOLIDATION.md.')
    raise SystemExit(2)


def cmd_audit(args):
    """Print (and optionally send) the daily MT5 performance report."""
    from src.auditor import get_daily_report
    from src.notifier import send_telegram_msg

    balance = _resolve_balance(args)
    report  = get_daily_report(broker=args.broker)
    print(report)
    sent    = send_telegram_msg(report)
    if sent:
        print("\nReport also sent to Telegram.")
    else:
        print("\n(Telegram not configured — see .env.example)")


def cmd_guardian(args):
    """Start the multi-TF health monitor loop."""
    from src.guardian import run_guardian

    balance  = _resolve_balance(args)
    tfs      = [t.strip().upper() for t in args.tf.split(",")]
    interval = getattr(args, "interval", 3600)

    run_guardian(
        tfs=tfs,
        broker=args.broker,
        account_size=balance,
        period=args.period,
        interval_sec=interval,
    )


def cmd_consolidate(args):
    """Consolidate multi-asset CSV exports in data/raw/ into per-TF master files.

    Processes 5 assets × 3 timeframes = 15 master files in data/processed/.
    Assets: USDCHF, XAGUSD, XTIUSD, US500, USDJPY.
    Timeframes: H1, M15, M5.
    """
    from src.data_consolidator import consolidate_asset, ASSET_CONFIGS

    assets = list(ASSET_CONFIGS.keys())
    tfs    = ["H1", "M15", "M5"]

    n_ok = 0
    for asset in assets:
        for tf in tfs:
            result = consolidate_asset(asset, tf)
            if not result.empty:
                out = ASSET_CONFIGS[asset][tf]["output"]
                print(f"  {asset} {tf}: {len(result)} rows → data/processed/{out}")
                n_ok += 1
            else:
                logger.warning("%s %s: no source files found — skipping.", asset, tf)

    print(f"\nConsolidation complete: {n_ok}/{len(assets) * len(tfs)} master files produced.")


def cmd_listen(args):
    """Start the Telegram remote control listener + nightly report scheduler."""
    import threading
    from src.remote_control import run_listener
    from src.notifier import send_telegram_msg
    from src.auditor import get_daily_report

    balance = _resolve_balance(args)

    # ── Nightly audit scheduler (runs in background thread) ──────────────────
    def _run_scheduler():
        try:
            import schedule
        except ImportError:
            logger.warning(
                "schedule package not installed — nightly report disabled. "
                "Install with: pip install schedule"
            )
            return

        def _send_nightly():
            report = get_daily_report(broker=args.broker)
            send_telegram_msg(f"<b>Nightly Report</b>\n{report}")
            logger.info("Nightly report sent via Telegram.")

        schedule.every().day.at("23:55").do(_send_nightly)
        logger.info("Nightly report scheduled at 23:55 UTC.")
        while True:
            schedule.run_pending()
            time.sleep(60)

    sched_thread = threading.Thread(target=_run_scheduler, daemon=True)
    sched_thread.start()

    # ── Blocking Telegram listener ────────────────────────────────────────────
    run_listener()



def cmd_sensitivity(args):
    print('[RETIRED] --mode sensitivity relied on the legacy ML stack (processor/HMM/XGB/backtester), removed in the notebook-engine consolidation. Use --mode optimize (grid-search-plateau); see CONSOLIDATION.md.')
    raise SystemExit(2)


def cmd_montecarlo(args):
    print('[RETIRED] --mode montecarlo relied on the legacy ML stack (processor/HMM/XGB/backtester), removed in the notebook-engine consolidation. Use --mode optimize (grid-search-plateau); see CONSOLIDATION.md.')
    raise SystemExit(2)


def _validate_h1_args(args) -> None:
    """Fail-fast validation for H1 gate/floor CLI overrides.

    Called before mode dispatch so invalid ranges produce a clear error rather
    than a cryptic crash inside the optimizer or signal engine.
    """
    checks = [
        ("h1_entry_prob",         0.50, 0.90,  "[0.50, 0.90]"),
        ("h1_min_median_sharpe",  -1.0, 2.0,   "[-1.0, 2.0]"),
        ("h1_min_median_pf",       0.5, 3.0,   "[0.5, 3.0]"),
        ("h1_max_trades_per_100",  0.5, 20.0,  "[0.5, 20.0]"),
    ]
    for attr, lo, hi, rng in checks:
        val = getattr(args, attr, None)
        if val is not None and not (lo <= val <= hi):
            print(
                f"\nERROR: --{attr} {val} is out of valid range {rng}.\n"
                f"Fix: pass a value within {rng} or omit the flag to use the default.\n"
            )
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Gold Regime X — Hybrid ML Trading System")
    parser.add_argument(
        "--mode",
        choices=["process", "optimize", "train", "compare", "export", "report",
                 "sync_validate", "sync_validation", "demo", "live", "audit", "guardian", "listen",
                 "consolidate", "wfa", "sensitivity", "extract_consensus",
                 "montecarlo"],
        required=True,
    )
    parser.add_argument("--trials",   type=int,   default=250)
    parser.add_argument("--n_jobs",   type=int,   default=1,
                        help="Parallel grid-search-plateau workers (default 1). "
                             "Passed through to the grid-search engine.")
    parser.add_argument("--interval", type=int,   default=3600,
                        help="Guardian check interval in seconds (default 3600 = 1h).")
    parser.add_argument("--min_cap", type=float, default=15.0,
                        help="Account capital in USD (legacy, use --balance)")
    parser.add_argument("--balance", type=float, default=None,
                        help="Account balance in USD — overrides --min_cap")
    parser.add_argument("--broker",  type=str,   default="standard",
                        choices=["standard", "headway_cent"])
    parser.add_argument("--tf",      type=str,   default="H1",
                        help="Timeframe: H1 | M15 | M15,H1 (process/compare accept comma list)")
    parser.add_argument("--period",  type=str,   default="3m",
                        help="Lookback window for MT5 sync, e.g. '3m' '6m' '12m'.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive live-account confirmation (used when launched as a subprocess).")
    parser.add_argument("--profit_target",  type=float, default=None,
                        help="Quick-profit close threshold in USD.  M5 defaults to 4.0; "
                             "other TFs disabled unless set.  Pass 0 to disable on M5.")
    parser.add_argument("--skip_stale_check", action="store_true",
                        help="Bypass the model-staleness gate on --mode live/demo. "
                             "Use when intentionally running an older model (e.g. demo testing).")
    parser.add_argument("--train_days", type=int, default=None,
                        help="WFA IS window in calendar days (default: H1=365, M15=180, M5=90).")
    parser.add_argument("--test_days",  type=int, default=None,
                        help="WFA OOS step size in calendar days (default: H1=90, M15=60, M5=30).")
    parser.add_argument("--fast_wfo", action="store_true",
                        help="Use faster WFO window sizes for --mode optimize and --mode wfa.")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Top-N trials to aggregate for --mode extract_consensus (default 10).")
    parser.add_argument("--min_wfe", type=float, default=0.0,
                        help="Minimum WFE ratio filter for --mode extract_consensus (default 0).")
    parser.add_argument(
        "--stage", type=str, default=None, choices=["xgb", "trading"],
        help=(
            "[DEPRECATED] --stage is deprecated and ignored. "
            "Unified single-stage CPCV optimization is always used. "
            "Accepted for backward compatibility only."
        ),
    )
    # ── H1 profitability safeguards (override optimizer defaults) ─────────────
    parser.add_argument(
        "--h1_entry_prob", type=float, default=None,
        help="H1 XGBoost entry probability gate override [0.50, 0.90] (default 0.575). "
             "Passed as fixed override to live/demo; also primes the search range start for optimize.",
    )
    parser.add_argument(
        "--h1_min_median_sharpe", type=float, default=None,
        help="H1 minimum acceptable median CPCV Sharpe floor. Trials below are penalized. "
             "Default 0.10. Valid range [-1.0, 2.0].",
    )
    parser.add_argument(
        "--h1_min_median_pf", type=float, default=None,
        help="H1 minimum acceptable median profit factor. Default 1.02. Valid range [0.5, 3.0].",
    )
    parser.add_argument(
        "--h1_max_trades_per_100", type=float, default=None,
        help="H1 maximum trades per 100 bars before turnover penalty fires. "
             "Default 4.0. Valid range [0.5, 20.0].",
    )
    args = parser.parse_args()
    _validate_h1_args(args)
    {
        "process":       cmd_process,
        "optimize":      cmd_optimize,
        "train":         cmd_train,
        "compare":       cmd_compare,
        "export":        cmd_export,
        "report":        cmd_report,
        "sync_validate":  cmd_sync_validate,
        "sync_validation": cmd_sync_validate,
        "demo":          cmd_demo,
        "live":          cmd_live,
        "audit":         cmd_audit,
        "guardian":      cmd_guardian,
        "listen":        cmd_listen,
        "consolidate":   cmd_consolidate,
        "wfa":           cmd_wfa,
        "sensitivity":   cmd_sensitivity,
        "extract_consensus": cmd_extract_consensus,
        "montecarlo":    cmd_montecarlo,
    }[args.mode](args)


if __name__ == "__main__":
    main()
