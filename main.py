import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load .env for Telegram credentials (silently ignored if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.logger import setup_logger
from src.processor import process_pipeline, TF_CONFIG, PROCESSED_PATH
from src.engine_hmm import fit_hmm, save_model as save_hmm, load_model as load_hmm, get_model_path as hmm_model_path
from src.engine_xgb import (
    prepare_features, train_xgb, get_predictions,
    export_onnx, save_xgb, load_xgb, ONNX_PATH, FEATURE_COLS,
    train_xgb_ensemble, get_predictions_ensemble,
    save_xgb_ensemble, load_xgb_ensemble, export_onnx_ensemble, ENSEMBLE_PKL_PATH,
    get_ensemble_path,
)
from src.optimizer import run_optimization, get_best_params
from src.backtester import vectorized_backtest
from src.visualizer import generate_full_report
from src.risk_manager import AdaptiveRiskManager

logger = setup_logger("main")


def _resolve_balance(args) -> float:
    """--balance takes precedence over --min_cap."""
    return args.balance if args.balance is not None else args.min_cap


_M5_EXPIRY_HOURS   = 120   # 5 days


def _m5_meta_path(broker: str) -> Path:
    return Path(f"models/m5_meta_{broker}.json")


def _check_m5_readiness(tf: str, broker: str = "headway_cent") -> bool:
    """Return True if the M5 model is fresh enough for live trading.

    The M5 timeframe is sensitive to microstructure changes, so the model
    must have been optimised within the last 5 days.  A meta.json timestamp
    is written by cmd_optimize after each successful M5 study.
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


def _train_for_tf(tf: str, balance: float, broker: str, params: dict):
    """Shared train logic for a single timeframe. Returns (result, model_hmm, states, X, metrics)."""
    df = process_pipeline(
        obs_cov=params.get("obs_cov"),
        trans_cov=params.get("trans_cov"),
        save=True,
        tf=tf,
    )
    model_hmm, states, state_map = fit_hmm(
        df, n_states=params.get("n_states", TF_CONFIG[tf].get("n_states_default", 3))
    )
    X, y, df_aligned = prepare_features(df, states)
    models_ensemble, thresholds, metrics = train_xgb_ensemble(
        X, y,
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=params.get("n_estimators", 200),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 5),
        gamma=params.get("gamma", 1.0),
        reg_alpha=params.get("reg_alpha", 0.1),
    )
    _, probabilities = get_predictions_ensemble(models_ensemble, thresholds, X)
    states_aligned = states[df.index.isin(df_aligned.index)]
    split_idx = metrics.get("split_idx")
    result = vectorized_backtest(
        df_aligned, probabilities, states_aligned,
        split_idx=split_idx,
        account_size=balance,
        broker=broker,
        tf=tf,
        prob_threshold=params.get("prob_threshold"),
        short_threshold=params.get("short_threshold"),
    )
    return result, model_hmm, state_map, models_ensemble, thresholds, metrics, df_aligned, states_aligned, X, probabilities


def cmd_process(args):
    tfs = [t.strip().upper() for t in args.tf.split(",")]
    for tf in tfs:
        if tf not in TF_CONFIG:
            logger.error("Unknown timeframe '%s'. Valid: %s", tf, list(TF_CONFIG))
            continue
        try:
            df = process_pipeline(save=True, tf=tf)
            logger.info(
                "[%s] Done: %d bars | Range: %s to %s",
                tf, len(df), df.index.min(), df.index.max(),
            )
        except FileNotFoundError as e:
            logger.error(str(e))


def cmd_optimize(args):
    balance = _resolve_balance(args)
    broker = args.broker
    tfs = [t.strip().upper() for t in args.tf.split(",")]
    for tf in tfs:
        logger.info("Optimizing [%s] broker=%s balance=$%.0f trials=%d", tf, broker, balance, args.trials)
        study = run_optimization(n_trials=args.trials, balance=balance, broker=broker, tf=tf, n_jobs=args.n_jobs)
        arm = AdaptiveRiskManager(balance, broker=broker)
        limits = arm.get_trade_limits()
        print(f"\n=== Best Result [{tf}] ===")
        print(f"Score:         {study.best_value:.3f}")
        print(f"Broker:        {broker}")
        print(f"Balance:       ${balance:.0f} USD  ({arm}")
        print(f"Session Limit: {limits['max_daily_trades']} trade(s)/day | {limits['pos_per_trade']} pos/trade")
        print("Best Params:")
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
                "\nOptimization Complete. HMM States and XGBoost Weights have been updated. "
                "You are now cleared for M5 Live Trading for the next 5 days."
            )


def cmd_train(args):
    balance = _resolve_balance(args)
    broker = args.broker
    tf = args.tf.upper()

    try:
        params = get_best_params(balance=balance, broker=broker, tf=tf)
        logger.info("Using Optuna best params [%s/%s]: %s", tf, broker, params)
    except Exception:
        logger.warning("No Optuna study found for tf=%s broker=%s — using defaults", tf, broker)
        params = {}

    result, model_hmm, state_map, models_ensemble, thresholds, metrics, *_ = _train_for_tf(tf, balance, broker, params)

    # Guard: refuse to save a degenerate HMM — identical state means + near-zero persistence
    # indicate the Kalman/HMM params were bad (usually because Optuna params weren't loaded).
    _min_persist = min(model_hmm.transmat_[i, i] for i in range(model_hmm.n_components))
    if _min_persist < 0.70:
        logger.critical(
            "TRAINING ABORTED: degenerate HMM (min persistence=%.4f). "
            "Model NOT saved. Usually caused by missing Optuna study for tf=%s broker=%s. "
            "Ensure --mode optimize has completed and re-run --mode train.",
            _min_persist, tf, broker,
        )
        print(
            f"\nERROR: Degenerate HMM detected (min state persistence={_min_persist:.4f} < 0.70).\n"
            f"The model was NOT saved — saving it would only produce garbage signals.\n"
            f"\nFix: python main.py --mode optimize --tf {tf} --broker {broker} --trials 300\n"
            f"Then: python main.py --mode train    --tf {tf} --broker {broker}\n"
        )
        sys.exit(1)

    save_hmm(model_hmm, hmm_model_path(tf, broker))
    save_xgb_ensemble(models_ensemble, thresholds, metrics, get_ensemble_path(tf, broker))

    arm = AdaptiveRiskManager(balance, broker=broker)
    limits = arm.get_trade_limits()
    print(f"\n=== Training Results [{tf}] ===")
    print(f"Broker: {broker} | Balance: ${balance:.0f} | Tier: {'small' if arm.is_small_account else 'growth'}")
    print(f"Session: {limits['max_daily_trades']}/day | Positions: {limits['pos_per_trade']}/trade")
    print(f"Sharpe: {result['sharpe_ratio']:.3f} | MaxDD: {result['max_drawdown']*100:.1f}% "
          f"| WR: {result['win_rate']*100:.1f}% | Trades: {result['n_trades']}")
    if "oos_sharpe_ratio" in result:
        print(f"\n--- In-Sample ---")
        print(f"  Sharpe: {result['is_sharpe_ratio']:.3f} | DD: {result['is_max_drawdown']*100:.1f}% "
              f"| WR: {result['is_win_rate']*100:.1f}% | Trades: {result['is_n_trades']}")
        print(f"--- Out-of-Sample ---")
        print(f"  Sharpe: {result['oos_sharpe_ratio']:.3f} | DD: {result['oos_max_drawdown']*100:.1f}% "
              f"| WR: {result['oos_win_rate']*100:.1f}% | Trades: {result['oos_n_trades']}")
    print(f"\nModels saved. Run --mode export to generate ONNX.")


def cmd_compare(args):
    """Train and backtest the specified timeframes, then show a side-by-side comparison."""
    balance = _resolve_balance(args)
    broker = args.broker
    tfs = [t.strip().upper() for t in args.tf.split(",")]
    results = {}

    for tf in tfs:
        try:
            params = get_best_params(balance=balance, broker=broker, tf=tf)
        except Exception:
            params = {}

        try:
            result, *_ = _train_for_tf(tf, balance, broker, params)
            results[tf] = result
        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", tf, e)
            continue

    if len(results) < 2:
        avail = ", ".join(tfs)
        print(f"\nOnly {len(results)} timeframe(s) available. Run --mode process --tf {avail} first.")
        if results:
            tf, r = next(iter(results.items()))
            print(f"\n[{tf}] Sharpe={r['sharpe_ratio']:.3f} | OOS={r.get('oos_sharpe_ratio',0):.3f} "
                  f"| DD={r['max_drawdown']*100:.1f}% | Trades={r['n_trades']}")
        return

    def _oos_sharpe(r):
        return r.get("oos_sharpe_ratio", r.get("sharpe_ratio", 0.0))

    winner   = max(results, key=lambda k: _oos_sharpe(results[k]))
    col_w    = 12
    tf_list  = list(results.keys())
    header   = f"{'Metric':<20}" + "".join(f" {tf:>{col_w}}" for tf in tf_list)

    print(f"\n{'='*50}")
    print(f"  TIMEFRAME COMPARISON  |  Balance: ${balance:.0f}  |  Broker: {broker}")
    print(f"{'='*50}")
    print(header)
    print("-" * len(header))
    metrics_to_show = [
        ("OOS Sharpe",   "oos_sharpe_ratio", ".3f"),
        ("IS Sharpe",    "is_sharpe_ratio",  ".3f"),
        ("OOS Max DD",   "oos_max_drawdown", ".1%"),
        ("OOS Win Rate", "oos_win_rate",     ".1%"),
        ("OOS Trades",   "oos_n_trades",     "d"),
        ("Full Sharpe",  "sharpe_ratio",     ".3f"),
    ]
    for label, key, fmt in metrics_to_show:
        row = f"{label:<20}"
        for tf in tf_list:
            r   = results[tf]
            val = r.get(key, r.get(key.replace("oos_", "").replace("is_", ""), 0))
            s   = f"{val:{fmt}}" if fmt != "d" else str(int(val))
            row += f" {s:>{col_w}}"
        print(row)

    for tf, r in results.items():
        logger.info(
            "  %s  Sharpe=%.3f (OOS=%.3f) | DD=%.1f%% | WR=%.1f%% | Trades=%d",
            tf, r["sharpe_ratio"], r.get("oos_sharpe_ratio", r["sharpe_ratio"]),
            r["max_drawdown"] * 100, r["win_rate"] * 100, r["n_trades"],
        )

    print(f"\n-> Recommended timeframe: {winner} (higher OOS Sharpe)")
    print(f"  Run: python main.py --mode train --tf {winner} --broker {broker} --balance {balance:.0f}")


def cmd_export(args):
    tf     = args.tf.upper()
    broker = args.broker
    xgb_path = get_ensemble_path(tf, broker)
    if not xgb_path.exists():
        xgb_path = ENSEMBLE_PKL_PATH
    try:
        models_ensemble, _, xgb_metrics = load_xgb_ensemble(xgb_path)
    except FileNotFoundError:
        logger.error("No trained ensemble model found. Run --mode train first.")
        sys.exit(1)
    feature_cols = xgb_metrics.get("feature_cols", FEATURE_COLS)
    n_features   = len(feature_cols)
    paths = export_onnx_ensemble(models_ensemble, n_features=n_features)
    print(f"\nONNX ensemble exported ({n_features} features: {feature_cols}):")
    for bucket, path in paths.items():
        print(f"  [{bucket:>4}]  {path}")
    print("\nCopy these files to your MT5 MQL5/Files/ directory.")
    print("The EA selects the model based on the current ATR volatility bucket.")


def cmd_sync_validate(args):
    """Download recent MT5 bars then run the model validation gatekeeper."""
    from src.mt5_sync import sync_mt5_data
    from src.validator import run_validation

    balance = _resolve_balance(args)
    tf      = args.tf.upper()

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

    print(f"\n=== Validation Result [{tf}] ===")
    print(f"  Recent Sharpe: {result['sharpe']:.3f}")
    print(f"  Trades:        {result['n_trades']}")
    print(f"  Win Rate:      {result['win_rate']*100:.1f}%")
    print(f"  Max Drawdown:  {result['max_dd']*100:.1f}%")
    print(f"  Status:        {result['status'].upper()}")
    print(f"  {result['message']}")

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

    logger.info("Starting demo loop — TF=%s  broker=%s  balance=$%.0f",
        tf, args.broker, balance,
    )
    run_live_loop(tf=tf, broker=args.broker, account_size=balance,
                  prob_threshold_override=getattr(args, "prob_threshold", None),
                  short_threshold_override=getattr(args, "short_threshold", None),
                  profit_target=getattr(args, "profit_target", None))


def cmd_live(args):
    """Connect to MT5 live account and start the live signal execution loop."""
    from src.mt5_trader import run_live_loop

    balance = _resolve_balance(args)
    tf      = args.tf.upper()

    if not _check_m5_readiness(tf, args.broker):
        sys.exit(1)

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
                  prob_threshold_override=getattr(args, "prob_threshold", None),
                  short_threshold_override=getattr(args, "short_threshold", None),
                  profit_target=getattr(args, "profit_target", None))


def cmd_report(args):
    balance = _resolve_balance(args)
    broker = args.broker
    tf = args.tf.upper()

    try:
        params = get_best_params(balance=balance, broker=broker, tf=tf)
    except Exception:
        params = {}

    df = process_pipeline(obs_cov=params.get("obs_cov"), trans_cov=params.get("trans_cov"),
                          save=False, tf=tf)

    _hmm_path = hmm_model_path(tf, broker)
    if not _hmm_path.exists():
        _hmm_path = None  # let the except branch fit a fresh one
    try:
        if _hmm_path is None:
            raise FileNotFoundError
        model_hmm = load_hmm(_hmm_path)
        from src.engine_hmm import predict_states, STATE_NAMES_3, STATE_NAMES_2, STATE_NAMES_4
        states = predict_states(model_hmm, df)
        n = model_hmm.n_components
        state_names = {2: STATE_NAMES_2, 3: STATE_NAMES_3, 4: STATE_NAMES_4}.get(n, STATE_NAMES_3)
    except Exception:
        model_hmm, states, state_names = fit_hmm(df, n_states=params.get("n_states", 3))

    X, y, df_aligned = prepare_features(df, states)

    _xgb_path = get_ensemble_path(tf, broker)
    if not _xgb_path.exists():
        _xgb_path = ENSEMBLE_PKL_PATH
    try:
        models_xgb, thresholds_xgb, metrics = load_xgb_ensemble(_xgb_path)
        # If the saved ensemble was trained with different features (e.g. before DXY
        # was added), retrain in-memory so the report stays consistent with the data.
        if metrics.get("feature_cols") != list(X.columns):
            raise ValueError("Feature mismatch — retraining for report.")
    except (FileNotFoundError, ValueError):
        models_xgb, thresholds_xgb, metrics = train_xgb_ensemble(
            X, y,
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 200),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 5),
            gamma=params.get("gamma", 1.0),
            reg_alpha=params.get("reg_alpha", 0.1),
        )

    _, probabilities = get_predictions_ensemble(models_xgb, thresholds_xgb, X)
    states_aligned = states[df.index.isin(df_aligned.index)]

    # split_idx in saved metrics may come from a different TF's training run
    # (e.g. M15 split_idx ~186k used in an H1 report with only ~58k bars).
    # Recompute from current data when the stored value doesn't fit.
    split_idx = metrics.get("split_idx")
    if split_idx is None or not (0 < split_idx < len(X)):
        split_idx = int(len(X) * 0.8)

    result = vectorized_backtest(
        df_aligned, probabilities, states_aligned,
        split_idx=split_idx, account_size=balance, broker=broker, tf=tf,
        prob_threshold=params.get("prob_threshold"),
        short_threshold=params.get("short_threshold"),
    )

    arm = AdaptiveRiskManager(balance, broker=broker)
    limits = arm.get_trade_limits()
    display_params = dict(params)
    display_params.update({
        "broker": broker,
        "balance_usd": balance,
        "tier": "small" if arm.is_small_account else "growth",
        "session_limit": limits["max_daily_trades"],
        "pos_per_trade": limits["pos_per_trade"],
        "tf": tf,
    })

    paths = generate_full_report(
        df_aligned, states_aligned, state_names, model_hmm,
        X, probabilities, metrics, result, display_params,
        split_idx=split_idx, tf=tf, broker=broker,
    )

    print(f"\n=== Report Generated [{tf} / {broker}] ===")
    for p in paths:
        print(f"  {p}")
    print(f"\n5 charts saved to reports/{tf}_{broker}/")


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
    """Consolidate all USDCHF CSV exports in data/raw/ into USDCHF_master.csv."""
    from src.data_consolidator import consolidate_usdchf
    result = consolidate_usdchf()
    if result.empty:
        logger.error(
            "Consolidation produced no data. Place USDCHF CSV files in data/raw/ "
            "with 'USDCHF' in the filename (e.g. USDCHF_5m_data.csv)."
        )
    else:
        print(f"\nUSDCHF master built: {len(result)} rows → data/processed/USDCHF_master.csv\n")


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


def main():
    parser = argparse.ArgumentParser(description="Gold Regime X — Hybrid ML Trading System")
    parser.add_argument(
        "--mode",
        choices=["process", "optimize", "train", "compare", "export", "report",
                 "sync_validate", "demo", "live", "audit", "guardian", "listen", "consolidate"],
        required=True,
    )
    parser.add_argument("--trials",   type=int,   default=250)
    parser.add_argument("--n_jobs",   type=int,   default=1,
                        help="Parallel Optuna trial workers (default 1). "
                             "See optimizer.py for caveats with n_jobs>1.")
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
    parser.add_argument("--prob_threshold",  type=float, default=None,
                        help="Override BUY probability threshold for --mode live (skips Optuna lookup).")
    parser.add_argument("--short_threshold", type=float, default=None,
                        help="Override SELL probability threshold for --mode live (skips Optuna lookup).")
    parser.add_argument("--profit_target",  type=float, default=None,
                        help="Quick-profit close threshold in USD.  M5 defaults to 4.0; "
                             "other TFs disabled unless set.  Pass 0 to disable on M5.")

    args = parser.parse_args()
    {
        "process":       cmd_process,
        "optimize":      cmd_optimize,
        "train":         cmd_train,
        "compare":       cmd_compare,
        "export":        cmd_export,
        "report":        cmd_report,
        "sync_validate": cmd_sync_validate,
        "demo":          cmd_demo,
        "live":          cmd_live,
        "audit":         cmd_audit,
        "guardian":      cmd_guardian,
        "listen":        cmd_listen,
        "consolidate":   cmd_consolidate,
    }[args.mode](args)


if __name__ == "__main__":
    main()
