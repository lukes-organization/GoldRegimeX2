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
from src.processor import process_pipeline, TF_CONFIG, PROCESSED_PATH, save_feature_scaler, load_feature_scaler
from src.engine_hmm import fit_hmm, save_model as save_hmm, load_model as load_hmm, get_model_path as hmm_model_path
from src.engine_xgb import (
    prepare_features, train_xgb, get_predictions,
    export_onnx, save_xgb, load_xgb, ONNX_PATH, FEATURE_COLS,
    train_xgb_ensemble, get_predictions_ensemble,
    save_xgb_ensemble, load_xgb_ensemble, export_onnx_ensemble, ENSEMBLE_PKL_PATH,
    get_ensemble_path, TF_TRAIN_RATIO, compute_regime_stats,
)
from src.optimizer import (
    run_optimization, get_best_params, _score_result as _calc_score,
    extract_consensus_params, run_wfa as optimizer_run_wfa,
    WFO_PARAMS, WFO_PARAMS_FAST,
    run_optimization_stage1,
    CPCV_N_BLOCKS, CPCV_K_TEST, _N_PATHS,
)
from src.backtester import vectorized_backtest, format_payout
from src.visualizer import generate_full_report
from src.risk_manager import AdaptiveRiskManager

logger = setup_logger("main")


def _resolve_balance(args) -> float:
    """--balance takes precedence over --min_cap."""
    return args.balance if args.balance is not None else args.min_cap


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
        save_models=True,
        broker=broker,
    )
    model_hmm, states, state_map = fit_hmm(
        df, n_states=params.get("n_states", TF_CONFIG[tf].get("n_states_default", 3))
    )
    X, y, df_aligned, feature_scaler = prepare_features(df, states, tf=tf)
    save_feature_scaler(feature_scaler, tf=tf, broker=broker)
    models_ensemble, thresholds, metrics = train_xgb_ensemble(
        X, y,
        train_ratio=1.0,  # full-data training — WFO validation happened in optimize step
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=params.get("n_estimators", 200),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 5),
        gamma=params.get("gamma", 1.0),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
    )
    _, probabilities = get_predictions_ensemble(models_ensemble, thresholds, X)
    states_aligned = states[df.index.isin(df_aligned.index)]
    metrics["regime_stats"] = compute_regime_stats(models_ensemble, thresholds, X, states_aligned)
    result = vectorized_backtest(
        df_aligned, probabilities, states_aligned,
        split_idx=None,   # full-data model — no IS/OOS split; CPCV scores used for validation
        account_size=balance,
        broker=broker,
        tf=tf,
        hmm_transmat=model_hmm.transmat_,  # use real persistence values, consistent with optimizer CPCV
    )
    return result, model_hmm, state_map, models_ensemble, thresholds, metrics, df_aligned, states_aligned, X, probabilities


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
    """Walk-Forward Analysis using CPCV.

    Uses the best Optuna params from the study DB to evaluate all
    C(N,K) combinatorial train/test paths (CPCV).  Prints per-path OOS
    diagnostics and path-score statistics.
    """
    import pandas as pd
    from src.notifier import send_telegram_msg

    balance  = _resolve_balance(args)
    broker   = args.broker
    tf       = args.tf.upper()
    wfo_mode = "fast" if getattr(args, "fast_wfo", False) else "standard"
    reconfigure_for_tf(tf)

    processed_path = TF_CONFIG[tf]["processed_path"]
    if not processed_path.exists():
        print(f"\nERROR: Run --mode process --tf {tf} first.")
        sys.exit(1)

    df = pd.read_parquet(processed_path)
    print(f"\n=== Walk-Forward Analysis [{tf} / {broker}] (mode: {wfo_mode}) ===")
    print(f"  Dataset: {len(df)} bars  ({df.index[0].date()} – {df.index[-1].date()})")
    print(f"  Running CPCV ({CPCV_N_BLOCKS} blocks, C({CPCV_N_BLOCKS},{CPCV_K_TEST})={_N_PATHS} paths) with best Optuna params...\n")

    wfa = optimizer_run_wfa(
        df=df, tf=tf, broker=broker,
        account_size=balance, wfo_mode=wfo_mode,
    )

    n_paths   = wfa["n_windows"]
    n_valid   = wfa["n_valid_windows"]
    scores    = wfa["window_scores"]
    verdict   = "ROBUST" if n_valid >= n_paths // 2 else "FRAGILE — re-optimise recommended"

    print(f"  Total CPCV paths  : {n_paths}")
    print(f"  Valid paths       : {n_valid}  (OOS trades above hard floor, DD < 20%)")
    print(f"  Median OOS score  : {float(__import__('numpy').median(scores)) if scores else 0.0:+.3f}")
    print(f"  Std OOS Sharpe    : {wfa['std_sharpe']:.3f}")
    print(f"  Median OOS trades : {wfa['median_trades']}")
    print(f"  Verdict           : {verdict}")
    print(f"  CPCV score        : {wfa['wfo_score']:+.3f}")

    if scores:
        print("\n  Per-path scores:")
        for i, s in enumerate(scores):
            bar = "#" * max(0, int((s + 1) * 5))
            print(f"    Path  {i+1:>2}: {s:+.3f}  {bar}")

    send_telegram_msg(
        f"<b>Walk-Forward Analysis [{tf}]</b>\n"
        f"Valid paths: <b>{n_valid}/{n_paths}</b>  |  CPCV score: <b>{wfa['wfo_score']:+.3f}</b>\n"
        + ("Robust" if n_valid >= n_paths // 2 else "Fragile — re-optimise")
    )


def cmd_process(args):
    tfs = [t.strip().upper() for t in args.tf.split(",")]
    for tf in tfs:
        if tf not in TF_CONFIG:
            logger.error("Unknown timeframe '%s'. Valid: %s", tf, list(TF_CONFIG))
            continue
        reconfigure_for_tf(tf)
        try:
            df = process_pipeline(save=True, tf=tf, save_models=True, broker=args.broker)
            logger.info(
                "[%s] Done: %d bars | Range: %s to %s",
                tf, len(df), df.index.min(), df.index.max(),
            )
        except FileNotFoundError as e:
            logger.error(str(e))


def cmd_optimize(args):
    import pandas as pd
    balance  = _resolve_balance(args)
    broker   = args.broker
    tfs      = [t.strip().upper() for t in args.tf.split(",")]
    wfo_mode = "fast" if getattr(args, "fast_wfo", False) else "standard"
    stage    = getattr(args, "stage", None)   # None | "xgb" | "trading"

    for tf in tfs:
        reconfigure_for_tf(tf)
        logger.info(
            "Optimizing [%s] broker=%s balance=$%.0f trials=%d wfo_mode=%s stage=%s",
            tf, broker, balance, args.trials, wfo_mode, stage or "joint",
        )

        # Pre-load parquet once — make_objective only calls kalman_smooth per trial
        processed_path = TF_CONFIG[tf]["processed_path"]
        if not processed_path.exists():
            print(
                f"\nERROR: Processed parquet not found for {tf}.\n"
                f"Run  python main.py --mode process --tf {tf}  first."
            )
            sys.exit(1)
        df = pd.read_parquet(processed_path)

        if stage == "xgb":
            # ── Stage-1: fast single hold-out XGB exploration (no CPCV) ─────
            study = run_optimization_stage1(
                df=df, tf=tf, broker=broker,
                account_size=balance, n_trials=args.trials,
            )
        else:
            # ── Stage-2 / joint: full CPCV optimization ───────────────────────
            study = run_optimization(
                df=df,
                tf=tf,
                broker=broker,
                account_size=balance,
                n_trials=args.trials,
                wfo_mode=wfo_mode,
                n_jobs=args.n_jobs,
                warm_start_stage1=(stage == "trading"),
            )

        print(f"\n=== Best Result [{tf}] ===")
        print(f"Score:         {study.best_value:.3f}")
        print(f"Broker:        {broker}  |  Balance: ${balance:.0f}  |  WFO mode: {wfo_mode}")
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
    reconfigure_for_tf(tf)

    try:
        params = get_best_params(balance=balance, broker=broker, tf=tf)
        logger.info("Using Optuna best params [%s/%s]: %s", tf, broker, params)
    except Exception:
        logger.warning("No Optuna study found for tf=%s broker=%s — using defaults", tf, broker)
        params = {}

    result, model_hmm, state_map, models_ensemble, thresholds, metrics, df_aligned, states_aligned, X, probabilities = _train_for_tf(tf, balance, broker, params)

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
    cpcv_score = None
    try:
        from src.optimizer import get_best_params as _gbp
        _study = __import__("optuna").load_study(
            study_name=None,
            storage=f"sqlite:///models/study_{broker}.db",
        )
        cpcv_score = _study.best_value
    except Exception:
        pass

    print(f"\n=== Training Results [{tf}] ===")
    print(f"Broker: {broker} | Balance: ${balance:.0f} | Tier: {'small' if arm.is_small_account else 'growth'}")
    print(f"Full-data model trained on 100% of available bars (CPCV validation via optimize step).")
    print(f"Full-period Sharpe: {result['sharpe_ratio']:.3f} | MaxDD: {result['max_drawdown']*100:.1f}%"
          f" | WR: {result['win_rate']*100:.1f}% | Trades: {result['n_trades']}")
    if cpcv_score is not None:
        print(f"CPCV Validation Score (best trial): {cpcv_score:.3f}")
    print(f"\nModels saved. Run --mode export to generate ONNX.")


def cmd_extract_consensus(args):
    """Extract consensus hyperparameters from top-N trials (median per param).

    More robust than the single best trial. Prints the consensus params and
    saves them nowhere — use with --mode train to apply the consensus params.
    """
    tf      = args.tf.upper()
    broker  = args.broker
    top_n   = getattr(args, "top_n", 10)
    min_wfe = getattr(args, "min_wfe", 0.0)
    reconfigure_for_tf(tf)

    try:
        consensus = extract_consensus_params(tf=tf, broker=broker, top_n=top_n, min_wfe=min_wfe)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    meta = consensus.pop("meta", {})
    print(f"\n=== Consensus Params [{tf} / {broker}] ===")
    print(f"  Study:      {meta.get('study_name', '?')}")
    print(f"  Top-N used: {meta.get('top_n_actual', top_n)}")
    print(f"  Score range: [{meta.get('min_score', 0.0):.3f}, {meta.get('max_score', 0.0):.3f}]")
    print(f"  Mean WFE:   {meta.get('mean_wfe', 0.0):.3f}")
    print("\n  Consensus parameters (median of top-N):")
    for k, v in sorted(consensus.items()):
        print(f"    {k}: {v}")


def cmd_compare(args):
    """Train and backtest the specified timeframes, then show a side-by-side comparison."""
    balance = _resolve_balance(args)
    broker = args.broker
    tfs = [t.strip().upper() for t in args.tf.split(",")]
    results = {}

    for tf in tfs:
        reconfigure_for_tf(tf)
        try:
            params = get_best_params(balance=balance, broker=broker, tf=tf)
        except Exception:
            params = {}

        try:
            result, *_ = _train_for_tf(tf, balance, broker, params)
            if "oos_sharpe_ratio" in result:
                _oos_fdd = result.get("oos_floating_max_drawdown", result.get("oos_max_drawdown", 0.0))
                result["oos_score"] = _calc_score({
                    "total_return":          result.get("oos_total_return", 0.0),
                    "floating_max_drawdown": _oos_fdd,
                    "sharpe_ratio":          result.get("oos_sharpe_ratio", 0.0),
                    "profit_factor":         result.get("oos_profit_factor", 1.0),
                })
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

    def _oos_score(r):
        return r.get("oos_score", r.get("oos_sharpe_ratio", r.get("sharpe_ratio", 0.0)))

    winner   = max(results, key=lambda k: _oos_score(results[k]))
    col_w    = 12
    tf_list  = list(results.keys())
    header   = f"{'Metric':<20}" + "".join(f" {tf:>{col_w}}" for tf in tf_list)

    print(f"\n{'='*50}")
    print(f"  TIMEFRAME COMPARISON  |  Balance: ${balance:.0f}  |  Broker: {broker}")
    print(f"{'='*50}")
    print(header)
    print("-" * len(header))
    metrics_to_show = [
        ("OOS Score",    "oos_score",                  ".2f"),
        ("OOS Sharpe",   "oos_sharpe_ratio",            ".3f"),
        ("OOS RF",       "oos_recovery_factor",         ".2f"),
        ("OOS PF",       "oos_profit_factor",           ".2f"),
        ("IS Sharpe",    "is_sharpe_ratio",             ".3f"),
        ("OOS Float DD", "oos_floating_max_drawdown",   ".1%"),
        ("OOS Win Rate", "oos_win_rate",                ".1%"),
        ("OOS Trades",   "oos_n_trades",                "d"),
        ("Full Sharpe",  "sharpe_ratio",                ".3f"),
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

    print(f"\n-> Recommended timeframe: {winner} (higher OOS Score)")
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

    print(f"\n=== Validation Result [{tf}] ===")
    _fdd     = result.get("max_dd", 0.0)
    _eff     = result.get("avg_efficiency", 0.0)
    _cost_e  = result.get("cost_efficiency", 0.0)
    _payout  = format_payout(result.get("total_return", 0.0), balance, args.broker)
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
    balance = _resolve_balance(args)
    broker = args.broker
    tf = args.tf.upper()
    reconfigure_for_tf(tf)

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

    # Load the saved feature scaler so the report uses identical scaling to training
    try:
        _feat_scaler = load_feature_scaler(tf=tf, broker=broker)
    except FileNotFoundError:
        _feat_scaler = None   # old model without scaler — prepare_features fits fresh

    X, y, df_aligned, _ = prepare_features(df, states, feature_scaler=_feat_scaler, tf=tf)

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

    # Ensure regime_stats are present (may be absent in pre-Z-Score saved models)
    if not metrics.get("regime_stats"):
        _rs_split = metrics.get("split_idx") or int(len(X) * 0.8)
        _X_is     = X.iloc[:_rs_split]
        _hs_is    = states_aligned[:len(_X_is)]
        metrics["regime_stats"] = compute_regime_stats(models_xgb, thresholds_xgb, _X_is, _hs_is)

    # split_idx=None means the model was trained with train_ratio=1.0 (full-data /
    # CPCV mode).  In that case keep split_idx=None so the report backtest is
    # also full-period — matching the training output exactly.
    # Only fall back to 80/20 when a stored split_idx exists but doesn't fit the
    # current data (e.g. loaded from a model trained on a different TF).
    split_idx = metrics.get("split_idx")
    if split_idx is not None and not (0 < split_idx < len(X)):
        split_idx = int(len(X) * 0.8)

    result = vectorized_backtest(
        df_aligned, probabilities, states_aligned,
        split_idx=split_idx, account_size=balance, broker=broker, tf=tf,
        hmm_transmat=model_hmm.transmat_,  # use real persistence values, consistent with optimizer CPCV
    )

    # ── Inject CPCV OOS stats from last --mode optimize run if available ─────
    _cpcv_json = Path(f"reports/cpcv_{tf.lower()}_{broker}.json")
    if not _cpcv_json.exists():
        # Try to backfill from the existing Optuna study DB
        try:
            from src.optimizer import save_best_trial_cpcv
            save_best_trial_cpcv(broker=broker, tf=tf, balance=balance)
        except Exception as _be:
            logger.warning("Could not backfill CPCV JSON from study: %s", _be)
    if _cpcv_json.exists():
        try:
            _cpcv = json.loads(_cpcv_json.read_text())
            result["cpcv_score"]          = _cpcv.get("cpcv_score", 0.0)
            result["cpcv_n_valid_paths"]  = _cpcv.get("n_valid_paths", 0)
            result["oos_sharpe_ratio"]    = _cpcv.get("median_sharpe", 0.0)
            result["oos_n_trades"]        = _cpcv.get("median_trades", 0)
            result["oos_win_rate"]        = _cpcv.get("median_win_rate", 0.0)
            result["oos_max_drawdown"]    = _cpcv.get("median_drawdown", 0.0)
            result["oos_floating_max_drawdown"] = _cpcv.get("median_drawdown", 0.0)
            result["oos_total_return"]    = _cpcv.get("median_return", 0.0)
            result["cpcv_path_scores"]    = _cpcv.get("path_scores", [])
            result["cpcv_std_sharpe"]     = _cpcv.get("std_sharpe", 0.0)
            logger.info(
                "Loaded CPCV OOS stats from %s — median_sharpe=%.3f  median_trades=%d",
                _cpcv_json, result["oos_sharpe_ratio"], result["oos_n_trades"],
            )
        except Exception as _e:
            logger.warning("Could not load CPCV JSON (%s): %s", _cpcv_json, _e)

    arm = AdaptiveRiskManager(balance, broker=broker)
    limits = arm.get_trade_limits()
    display_params = dict(params)
    display_params.update({
        "broker": broker,
        "balance_usd": balance,
        "tier": "small" if arm.is_small_account else "growth",
        "pos_per_trade": limits["pos_per_trade"],
        "tf": tf,
    })

    paths = generate_full_report(
        df_aligned, states_aligned, state_names, model_hmm,
        X, probabilities, metrics, result, display_params,
        split_idx=split_idx, tf=tf, broker=broker,
        account_size=balance,
    )

    print(f"\n=== Report Generated [{tf} / {broker}] ===")
    for p in paths:
        print(f"  {p}")
    print(f"\n{len(paths)} charts saved to reports/{tf}_{broker}/")


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
    """Z-Score sensitivity analysis on already-trained models.

    Loops over Bull/Bear Z-Score cutoff values (1.5 .. 3.0 in 0.25 steps),
    reruns the OOS backtest for each, and prints a comparison table so you can
    see whether the current cutoff is optimal.  MR (Chop) cutoffs are held
    constant throughout.

    Usage:
        python main.py --mode sensitivity --tf H1 --broker headway_cent --balance 15
    """
    from src.sensitivity import run_sensitivity

    balance = _resolve_balance(args)
    tf      = args.tf.upper()
    broker  = args.broker
    reconfigure_for_tf(tf)

    # Load processed data + models exactly as cmd_report does
    try:
        params = get_best_params(balance=balance, broker=broker, tf=tf)
    except Exception:
        params = {}

    df = process_pipeline(
        obs_cov=params.get("obs_cov"), trans_cov=params.get("trans_cov"),
        save=False, tf=tf,
    )

    _hmm_path = hmm_model_path(tf, broker)
    if not _hmm_path.exists():
        _hmm_path = None
    try:
        if _hmm_path is None:
            raise FileNotFoundError
        model_hmm = load_hmm(_hmm_path)
        from src.engine_hmm import predict_states
        states = predict_states(model_hmm, df)
    except Exception:
        logger.warning("No saved HMM found for %s/%s — fitting fresh.", tf, broker)
        model_hmm, states, _ = fit_hmm(df, n_states=params.get("n_states", 3))

    try:
        _feat_scaler = load_feature_scaler(tf=tf, broker=broker)
    except FileNotFoundError:
        _feat_scaler = None

    X, y, df_aligned, _ = prepare_features(df, states, feature_scaler=_feat_scaler, tf=tf)

    _xgb_path = get_ensemble_path(tf, broker)
    if not _xgb_path.exists():
        _xgb_path = ENSEMBLE_PKL_PATH
    try:
        models_xgb, thresholds_xgb, metrics = load_xgb_ensemble(_xgb_path)
        if metrics.get("feature_cols") != list(X.columns):
            raise ValueError("Feature mismatch")
    except (FileNotFoundError, ValueError):
        logger.warning("No saved ensemble found for %s/%s — can't run sensitivity.", tf, broker)
        return

    _, probabilities = get_predictions_ensemble(models_xgb, thresholds_xgb, X)
    states_aligned = states[df.index.isin(df_aligned.index)]

    if not metrics.get("regime_stats"):
        _rs_split = metrics.get("split_idx") or int(len(X) * 0.8)
        metrics["regime_stats"] = compute_regime_stats(
            models_xgb, thresholds_xgb,
            X.iloc[:_rs_split], states_aligned[:_rs_split],
        )

    split_idx = metrics.get("split_idx")
    if split_idx is None or not (0 < split_idx < len(X)):
        split_idx = int(len(X) * 0.8)

    run_sensitivity(
        tf=tf,
        broker=broker,
        balance=balance,
        df_aligned=df_aligned,
        probabilities=probabilities,
        states_aligned=states_aligned,
        split_idx=split_idx,
        regime_stats=metrics["regime_stats"],
    )


def main():
    parser = argparse.ArgumentParser(description="Gold Regime X — Hybrid ML Trading System")
    parser.add_argument(
        "--mode",
        choices=["process", "optimize", "train", "compare", "export", "report",
                 "sync_validate", "demo", "live", "audit", "guardian", "listen",
                 "consolidate", "wfa", "sensitivity", "extract_consensus"],
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
    parser.add_argument("--epochs",     type=int, default=100,
                        help="TCN training epochs for --mode train_tcn (default 100).")
    parser.add_argument("--seq_len", type=int, default=100,
                        help="TCN sequence length in bars (default 100).")
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="Temperature for TCN probability calibration (default 1.5). "
                             ">1.0 softens; 1.0 = raw output; <1.0 sharpens.")
    parser.add_argument("--fine_tune", action="store_true",
                        help="Fine-tune an existing TCN on recent data instead of full retraining.")
    parser.add_argument("--recent_years", type=int, default=2,
                        help="Years of recent data used for --fine_tune (default 2).")
    parser.add_argument("--fast_wfo", action="store_true",
                        help="Use faster WFO window sizes for --mode optimize and --mode wfa.")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Top-N trials to aggregate for --mode extract_consensus (default 10).")
    parser.add_argument("--min_wfe", type=float, default=0.0,
                        help="Minimum WFE ratio filter for --mode extract_consensus (default 0).")
    parser.add_argument(
        "--stage", type=str, default=None, choices=["xgb", "trading"],
        help=(
            "Two-stage optimisation mode for --mode optimize.\n"
            "  xgb     : Stage-1 — fast single hold-out XGB exploration (no CPCV, ~5x faster).\n"
            "            Saves best params to models/stage1_{tf}_{broker}.json.\n"
            "  trading : Stage-2 — full CPCV optimization warm-started from Stage-1 params.\n"
            "If omitted, runs the standard joint optimization (same as before)."
        ),
    )

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
        "wfa":           cmd_wfa,
        "sensitivity":   cmd_sensitivity,
        "extract_consensus": cmd_extract_consensus,
    }[args.mode](args)


if __name__ == "__main__":
    main()
