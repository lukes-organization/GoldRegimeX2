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
from src.processor import process_pipeline, TF_CONFIG, PROCESSED_PATH, save_feature_scaler, load_feature_scaler
from src.engine_hmm import fit_hmm, save_model as save_hmm, load_model as load_hmm, get_model_path as hmm_model_path
from src.engine_xgb import (
    prepare_features, train_xgb, get_predictions,
    export_onnx, save_xgb, load_xgb, ONNX_PATH, FEATURE_COLS,
    train_xgb_ensemble, get_predictions_ensemble,
    save_xgb_ensemble, load_xgb_ensemble, export_onnx_ensemble, ENSEMBLE_PKL_PATH,
    get_ensemble_path, TF_TRAIN_RATIO, compute_regime_stats,
)
from src.optimizer import run_optimization, get_best_params, _score_result as _calc_score
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
    _train_ratio = TF_TRAIN_RATIO.get(tf.upper(), 0.70)
    models_ensemble, thresholds, metrics = train_xgb_ensemble(
        X, y,
        train_ratio=_train_ratio,
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
    # Compute IS regime statistics for Z-Score signal calibration
    _X_is = X.iloc[:split_idx] if split_idx else X
    _hs_is = states_aligned[:len(_X_is)]
    metrics["regime_stats"] = compute_regime_stats(models_ensemble, thresholds, _X_is, _hs_is)
    result = vectorized_backtest(
        df_aligned, probabilities, states_aligned,
        split_idx=split_idx,
        account_size=balance,
        broker=broker,
        tf=tf,
        regime_stats=metrics["regime_stats"],
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
    """Walk-Forward Analysis: evaluate model consistency across rolling time windows.

    Uses the globally-trained model's probability outputs (no per-window retraining)
    to score IS + OOS Sharpe across every train_days/test_days window.  Prints a
    per-window breakdown and the aggregate Walk-Forward Efficiency (WFE) ratio.

    WFE target: > 50%.  A WFE of 60% means the model's OOS performance is 60% of
    its IS performance — i.e. it generalises well rather than curve-fitting a few
    favourable years.
    """
    from src.backtester import run_walk_forward
    from src.notifier import send_telegram_msg

    balance = _resolve_balance(args)
    broker  = args.broker
    tf      = args.tf.upper()

    # TF-specific defaults (calendar days) — tuned to each bar frequency
    _wfa_defaults = {"H1": (365, 90), "M15": (180, 60), "M5": (90, 30)}
    default_train, default_test = _wfa_defaults.get(tf, (365, 90))
    train_days = getattr(args, "train_days", None) or default_train
    test_days  = getattr(args, "test_days",  None) or default_test

    try:
        params = get_best_params(balance=balance, broker=broker, tf=tf)
        logger.info("WFA using Optuna params [%s/%s]: %s", tf, broker, params)
    except Exception:
        logger.warning("No Optuna study found for %s/%s — using defaults.", tf, broker)
        params = {}

    logger.info(
        "Walk-Forward Analysis [%s/%s] balance=$%.0f  train=%dd  test=%dd",
        tf, broker, balance, train_days, test_days,
    )
    print(f"\n=== Walk-Forward Analysis [{tf} / {broker}] ===")
    print(f"  Train window : {train_days} days  |  Test step : {test_days} days")
    print("  Loading full dataset...")

    df = process_pipeline(
        obs_cov=params.get("obs_cov"),
        trans_cov=params.get("trans_cov"),
        save=False,
        tf=tf,
    )

    n_states = params.get("n_states", TF_CONFIG[tf].get("n_states_default", 3))
    model_hmm, states, _ = fit_hmm(df, n_states=n_states)

    _min_persist = min(model_hmm.transmat_[i, i] for i in range(model_hmm.n_components))
    if _min_persist < 0.65:
        print(
            f"\nERROR: Degenerate HMM (min persistence={_min_persist:.4f}). "
            "Run --mode optimize first to find stable Kalman parameters."
        )
        sys.exit(1)

    X, y, df_aligned, _scaler = prepare_features(df, states, tf=tf)
    _train_ratio = TF_TRAIN_RATIO.get(tf.upper(), 0.70)
    models_e, thresholds_e, metrics_e = train_xgb_ensemble(
        X, y,
        train_ratio      = _train_ratio,
        max_depth        = params.get("max_depth", 4),
        learning_rate    = params.get("learning_rate", 0.1),
        n_estimators     = params.get("n_estimators", 200),
        subsample        = params.get("subsample", 0.8),
        colsample_bytree = params.get("colsample_bytree", 0.8),
        min_child_weight = params.get("min_child_weight", 5),
        gamma            = params.get("gamma", 1.0),
        reg_alpha        = params.get("reg_alpha", 0.1),
    )
    _, probs    = get_predictions_ensemble(models_e, thresholds_e, X)
    states_aln  = states[df.index.isin(df_aligned.index)]
    _cmp_split  = metrics_e.get("split_idx") or int(len(X) * 0.70)
    _X_is_cmp   = X.iloc[:_cmp_split]
    _hs_is_cmp  = states_aln[:len(_X_is_cmp)]
    _regime_stats_cmp = compute_regime_stats(models_e, thresholds_e, _X_is_cmp, _hs_is_cmp)

    print(
        f"  Dataset: {len(df_aligned)} bars  "
        f"({df_aligned.index[0].date()} – {df_aligned.index[-1].date()})"
    )
    print("  Evaluating rolling windows (fixed model — no per-window retraining)...")

    wfa = run_walk_forward(
        df_aligned, probs, states_aln,
        train_days    = train_days,
        test_days     = test_days,
        account_size  = balance,
        broker        = broker,
        tf            = tf,
        regime_stats  = _regime_stats_cmp,
    )

    n_win       = wfa["n_windows"]
    n_valid     = wfa.get("n_valid_windows", n_win)
    wfe         = wfa["wfe_ratio"]
    mean_is     = wfa["mean_is_sharpe"]
    mean_oos    = wfa["mean_oos_sharpe"]
    verdict     = "ROBUST ✅" if wfe >= 0.50 else "FRAGILE ⚠️ — consider re-optimising"
    n_silent    = sum(1 for w in wfa["windows"] if w.get("oos_n_trades", 0) == 0)

    print(f"\n  Total windows     : {n_win}")
    print(f"  Valid windows     : {n_valid}  (≥5 IS trades, ≥1 OOS trade — used for WFE)")
    print(f"  Silent windows    : {n_silent}  (0 OOS trades — model held back by efficiency filter)")
    print(f"  Mean IS  Sharpe   : {mean_is:+.3f}  [valid only]")
    print(f"  Mean OOS Sharpe   : {mean_oos:+.3f}  [valid only]")
    print(f"  Walk-Forward Eff  : {wfe * 100:.1f}%  [{verdict}]")

    if n_win > 0:
        print("\n  Per-window OOS breakdown:")
        WFA_FOLD_PASS_SCORE = 1.0
        for w in wfa["windows"]:
            oos_s  = w.get("oos_sharpe_ratio", 0.0)
            oos_t  = w.get("oos_n_trades",    0)
            period = (
                f"{w['oos_start'].strftime('%Y-%m')} → "
                f"{w['oos_end'].strftime('%Y-%m')}"
            )
            if oos_t == 0:
                flag = "—"          # model was silent; not a warning
            elif oos_t < 3:
                flag = "⚠️ (thin)"  # signal fired but sample too small to trust
            else:
                oos_fdd = w.get("oos_floating_max_drawdown", w.get("oos_max_drawdown", 0.0))
                fold_r  = {
                    "total_return":          w.get("oos_total_return", 0.0),
                    "floating_max_drawdown": oos_fdd,
                    "sharpe_ratio":          oos_s,
                    "profit_factor":         w.get("oos_profit_factor", 1.0),
                    "return_consistency":    w.get("oos_return_consistency", 0.0),
                }
                fold_score = _calc_score(fold_r, tf=tf)
                if fold_score >= WFA_FOLD_PASS_SCORE:
                    flag = f"✅ ({fold_score:.2f})"
                elif fold_score >= 0:
                    flag = f"⚠️ ({fold_score:.2f})"
                else:
                    flag = f"❌ ({fold_score:.2f})"
                if w.get("oos_profit_factor", 1.0) < 1.2:
                    flag += " ⚠️PF<1.2"
            oos_pf  = w.get("oos_profit_factor", 1.0)
            oos_eff = w.get("oos_avg_efficiency", 0.0)
            eff_str = f"{oos_eff:.2f}x" if oos_t >= 3 else "  —  "
            print(f"    {period}  OOS={oos_s:+.3f}  PF={oos_pf:.2f}  Eff={eff_str}  trades={oos_t}  {flag}")

    send_telegram_msg(
        f"📊 <b>Walk-Forward Analysis [{tf}]</b>\n"
        f"Valid windows: <b>{n_valid}/{n_win}</b>  |  "
        f"IS: <b>{mean_is:+.3f}</b>  |  OOS: <b>{mean_oos:+.3f}</b>\n"
        f"WFE: <b>{wfe * 100:.1f}%</b>  "
        + ("✅ Robust" if wfe >= 0.50 else "⚠️ Fragile — re-optimise recommended")
    )


def cmd_process(args):
    tfs = [t.strip().upper() for t in args.tf.split(",")]
    for tf in tfs:
        if tf not in TF_CONFIG:
            logger.error("Unknown timeframe '%s'. Valid: %s", tf, list(TF_CONFIG))
            continue
        try:
            df = process_pipeline(save=True, tf=tf, save_models=True, broker=args.broker)
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
        print(f"\n=== Best Result [{tf}] ===")
        print(f"Score:         {study.best_value:.3f}")
        print(f"Broker:        {broker}")
        print(f"Balance:       ${balance:.0f} USD")
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
    print(f"\n=== Training Results [{tf}] ===")
    print(f"Broker: {broker} | Balance: ${balance:.0f} | Tier: {'small' if arm.is_small_account else 'growth'}")
    print(f"Sharpe: {result['sharpe_ratio']:.3f} | MaxDD: {result['max_drawdown']*100:.1f}% "
          f"| WR: {result['win_rate']*100:.1f}% | Trades: {result['n_trades']}")
    if "oos_sharpe_ratio" in result:
        def _fmt_block(label, r, prefix):
            fdd    = r.get(f"{prefix}floating_max_drawdown", r.get(f"{prefix}max_drawdown", 0.0))
            r_dict = {
                "total_return":          r.get(f"{prefix}total_return", 0.0),
                "floating_max_drawdown": fdd,
                "sharpe_ratio":          r.get(f"{prefix}sharpe_ratio", 0.0),
                "profit_factor":         r.get(f"{prefix}profit_factor", 1.0),
            }
            score   = _calc_score(r_dict)
            rf      = r.get(f"{prefix}recovery_factor", 0.0)
            pf      = r.get(f"{prefix}profit_factor", 1.0)
            payoff  = r.get(f"{prefix}expected_payoff", 0.0) * balance
            eff     = r.get(f"{prefix}avg_efficiency", 0.0)
            cost_e  = r.get(f"{prefix}cost_efficiency", 0.0)
            payout_str = format_payout(r.get(f"{prefix}total_return", 0.0), balance, broker)
            print(
                f"  [{tf} {label}] Score: {score:.2f} | RF: {rf:.2f} | PF: {pf:.2f}"
                f" | Payoff: ${payoff:.4f} | MaxDD: {fdd*100:.1f}% (Floating)"
                f" | WR: {r.get(f'{prefix}win_rate', 0.0)*100:.1f}% | Trades: {r.get(f'{prefix}n_trades', 0)}"
            )
            print(
                f"  [{tf} {label}] Efficiency: {eff:.2f}x ATR/Spread"
                f" | CostEff: {cost_e*100:.1f}%"
                f" | Total Payout: {payout_str}"
            )
            if cost_e < 0.50:
                print(f"  ⚠️  WARNING: Broker is consuming >{(1-cost_e)*100:.0f}% of gross profit via spread/commission.")
        print(f"\n--- In-Sample ---")
        _fmt_block("IS",  result, "is_")
        print(f"--- Out-of-Sample ---")
        _fmt_block("OOS", result, "oos_")
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
        print("  ⚠️  WARNING: Low Market Efficiency — Spread is eating your edge.")
    if _cost_e < 0.50:
        print(f"  ⚠️  WARNING: Broker is consuming >{(1-_cost_e)*100:.0f}% of gross profit.")
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

    # split_idx in saved metrics may come from a different TF's training run
    # (e.g. M15 split_idx ~186k used in an H1 report with only ~58k bars).
    # Recompute from current data when the stored value doesn't fit.
    split_idx = metrics.get("split_idx")
    if split_idx is None or not (0 < split_idx < len(X)):
        split_idx = int(len(X) * 0.8)

    result = vectorized_backtest(
        df_aligned, probabilities, states_aligned,
        split_idx=split_idx, account_size=balance, broker=broker, tf=tf,
        regime_stats=metrics.get("regime_stats"),
    )

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
    """Consolidate USDCHF CSV exports in data/raw/ into per-TF master files.

    Produces:
      data/processed/USDCHF_master.csv      (H1  — source: USDCHF_H1.csv)
      data/processed/USDCHF_master_M15.csv  (M15 — source: USDCHF_M15_*.csv)
      data/processed/USDCHF_master_M5.csv   (M5  — source: USDCHF_M5_*.csv)
    """
    from src.data_consolidator import (
        consolidate_usdchf,
        consolidate_usdchf_m15,
        consolidate_usdchf_m5,
    )
    for fn, label, out in [
        (consolidate_usdchf,     "H1",  "data/processed/USDCHF_master.csv"),
        (consolidate_usdchf_m15, "M15", "data/processed/USDCHF_master_M15.csv"),
        (consolidate_usdchf_m5,  "M5",  "data/processed/USDCHF_master_M5.csv"),
    ]:
        result = fn()
        if not result.empty:
            print(f"  USDCHF {label}: {len(result)} rows → {out}")
        else:
            logger.warning("USDCHF %s consolidation produced no data — skipping.", label)


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
                 "sync_validate", "demo", "live", "audit", "guardian", "listen",
                 "consolidate", "wfa"],
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
    }[args.mode](args)


if __name__ == "__main__":
    main()
