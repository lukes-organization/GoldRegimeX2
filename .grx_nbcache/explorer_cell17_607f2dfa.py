# -----------------------------
# Model lifecycle + Final model selection + IS/OOS validation
# Phase 2-3: Separate train_ml_model from evaluate_ml_model
# Phase 4:   No global best timeframe -- both TFs export independently
# Phase 6-7: Pipeline-scoped registry instead of global variables
# Phase 8:   Validation loop uses pre-trained models (no retraining)
# Phase 5:   Consumes split from pipeline container (split_dataset)
# -----------------------------
import warnings

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default

def _normalize_metrics(met, ending_balance):
    out = dict(met)
    out["max_drawdown_pct"] = _safe_float(met.get("max_drawdown", met.get("max_drawdown_pct", 0.0)), 0.0)
    out["n_trades"] = _safe_int(met.get("trade_count", met.get("n_trades", 0)), 0)
    out["ending_balance_cents"] = _safe_float(ending_balance, 0.0)
    return out


# -----------------------------
# Phase 2-3: train_ml_model and evaluate_ml_model
# -----------------------------
def train_ml_model(train_data, exec_tf):
    "Train an HMMXGBComposite model on training data ONCE."
    warnings.filterwarnings("ignore")
    tf = exec_tf.upper()
    if ML_TARGET_PARAMS.get(tf) is None:
        raise RuntimeError("No bridge baseline for %s" % tf)

    feat = build_features(train_data, tf)
    hmm_feats = hmm_feature_columns(feat)

    model = HMMXGBComposite(random_state=RANDOM_STATE)
    model.fit(feat, feat["tb_label"], hmm_features=hmm_feats)
    print("  [%s] Model trained on %d samples, %d HMM features" % (tf, len(feat), len(hmm_feats)))
    return model, feat


def evaluate_ml_model(trained_model, data, exec_tf, xgb_threshold):
    "Evaluate a PRE-TRAINED model on provided data. NO training occurs here."
    warnings.filterwarnings("ignore")
    tf = exec_tf.upper()
    if ML_TARGET_PARAMS.get(tf) is None:
        raise RuntimeError("No bridge baseline for %s" % tf)

    base_params = dict(ML_TARGET_PARAMS[tf]["parameters"])
    base_params["exit_model"] = ML_TARGET_PARAMS[tf]["exit_model"]

    feat = build_features(data, tf)

    # Inference only -- no model.fit() call
    probs = trained_model.predict_proba_raw(feat)

    trades, met = run_ml_filtered_backtest(tf, feat, probs, base_params, float(xgb_threshold))
    ending_balance = _safe_float(INITIAL_BALANCE_CENTS + _safe_float(met.get("net_profit", 0.0)), INITIAL_BALANCE_CENTS)
    return trades, _normalize_metrics(met, ending_balance)


def _select_center_with_fallback(tf_fine):
    "Plateau-center selection with the original fallback logic, per TF."
    center = select_plateau_center(tf_fine, min_mean_sharpe=-1e9)
    if center is not None:
        return center

    print("Standard center selection failed. Attempting fallback...")
    d = tf_fine.copy()
    for col in ["xgb_threshold", "mean_sharpe", "stability_adjusted_sharpe"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d[np.isfinite(d["mean_sharpe"]) & np.isfinite(d["stability_adjusted_sharpe"])]
    if d.empty:
        raise RuntimeError("No valid fine results found after filtering.")

    row = d.sort_values(
        ["stability_adjusted_sharpe", "mean_sharpe"],
        ascending=False,
        na_position="last",
    ).iloc[0]

    center = {
        "xgb_threshold": float(row["xgb_threshold"]),
        "mean_sharpe": float(row["mean_sharpe"]),
        "stability_adjusted_sharpe": float(row["stability_adjusted_sharpe"]),
        "selection_score": float(row["stability_adjusted_sharpe"]),
    }
    print("Fallback center selected:")
    print(center)
    return center


# ---------------------------------------------------------------
# 1. Select the best plateau center PER timeframe (Phase 4)
#    NO global "best timeframe" selection.
#    Each TF stores its own threshold and plateau in the pipeline.
# ---------------------------------------------------------------
for tf in TIMEFRAMES:
    tf_fine = fine_results[fine_results["timeframe"] == tf].copy()
    if tf_fine.empty:
        raise RuntimeError("No fine results for timeframe %s." % tf)

    print("\n[%s] Selected plateau center:" % tf)
    center = _select_center_with_fallback(tf_fine)
    print(center)

    thr = _safe_float(center.get("xgb_threshold"), np.nan)
    if np.isnan(thr):
        raise RuntimeError("Invalid xgb_threshold selected for %s: %r" % (tf, center.get('xgb_threshold')))

    pipeline[tf].plateau = center
    pipeline[tf].threshold = thr
    print("[%s] Threshold stored in pipeline: %s" % (tf, thr))

# ---------------------------------------------------------------
# 2. Train models PER timeframe (Phase 2-3: train once)
#    Models are stored in pipeline[tf].model.
#    They will NOT be retrained during evaluation.
# ---------------------------------------------------------------
print("\nTraining models (one per timeframe, frozen after this)...")
for tf in TIMEFRAMES:
    p = pipeline[tf]
    print("\n[%s] Training on IS data (%d rows)..." % (tf, len(p.train_df)))
    model, feat = train_ml_model(p.train_df, tf)
    p.model = model
    p.train_feat = feat
    print("[%s] Model frozen and stored in pipeline" % tf)

# ---------------------------------------------------------------
# 3. Evaluate models: IS and OOS using pre-trained models (Phase 8)
#    Training must never occur here.
# ---------------------------------------------------------------
print("\nRunning final validation segments for both timeframes (inference only)...")

validation_rows = []

for tf in TIMEFRAMES:
    p = pipeline[tf]
    thr = float(p.threshold)

    print("\n[%s] threshold=%s  (train rows=%d, oos rows=%d)" % (tf, thr, len(p.train_df), len(p.oos_df)))

    # IS evaluation -- uses pre-trained model
    trades_is, met_is = evaluate_ml_model(p.model, p.train_df, tf, thr)
    p.trades_is = trades_is
    p.metrics_is = met_is

    # OOS evaluation -- uses SAME pre-trained model (no retraining!)
    trades_oos, met_oos = evaluate_ml_model(p.model, p.oos_df, tf, thr)
    p.trades_oos = trades_oos
    p.metrics_oos = met_oos

    score_is = score_metrics(met_is)
    score_oos = score_metrics(met_oos)
    rel_drop = (score_is - score_oos) / (abs(score_is) + 1e-9)

    validation_rows.extend([
        {
            "timeframe": tf,
            "scenario": "IS",
            "xgb_threshold": thr,
            "score": score_is,
            "win_rate": met_is.get("win_rate", 0),
            "profit_factor": met_is.get("profit_factor", 0),
            "max_drawdown_pct": met_is.get("max_drawdown_pct", 0),
            "n_trades": met_is.get("n_trades", 0),
            "ending_balance_cents": met_is.get("ending_balance_cents", 0),
            "net_return_pct": met_is.get("net_return_pct", 0),
            "net_profit_cents": met_is.get("net_profit", 0),
            "profit_per_trade_cents": met_is.get("profit_per_trade", 0),
            "profit_per_trade_usd": met_is.get("profit_per_trade", 0) / 100.0,
            "relative_score_drop_is_to_oos": rel_drop,
        },
        {
            "timeframe": tf,
            "scenario": "OOS",
            "xgb_threshold": thr,
            "score": score_oos,
            "win_rate": met_oos.get("win_rate", 0),
            "profit_factor": met_oos.get("profit_factor", 0),
            "max_drawdown_pct": met_oos.get("max_drawdown_pct", 0),
            "n_trades": met_oos.get("n_trades", 0),
            "ending_balance_cents": met_oos.get("ending_balance_cents", 0),
            "net_return_pct": met_oos.get("net_return_pct", 0),
            "net_profit_cents": met_oos.get("net_profit", 0),
            "profit_per_trade_cents": met_oos.get("profit_per_trade", 0),
            "profit_per_trade_usd": met_oos.get("profit_per_trade", 0) / 100.0,
            "relative_score_drop_is_to_oos": rel_drop,
        },
    ])

# 4. Output summary
validation_summary = pd.DataFrame(validation_rows)

print("\nValidation summary (both timeframes):")
display(validation_summary)

print("\nRelative score drop IS->OOS:")
for tf in TIMEFRAMES:
    row = validation_summary[
        (validation_summary["timeframe"] == tf) & (validation_summary["scenario"] == "OOS")
    ].iloc[0]
    drop = float(row["relative_score_drop_is_to_oos"])
    warn = "  <- WARNING: >30% drop" if drop > 0.30 else ""
    print("  %s: %.2f%%%s" % (tf, drop * 100, warn))

if not validation_summary.empty:
    validation_summary.to_csv("notebooks/reports/final_is_oos_summary.csv", index=False)
    print("\nSaved: notebooks/reports/final_is_oos_summary.csv")