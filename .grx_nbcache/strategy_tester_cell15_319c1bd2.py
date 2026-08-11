# ============================================================================
# Profit-Per-Trade Target Sizing  (position-size solver)
# ----------------------------------------------------------------------------
# WHY per-trade profit looks tiny:
#   The backtest trades a FIXED 0.01 micro-lot (POSITION_A = POSITION_B = 0.01),
#   pip value = 100 cents/lot, commission = 0, so
#       pnl_cents = move_pips * (lot * 100)
#   is PERFECTLY LINEAR in lot size. The ~12.3c (M15) / ~4.0c (M5) figures are
#   therefore per-0.01-lot (i.e. ~$0.12 / ~$0.04 per trade). max_drawdown is a
#   PERCENT of an equity curve seeded by a fixed 1500c ($15) cushion; because
#   accumulated profit dwarfs that cushion, %DD is nearly INVARIANT to lot size.
#   => per-trade profit is essentially a position-size dial you can turn while
#   %DD barely moves.
#
# This cell solves, per candidate, for the position size that reaches a TARGET
# per-trade profit while keeping max_drawdown <= a hard cap. It re-runs only a
# shortlist (no full-grid rerun) and recomputes drawdown at each lot multiplier
# EXACTLY from the trade PnL stream (valid because PnL is linear in lot).
# NOTE: this is POSITION SIZING (leverage), NOT new alpha -- profit factor and
# win rate are unchanged. Mind real-world margin/broker lot limits.
# ============================================================================

TARGET_PPT_USD   = 0.75      # desired profit per trade, in DOLLARS
MAX_DD_CAP_PCT   = 30.0      # hard drawdown cap (percent)
BASE_LOT         = POSITION_A  # current per-leg lot (0.01)
SHORTLIST_PER_TF = 60        # candidates re-run per timeframe (by ppt & robustness)
K_MAX            = 1000.0    # max lot multiplier searched (=> up to BASE_LOT * K_MAX)

TARGET_PPT_CENTS = TARGET_PPT_USD * 100.0


def _dd_pct_for_scale(pnl_cents, k, initial_balance=INITIAL_BALANCE_CENTS):
    """Rebuild equity at lot multiplier k and replicate compute_metrics' drawdown
    (rolling window = min(3, len) mean, peak-to-trough percent)."""
    cum = np.cumsum(pnl_cents) * float(k)
    eq = np.empty(len(cum) + 1, dtype=float)
    eq[0] = initial_balance
    eq[1:] = initial_balance + cum
    window = min(3, len(eq))
    avg_eq = pd.Series(eq).rolling(window=window, min_periods=1).mean().to_numpy()
    peaks = np.maximum.accumulate(avg_eq)
    dd = peaks - avg_eq
    return float(np.max(dd / np.maximum(peaks, 1e-9)) * 100.0) if len(dd) else 0.0


def _max_scale_within_dd(pnl_cents, cap_pct, k_hi=K_MAX):
    """Largest k in (0, k_hi] with max_drawdown <= cap_pct. dd% is monotonically
    increasing in k, so binary-search the crossing."""
    if _dd_pct_for_scale(pnl_cents, k_hi) <= cap_pct:
        return k_hi
    lo, hi = 0.0, k_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _dd_pct_for_scale(pnl_cents, mid) <= cap_pct:
            lo = mid
        else:
            hi = mid
    return lo


_base = globals().get("valid_results", None)
if _base is None or _base.empty:
    raise NameError("valid_results is not defined/empty. Run the backtest cells first.")

_cand_frames = []
for _tf in TIMEFRAMES:
    _sub = _base[_base["timeframe"] == _tf]
    _by_ppt = _sub.sort_values("profit_per_trade", ascending=False).head(SHORTLIST_PER_TF)
    _by_rob = _sub.sort_values("robust_score", ascending=False).head(SHORTLIST_PER_TF)
    _cand_frames.append(pd.concat([_by_ppt, _by_rob], ignore_index=True))

candidates = (
    pd.concat(_cand_frames, ignore_index=True)
    .drop_duplicates(subset=["timeframe", "strategy_name", "exit_model", "parameter_set"])
    .reset_index(drop=True)
)
print("Sizing shortlist: %d candidates across %d timeframes" % (len(candidates), len(TIMEFRAMES)))

_exit_keys = ["leg_a_atr_target", "time_stop_minutes", "trail_mult"]
rows = []
for _i, r in candidates.iterrows():
    tf = r["timeframe"]
    strat_name = r["strategy_name"]
    exit_model = r["exit_model"]
    params_all = json.loads(r["parameter_set"])
    entry_params = {k: params_all[k] for k in STRATEGIES[strat_name].param_cols if k in params_all}
    exit_params = {k: params_all[k] for k in _exit_keys if k in params_all}

    df = FEATURES_BY_TF[tf]
    signals = generate_routed_signals(df, entry_params, strat_name)
    trades_df, met = run_backtest(
        timeframe=tf, df=df, signals=signals,
        entry_params=entry_params, exit_model=exit_model, exit_params=exit_params,
    )
    if trades_df.empty:
        continue

    pnl = trades_df["pnl_cents"].to_numpy(dtype=float)
    n = int(len(pnl))
    base_ppt_cents = float(np.sum(pnl) / max(n, 1))   # per BASE_LOT
    if base_ppt_cents <= 0.0:
        continue

    dd_base = _dd_pct_for_scale(pnl, 1.0)             # ~ matches met["max_drawdown"]
    k_target = TARGET_PPT_CENTS / base_ppt_cents
    k_maxdd = _max_scale_within_dd(pnl, MAX_DD_CAP_PCT)
    feasible = bool(k_target <= k_maxdd)

    rows.append({
        "timeframe": tf, "strategy_name": strat_name, "exit_model": exit_model,
        "trade_count": n,
        "base_ppt_usd": base_ppt_cents / 100.0,
        "base_max_dd_pct": dd_base,
        "engine_max_dd_pct": float(met.get("max_drawdown", np.nan)),
        "lot_for_target": round(BASE_LOT * k_target, 4),
        "dd_at_target_pct": _dd_pct_for_scale(pnl, k_target),
        "meets_target_within_dd": feasible,
        "max_lot_within_dd": round(BASE_LOT * k_maxdd, 4),
        "max_ppt_usd_within_dd": (base_ppt_cents * k_maxdd) / 100.0,
        "profit_factor": float(r.get("profit_factor", np.nan)),
        "win_rate": float(r.get("win_rate", np.nan)),
        "robust_score": float(r.get("robust_score", np.nan)),
        "parameter_set": r["parameter_set"],
    })

sizing_results = pd.DataFrame(rows)
if sizing_results.empty:
    print("No positive-expectancy candidates to size.")
else:
    os.makedirs("reports", exist_ok=True)
    _sorted = sizing_results.sort_values(
        ["timeframe", "max_ppt_usd_within_dd"], ascending=[True, False]
    ).reset_index(drop=True)
    _sorted.to_csv("reports/profit_per_trade_sizing.csv", index=False)

    print("\nTarget: >= $%.2f / trade   |   Max DD cap: %.1f%%   |   base lot: %.2f\n" %
          (TARGET_PPT_USD, MAX_DD_CAP_PCT, BASE_LOT))

    print("=" * 78)
    print("  RECOMMENDED SIZING PER TIMEFRAME")
    print("=" * 78)
    for tf in TIMEFRAMES:
        sub = sizing_results[sizing_results["timeframe"] == tf]
        if sub.empty:
            print("\n[%s] no candidates." % tf)
            continue
        feas = sub[sub["meets_target_within_dd"]]
        if not feas.empty:
            pick = feas.sort_values("robust_score", ascending=False).iloc[0]
        else:
            pick = sub.sort_values("max_ppt_usd_within_dd", ascending=False).iloc[0]
        print("\n[%s] %s / %s" % (tf, pick["strategy_name"], pick["exit_model"]))
        print("   base profit/trade : $%.4f  at lot %.2f  (DD %.2f%%)" %
              (pick["base_ppt_usd"], BASE_LOT, pick["base_max_dd_pct"]))
        if pick["meets_target_within_dd"]:
            print("   -> set lot = %.4f  =>  profit/trade $%.2f  at DD %.2f%%  (<= %.0f%% cap)" %
                  (pick["lot_for_target"], TARGET_PPT_USD, pick["dd_at_target_pct"], MAX_DD_CAP_PCT))
        else:
            print("   -> target $%.2f NOT reachable within %.0f%% DD." %
                  (TARGET_PPT_USD, MAX_DD_CAP_PCT))
            print("      best within cap: lot %.4f => profit/trade $%.2f at DD ~%.1f%%" %
                  (pick["max_lot_within_dd"], pick["max_ppt_usd_within_dd"], MAX_DD_CAP_PCT))

    print("\nSaved: reports/profit_per_trade_sizing.csv")
    _show = ["timeframe", "strategy_name", "exit_model", "trade_count",
             "base_ppt_usd", "base_max_dd_pct", "lot_for_target", "dd_at_target_pct",
             "meets_target_within_dd", "max_lot_within_dd", "max_ppt_usd_within_dd",
             "profit_factor", "win_rate", "robust_score"]
    display(_sorted[_show])