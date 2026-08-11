# Regime Performance Attribution Analytics Report

def generate_regime_report(trades_df: pd.DataFrame):
    if trades_df.empty:
        print("Regime Report Error: Zero transaction logs generated.")
        return

    regimes = ["TREND", "SHOCK", "MR"]
    print("=" * 50)
    print("      REGIME RESEARCH ATTRIBUTION SUMMARY      ")
    print("=" * 50)

    total_pnl = trades_df["pnl_cents"].sum()

    for regime in regimes:
        r_trades = trades_df[trades_df["entry_regime"] == regime]

        if r_trades.empty:
            print(f"\nREGIME: {regime}\n" + "-" * 20)
            print("Buys: 0 | Sells: 0\nNet Profit: 0.00 Cents\nProfit Factor: N/A\nPnL Contribution: 0.0%")
            continue

        buys = len(r_trades[r_trades["side"] == "BUY"])
        sells = len(r_trades[r_trades["side"] == "SELL"])

        r_pnl = r_trades["pnl_cents"].to_numpy()
        gross_profit = np.sum(r_pnl[r_pnl > 0])
        gross_loss = -np.sum(r_pnl[r_pnl < 0])

        pf = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)
        net_profit = np.sum(r_pnl)
        contribution = (net_profit / total_pnl * 100.0) if total_pnl != 0 else 0.0

        print(f"\nREGIME: {regime}\n" + "-" * 20)
        print(f"Buys: {buys} | Sells: {sells}")
        print(f"Net Profit: {net_profit:.2f} Cents")
        print(f"Profit Factor: {pf:.2f}")
        print(f"PnL Contribution: {contribution:.1f}%")

    print("=" * 50)


# Run a regime report on the top valid run for EACH Timeframe and EACH Strategy
if valid_results.empty:
    print("No valid_results available (trade_count >= 200). Run the backtest cell first.")
else:
    for tf in TIMEFRAMES:
        for strat_name in STRATEGIES.keys():
            # Filter for specific timeframe and strategy
            strat_results = valid_results[
                (valid_results["strategy_name"] == strat_name)
                & (valid_results["timeframe"] == tf)
            ]

            if strat_results.empty:
                print(f"\nNo valid runs found for {tf} | {strat_name}.")
                continue

            best_strat_row = strat_results.sort_values(
                ["robust_score", "profit_per_trade", "profit_factor", "sharpe"],
                ascending=False,
            ).iloc[0]

            exit_model = best_strat_row["exit_model"]

            combined_params = json.loads(best_strat_row["parameter_set"])
            entry_params = {
                k: combined_params[k]
                for k in STRATEGIES[strat_name].param_cols
                if k in combined_params
            }

            exit_keys = ["leg_a_atr_target", "time_stop_minutes", "trail_mult"]
            exit_params = {k: combined_params[k] for k in exit_keys if k in combined_params}

            df = FEATURES_BY_TF[tf]
            signals = generate_routed_signals(df, entry_params, strat_name)
            
            # FIX: Passed timeframe down correctly here
            trades_df, _ = run_backtest(
                timeframe=tf,
                df=df,
                signals=signals,
                entry_params=entry_params,
                exit_model=exit_model,
                exit_params=exit_params,
            )

            print(f"\n{'=' * 50}")
            print(f"  BEST RUN: {tf} | {strat_name.upper()} | {exit_model}")
            print(f"{'=' * 50}")
            generate_regime_report(trades_df)