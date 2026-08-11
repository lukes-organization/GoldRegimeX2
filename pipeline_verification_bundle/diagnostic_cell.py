# DIAGNOSTIC: run this after Cell 5 (signal generation) and before Cell 9 (execute research).
# Shows raw signal counts and trade counts for each QUICK_MODE parameter set,
# so you can see why qualify() rejects everything instead of guessing.

if 'FEATURES' not in globals():
    raise NameError("FEATURES is not defined. Run Cell 5 (signal generation) first.")

for tf in ('M15', 'M5'):
    splits = chronological_split(FEATURES[tf])
    dev = splits['development']
    print(f'--- {tf} development slice: {len(dev)} bars, {dev.index.min()} to {dev.index.max()} ---')
    for params in parameter_grid(tf):
        sig = generate_signals(dev, tf, params)
        n_long = int((sig == 1).sum())
        n_short = int((sig == -1).sum())
        trades, equity, events = run_backtest(dev, sig, params)
        m = compute_metrics(trades, equity)
        print(
            f'{tf} | adx>={params["adx_threshold"]} rsi_p={params["rsi_period"]} '
            f'pullback={params["pullback_rsi"]} | raw signals L{n_long}/S{n_short} '
            f'| trades={m["trade_count"]} | pf={m["profit_factor"]:.2f} '
            f'| dd={m["max_equity_drawdown_pct"]:.1f}% | expectancy=${m["expectancy_usd"]:.4f}'
        )
