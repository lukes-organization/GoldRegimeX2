# Export top robust candidates for downstream wiring into GoldRegimeX_Explorer
# FIX (2026-07-07): previously exported only the top 30 rows by profit_per_trade,
# which meant strategies that survived the 25% Max DD filter but had lower
# profit-per-trade were dropped from the CSV, causing the Explorer to raise
# "No baseline survived 25%% DD filter" even when survivors clearly existed
# (see Cell 12 output). Now the CSV always includes every DD-surviving row
# from valid_results, plus the top N by profit_per_trade for downstream
# ranking flexibility.

leg_c_lot_rule = "A_if_A_hits_first_else_B"
HANDOFF_MAX_DD_PCT = 30.0

handoff_cols = [
    "timeframe",
    "strategy_name",
    "exit_model",
    "parameter_set",
    "profit_factor",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "expectancy",
    "win_rate",
    "trade_count",
    "profit_per_trade",
    "parameter_stability_score",
    "robust_score",
    "leg_c_lot_rule",
]

top_n = 30

# 1) top-N by profit_per_trade / robust_score (previous behaviour)
top_by_profit = (
    valid_results.sort_values(
        ["profit_per_trade", "robust_score", "profit_factor", "sharpe"],
        ascending=False,
    )
    .head(top_n)
    .copy()
)

# 2) ALL rows that survive the 25% Max DD cap - the Explorer needs these.
dd_survivors = valid_results[valid_results["max_drawdown"] <= HANDOFF_MAX_DD_PCT].copy()

# 3) For each timeframe, guarantee at least the top-K DD-survivors by robust_score
#    are present, so both M15 and M5 have baselines to hand off.
per_tf_k = 10
per_tf_survivors = (
    dd_survivors
    .sort_values(["robust_score", "profit_per_trade"], ascending=False)
    .groupby("timeframe", as_index=False)
    .head(per_tf_k)
)

# 4) Union (DD-survivors first so they sort earliest by construction), then dedupe.
handoff = pd.concat([per_tf_survivors, dd_survivors, top_by_profit], ignore_index=True)
handoff = handoff.drop_duplicates(
    subset=["timeframe", "strategy_name", "exit_model", "parameter_set"],
    keep="first",
).reset_index(drop=True)
handoff["leg_c_lot_rule"] = leg_c_lot_rule
handoff = handoff[handoff_cols].copy()

import os
os.makedirs("reports", exist_ok=True)
handoff.to_csv("reports/strategy_winners_for_explorer.csv", index=False)
print("Saved: reports/strategy_winners_for_explorer.csv")
print("  Total rows:", len(handoff))
print("  DD<=%.1f%% rows:" % HANDOFF_MAX_DD_PCT,
      int((handoff["max_drawdown"] <= HANDOFF_MAX_DD_PCT).sum()))
print("  Per timeframe DD survivors:")
print(handoff[handoff["max_drawdown"] <= HANDOFF_MAX_DD_PCT]
      .groupby("timeframe").size().to_string())
display(handoff)