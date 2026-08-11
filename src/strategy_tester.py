"""src/strategy_tester.py -- Strategy-Tester pipeline (grid-search plateau).

Runs Strategy_Tester.ipynb as-is: its auto data-updater cell refreshes the CSVs
at the top of every run (AUTO_UPDATE_DATA=True) -- so the data update fires
whenever the grid-search plateau runs -- then the notebook optimises and writes
reports/strategy_winners_for_explorer.csv, the handoff the Explorer consumes.
Pass update_data=False to optimise on the existing CSVs.
"""
from __future__ import annotations
from . import notebook_runner as nr

WINNERS_CSV = nr.REPO_ROOT / "reports" / "strategy_winners_for_explorer.csv"


def run_optimization(update_data=True):
    overrides = None if update_data else {"AUTO_UPDATE_DATA": False}
    ns = nr.run_notebook("strategy_tester", overrides=overrides)
    print("\n[strategy_tester] winners handoff -> %s" % WINNERS_CSV)
    return {"winners_csv": str(WINNERS_CSV), "namespace": ns}
