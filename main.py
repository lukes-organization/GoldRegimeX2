#!/usr/bin/env python3
"""GoldRegimeX2 -- main command hub.

All terminal commands live here; main.py is the interconnector for the modules in
src/.  Everything is built ONLY from the pipeline_verification_bundle in the
review zip:
  * optimize / explore / backtest / montecarlo / certify run the bundle's OWN
    notebook code (Strategy_Tester.ipynb + GoldRegimeX_Explorer.ipynb) via
    src.notebook_runner, so the pipeline is the notebooks themselves.
  * update-data runs the notebooks' data-updater cell (src.data_updater).
  * live / demo auto-launch mt5_live_app.py -- the front end -- which forces the
    user to enter login / password / server and only trades AFTER MT5 has
    authenticated all three.

Interconnection (identical to the notebooks):
    optimize  -> writes reports/strategy_winners_for_explorer.csv
    explore   -> reads that CSV, trains, exports models/goldregimex_live_model.pkl
    backtest  -> Explorer Live-Trading-Simulation (mirror of live/demo)
    live/demo -> mt5_live_app trades 1:1 with that backtester

Run ``python main.py <command> --help`` for options.  PAPER-TRADE on a demo
login first.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP_PATH = REPO_ROOT / "mt5_live_app.py"


def _cmd_update_data(args):
    from src import data_updater
    data_updater.update_all_data()


def _cmd_optimize(args):
    from src import strategy_tester
    strategy_tester.run_optimization(update_data=not args.no_update)


def _cmd_explore(args):
    from src import explorer
    explorer.train_and_export(update_data=not args.no_update)


def _cmd_backtest(args):
    from src import backtester
    backtester.run_backtest(update_data=not args.no_update)


def _cmd_montecarlo(args):
    from src import explorer
    explorer.run_monte_carlo(update_data=not args.no_update)


def _cmd_certify(args):
    from src import explorer
    explorer.run_certify(update_data=not args.no_update)


def _cmd_verify(args):
    import runpy
    script = REPO_ROOT / "pipeline_verification_bundle" / "pipeline_verification.py"
    if not script.exists():
        print("pipeline_verification.py not found in the bundle.")
        return
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")


def _launch_app(mode):
    """Auto-open the MT5 login/trading window (the front end).  The window itself
    enforces login/password/server auth before any live/demo trading."""
    if not APP_PATH.exists():
        print("mt5_live_app.py not found at %s" % APP_PATH)
        return
    print("Launching MT5 %s trading window -- enter login / password / server to "
          "authenticate. Trading is enabled only after MT5 accepts all three." % mode)
    subprocess.run([sys.executable, str(APP_PATH)], cwd=str(REPO_ROOT))


def _cmd_live(args):
    _launch_app("LIVE/REAL")


def _cmd_demo(args):
    _launch_app("DEMO")


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description="GoldRegimeX2 command hub (interconnector for src/).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_text, with_update=True):
        sp = sub.add_parser(name, help=help_text)
        if with_update:
            sp.add_argument("--no-update", action="store_true",
                            help="skip the MT5 data refresh; use existing CSVs")
        sp.set_defaults(func=fn)
        return sp

    add("update-data", _cmd_update_data,
        "Refresh the CSVs from MT5 (notebook data-updater).", with_update=False)
    add("optimize", _cmd_optimize,
        "Strategy Tester: grid-search plateau -> winners CSV handoff.")
    add("explore", _cmd_explore,
        "Explorer: read winners, train HMM+XGB, export live model bundle.")
    add("backtest", _cmd_backtest,
        "Explorer live-trading simulation (mirror of the live/demo engine).")
    add("montecarlo", _cmd_montecarlo,
        "Explorer Monte Carlo robustness verdict.")
    add("certify", _cmd_certify,
        "Explorer MT5 parity / engine self-check.")
    add("verify", _cmd_verify,
        "Run the bundle's pipeline_verification.py.", with_update=False)
    add("live", _cmd_live,
        "Open the MT5 window for LIVE/REAL trading (auth required).", with_update=False)
    add("demo", _cmd_demo,
        "Open the MT5 window for DEMO trading (auth required).", with_update=False)

    # aliases
    for alias, target in (("train", _cmd_explore), ("simulate", _cmd_backtest)):
        sp = sub.add_parser(alias, help="alias")
        sp.add_argument("--no-update", action="store_true")
        sp.set_defaults(func=target)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
