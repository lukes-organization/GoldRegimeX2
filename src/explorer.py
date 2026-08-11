"""src/explorer.py -- Explorer pipeline (HMM+XGB research / training / export /
live-trading simulation / parity / Monte Carlo).

Consumes reports/strategy_winners_for_explorer.csv produced by the Strategy
Tester (notebook cell 5) -- the notebook-to-notebook interconnection is kept
exactly.  The notebook is sequential, so a prefix run is always valid:

    train_and_export -> up to the model-export cell
    run_backtest     -> up to the Live-Trading-Simulation cell (backtester mirror)
    run_certify      -> up to the MT5 parity / self-check cell
    run_monte_carlo  -> through the Monte Carlo robustness verdict
    run_full         -> the entire notebook

The auto data-updater cell fires at the top of every run; pass update_data=False
to skip it.
"""
from __future__ import annotations
from . import notebook_runner as nr


def _run(stop_after=None, update_data=True):
    overrides = None if update_data else {"AUTO_UPDATE_DATA": False}
    return nr.run_notebook("explorer", overrides=overrides, stop_after=stop_after)


def train_and_export(update_data=True):
    return _run(stop_after=nr.EXPLORER_EXPORT_CELL, update_data=update_data)


def run_backtest(update_data=True):
    return _run(stop_after=nr.EXPLORER_LIVESIM_CELL, update_data=update_data)


def run_certify(update_data=True):
    return _run(stop_after=nr.EXPLORER_PARITY_CELL, update_data=update_data)


def run_monte_carlo(update_data=True):
    return _run(stop_after=nr.EXPLORER_MONTECARLO_CELL, update_data=update_data)


def run_full(update_data=True):
    return _run(stop_after=None, update_data=update_data)
