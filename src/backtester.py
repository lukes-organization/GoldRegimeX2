"""src/backtester.py -- named entry for THE backtester.

The backtester is the Explorer notebook's Live-Trading-Simulation cell: the exact
engine the live/demo platform mirrors.  A single named entry makes the 'mirror'
relationship explicit -- 'backtest' here and mt5_trader's live loop both derive
their entries/exits from this simulation and the same exported model bundle.
"""
from __future__ import annotations
from . import explorer


def run_backtest(update_data=True):
    "Run the live-trading simulation (mirror of the live/demo engine)."
    return explorer.run_backtest(update_data=update_data)
