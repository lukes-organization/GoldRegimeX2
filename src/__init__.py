"""GoldRegimeX2 source package.

Orchestration + live-engine layer built ONLY from the pipeline_verification_bundle
in the review zip.  The research/optimization/backtest pipelines run the bundle's
OWN notebook code (the user's source of truth) via ``notebook_runner`` so nothing
is re-authored; the live-engine seam (strategy_backtest / risk_manager /
mt5_trader) mirrors the Explorer live-trading simulation so the backtester and
the live/demo platform share one behaviour.
"""
