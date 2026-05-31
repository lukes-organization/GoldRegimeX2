from __future__ import annotations

from typing import Iterable

APPROVED_PARAMS: set[str] = {
    "obs_cov",
    "trans_cov",
    "persistence_threshold",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "n_estimators",
    "reg_alpha",
    "reg_lambda",
    "scale_pos_weight",
    "entry_probability_threshold",
    "confirmation_bars",
    "atr_stop_multiplier",
    "atr_target_multiplier",
}

FORBIDDEN_PARAMS: set[str] = {
    "h1_min_median_sharpe",
    "h1_min_median_pf",
    "h1_max_trades_per_100",
    "m15_min_median_sharpe",
    "m15_min_median_pf",
    "m15_max_trades_per_100",
    "m5_min_median_sharpe",
    "m5_min_median_pf",
    "m5_max_trades_per_100",
}


def audit_search_space(param_names: Iterable[str]) -> None:
    names = sorted({str(p) for p in param_names})

    forbidden = [p for p in names if p in FORBIDDEN_PARAMS]
    if forbidden:
        raise RuntimeError(f"Forbidden Optuna parameters detected: {forbidden}")

    unknown = [p for p in names if p not in APPROVED_PARAMS]
    if unknown:
        raise RuntimeError(f"Unknown Optuna parameters detected: {unknown}")


def search_space_as_text(param_names: Iterable[str]) -> str:
    names = sorted({str(p) for p in param_names})
    return "\n".join(f"  - {name}" for name in names)
