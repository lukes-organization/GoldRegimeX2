"""shared/prediction.py -- Single Prediction Contract (Phase 7).

One canonical object that the Explorer, Backtester, Live Trader and
Diagnostics all consume. Diagnostics may NEVER index probability arrays
manually (`probs[:, 2]` etc.) -- they must read `.probability` / `.prediction`
/ `.regime` off this object.

This wraps whatever model interface exists (binary or 3-class) behind one API
so that the reported probability is exactly the one used to make the trade
decision. It does not change model logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PredictionResult:
    candidate_id: str
    timeframe: str
    probability: float          # directional probability actually thresholded
    prediction: int             # +1 long, -1 short, 0 flat/no-trade
    regime: int                 # canonical HMM state (0/1/2), production mapping
    model_uuid: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def directional_probability(proba_row, direction: int) -> float:
    """Extract the single directional probability from a model proba row.

    Supports both the 3-class [down, flat, up] and binary [down, up] layouts
    behind ONE function, so no caller indexes arrays by hand.
    """
    row = np.asarray(proba_row, dtype=float).ravel()
    if row.size == 3:
        up, down = row[2], row[0]
    elif row.size == 2:
        up, down = row[1], row[0]
    elif row.size == 1:
        up = float(row[0]); down = 1.0 - up
    else:
        raise ValueError("Unsupported proba row width: %d" % row.size)
    if direction > 0:
        return float(up)
    if direction < 0:
        return float(down)
    return float(max(up, down))


def make_prediction(
    candidate_id: str,
    timeframe: str,
    proba_row,
    direction: int,
    threshold: float,
    regime: int,
    model_uuid: Optional[str] = None,
) -> PredictionResult:
    p = directional_probability(proba_row, direction)
    passed = p >= float(threshold)
    pred = int(np.sign(direction)) if passed and direction != 0 else 0
    return PredictionResult(
        candidate_id=str(candidate_id),
        timeframe=str(timeframe).upper(),
        probability=p,
        prediction=pred,
        regime=int(regime),
        model_uuid=model_uuid,
        metadata={"threshold": float(threshold), "passed": bool(passed)},
    )
