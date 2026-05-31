from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class RegimeStats:
    bars: int
    percentage: float
    avg_return: float
    volatility: float
    avg_duration: float
    transition_in: int
    transition_out: int


def _state_runs(states: np.ndarray, state_id: int) -> list[int]:
    idx = np.where(states == state_id)[0]
    if len(idx) == 0:
        return []

    runs: list[int] = []
    start = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        ii = int(i)
        if ii == prev + 1:
            prev = ii
            continue
        runs.append(prev - start + 1)
        start = ii
        prev = ii
    runs.append(prev - start + 1)
    return runs


def summarize_regimes(
    states: np.ndarray,
    returns: np.ndarray,
    state_labels: dict[int, str],
) -> dict[str, dict]:
    states_arr = np.asarray(states)
    returns_arr = np.asarray(returns)
    n = int(min(len(states_arr), len(returns_arr)))
    if n == 0:
        return {
            name: asdict(
                RegimeStats(
                    bars=0,
                    percentage=0.0,
                    avg_return=0.0,
                    volatility=0.0,
                    avg_duration=0.0,
                    transition_in=0,
                    transition_out=0,
                )
            )
            for _, name in sorted(state_labels.items())
        }

    states_arr = states_arr[:n]
    returns_arr = returns_arr[:n]

    out: dict[str, dict] = {}
    for sid, name in sorted(state_labels.items()):
        idx = np.where(states_arr == sid)[0]
        bars = int(len(idx))
        pct = float(bars / max(n, 1))
        rets = returns_arr[idx] if bars else np.array([])
        avg_ret = float(np.mean(rets)) if bars else 0.0
        vol = float(np.std(rets)) if bars else 0.0

        runs = _state_runs(states_arr, sid)
        avg_dur = float(np.mean(runs)) if runs else 0.0

        trans_in = 0
        trans_out = 0
        for i in range(1, n):
            if states_arr[i] == sid and states_arr[i - 1] != sid:
                trans_in += 1
            if states_arr[i] != sid and states_arr[i - 1] == sid:
                trans_out += 1

        out[name] = asdict(
            RegimeStats(
                bars=bars,
                percentage=round(100.0 * pct, 3),
                avg_return=avg_ret,
                volatility=vol,
                avg_duration=avg_dur,
                transition_in=int(trans_in),
                transition_out=int(trans_out),
            )
        )

    return out


def summarize_trade_distribution(
    signals: np.ndarray,
    states: np.ndarray,
    state_labels: dict[int, str],
) -> dict[str, int]:
    sig = np.asarray(signals)
    st = np.asarray(states)
    n = int(min(len(sig), len(st)))
    if n == 0:
        return {name: 0 for _, name in sorted(state_labels.items())}

    sig = sig[:n]
    st = st[:n]

    prev = np.concatenate([[0], sig[:-1]])
    is_entry = (sig != 0) & (sig != prev)

    out = {name: 0 for _, name in sorted(state_labels.items())}
    for i in np.where(is_entry)[0]:
        sid = int(st[i])
        label = state_labels.get(sid, f"state_{sid}")
        out[label] = int(out.get(label, 0) + 1)
    return out


def write_regime_diagnostics(
    tf: str,
    states: np.ndarray,
    returns: np.ndarray,
    state_labels: dict[int, str],
    output_dir: str | Path = "reports",
) -> Path:
    report = {
        "timeframe": tf.upper(),
        "regimes": summarize_regimes(states, returns, state_labels),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"regime_diagnostics_{tf.upper()}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def get_occupancy_percentage(summary: dict[str, dict], label: str) -> float:
    row = summary.get(label, {}) if isinstance(summary, dict) else {}
    return float(row.get("percentage", 0.0)) / 100.0
