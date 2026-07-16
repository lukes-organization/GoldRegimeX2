"""shared/session_filter.py -- Single Session Filter (Phase 9).

This is the ONE session-filter implementation for the whole project. The
Strategy Tester, Explorer, Validator, Backtester and Live Trader must all
import `SessionFilter` / `ENABLE_SESSION_FILTER` from here. There must be no
inline notebook session logic, and no separate observability session buckets.

Hour boundaries are the faithful extraction of the notebooks'
`add_session_features` (London 7-16, NY 13-21, overlap 13-16) -- unchanged,
only centralised.

Session gating is CONFIGURABLE per timeframe. The audit established that the
production pipeline does NOT session-gate M15/M5 (processor.py applies the
session window to H1 only). Diagnostics must reflect the ACTUAL configuration
rather than pretend a Session filter ran.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

SESSION_FILTER_VALUES = [None, "London", "NY", "London_NY"]

LONDON_START_HOUR = 7
LONDON_END_HOUR = 16
NY_START_HOUR = 13
NY_END_HOUR = 21
OVERLAP_START_HOUR = 13
OVERLAP_END_HOUR = 16

# Phase 9: single source of truth for whether a timeframe is session-gated.
# Matches production reality (H1 only). Diagnostics read this dict; they never
# assume a Session stage filtered candidates.
ENABLE_SESSION_FILTER = {
    "H1": True,
    "M15": False,
    "M5": False,
}


@dataclass
class SessionCheckResult:
    timestamp: pd.Timestamp
    utc_time: str
    detected_session: str
    is_weekend: bool
    passes_london: bool
    passes_ny: bool
    passes_london_ny: bool


class SessionFilter:
    def __init__(
        self,
        london_start=LONDON_START_HOUR,
        london_end=LONDON_END_HOUR,
        ny_start=NY_START_HOUR,
        ny_end=NY_END_HOUR,
        overlap_start=OVERLAP_START_HOUR,
        overlap_end=OVERLAP_END_HOUR,
    ):
        self.london_start = london_start
        self.london_end = london_end
        self.ny_start = ny_start
        self.ny_end = ny_end
        self.overlap_start = overlap_start
        self.overlap_end = overlap_end

    def _hour(self, timestamp):
        return int(pd.Timestamp(timestamp).hour)

    def is_london(self, timestamp):
        h = self._hour(timestamp)
        return self.london_start <= h < self.london_end

    def is_ny(self, timestamp):
        h = self._hour(timestamp)
        return self.ny_start <= h < self.ny_end

    def is_overlap(self, timestamp):
        h = self._hour(timestamp)
        return self.overlap_start <= h < self.overlap_end

    def is_london_ny(self, timestamp):
        return self.is_london(timestamp) or self.is_ny(timestamp)

    def is_weekend(self, timestamp):
        return pd.Timestamp(timestamp).dayofweek >= 5

    def detect_session(self, timestamp):
        if self.is_overlap(timestamp):
            return "OVERLAP"
        if self.is_london(timestamp):
            return "LONDON"
        if self.is_ny(timestamp):
            return "NEW_YORK"
        return "ASIA"

    def session_column_name(self, session_filter):
        if session_filter is None:
            return "session_mask_none"
        s = str(session_filter).lower()
        if s == "london":
            return "session_mask_london"
        if s == "ny":
            return "session_mask_ny"
        if s in ("london_ny", "london-ny", "london ny"):
            return "session_mask_london_ny"
        raise ValueError("Unsupported session_filter: %r" % session_filter)

    def passes(self, timestamp, session_filter):
        if session_filter is None:
            return True
        s = str(session_filter).lower()
        if s == "london":
            return self.is_london(timestamp)
        if s == "ny":
            return self.is_ny(timestamp)
        if s in ("london_ny", "london-ny", "london ny"):
            return self.is_london_ny(timestamp)
        raise ValueError("Unsupported session_filter: %r" % session_filter)

    def add_session_features(self, df):
        out = df.copy()
        hour = out.index.hour
        london = (hour >= self.london_start) & (hour < self.london_end)
        ny = (hour >= self.ny_start) & (hour < self.ny_end)
        overlap = (hour >= self.overlap_start) & (hour < self.overlap_end)
        out["session"] = np.where(overlap, "OVERLAP", np.where(london, "LONDON", np.where(ny, "NEW_YORK", "ASIA")))
        out["session_mask_none"] = True
        out["session_mask_london"] = london
        out["session_mask_ny"] = ny
        out["session_mask_london_ny"] = london | ny
        return out

    def is_enabled(self, timeframe: str) -> bool:
        return bool(ENABLE_SESSION_FILTER.get(str(timeframe).upper(), False))

    def session_mask(self, df, timeframe: str, session_filter="London_NY"):
        """Boolean Series aligned to df.index.

        If gating is disabled for this timeframe (production reality for
        M15/M5) EVERY bar passes -- so the SESSION diagnostic stage will show
        0 losses, which is the truthful outcome, not a fabricated 100%.
        """
        idx = df.index
        if not self.is_enabled(timeframe):
            return pd.Series(True, index=idx)
        hour = idx.hour
        london = (hour >= self.london_start) & (hour < self.london_end)
        ny = (hour >= self.ny_start) & (hour < self.ny_end)
        s = None if session_filter is None else str(session_filter).lower()
        if s is None:
            mask = np.ones(len(idx), dtype=bool)
        elif s == "london":
            mask = london
        elif s == "ny":
            mask = ny
        else:
            mask = london | ny
        return pd.Series(mask, index=idx)

    def version_hash(self):
        blob = "london=%d-%d|ny=%d-%d|overlap=%d-%d" % (
            self.london_start, self.london_end, self.ny_start,
            self.ny_end, self.overlap_start, self.overlap_end,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def session_col_from_value(session_filter):
    """Module-level compatibility helper (single source of truth).

    Returns the session_mask_* column name for a given session_filter value.
    Both notebooks previously defined this inline; they now import it here.
    """
    return SessionFilter().session_column_name(session_filter)
