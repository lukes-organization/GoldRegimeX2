# Unified Session Filter (Phase 8).
#
# Single reference implementation of session/time-of-day filtering for the
# GoldRegime_X pipeline. Faithful extraction of the session logic that
# already exists in both notebooks (add_session_features /
# session_col_from_value, defined identically in Strategy_Tester.ipynb and
# GoldRegimeX_Explorer.ipynb) -- hour boundaries below were NOT changed,
# only centralised.
#
# There is intentionally no StrategyTesterSessionFilter,
# ExplorerSessionFilter, MT5SessionFilter, or BacktesterSessionFilter --
# only this one class.

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

    def check(self, timestamp):
        ts = pd.Timestamp(timestamp)
        return SessionCheckResult(
            timestamp=ts,
            utc_time=ts.strftime("%Y-%m-%d %H:%M:%S"),
            detected_session=self.detect_session(ts),
            is_weekend=self.is_weekend(ts),
            passes_london=self.is_london(ts),
            passes_ny=self.is_ny(ts),
            passes_london_ny=self.is_london_ny(ts),
        )

    def version_hash(self):
        blob = "london=%d-%d|ny=%d-%d|overlap=%d-%d" % (
            self.london_start,
            self.london_end,
            self.ny_start,
            self.ny_end,
            self.overlap_start,
            self.overlap_end,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_session_filter_test_suite(session_filter=None, verbose=True):
    # Phase 9: automated session filter test suite.
    sf = session_filter or SessionFilter()

    cases = [
        {"name": "London open (Mon 07:00 UTC)", "ts": "2025-01-06 07:00", "filter": "London", "expected": True},
        {"name": "London close (Mon 15:59 UTC)", "ts": "2025-01-06 15:59", "filter": "London", "expected": True},
        {"name": "Post-London close (Mon 16:00 UTC)", "ts": "2025-01-06 16:00", "filter": "London", "expected": False},
        {"name": "New York overlap (Mon 14:00 UTC)", "ts": "2025-01-06 14:00", "filter": "London_NY", "expected": True},
        {"name": "Asia (Mon 03:00 UTC)", "ts": "2025-01-06 03:00", "filter": "London_NY", "expected": False},
        {"name": "Friday close (Fri 20:59 UTC)", "ts": "2025-01-10 20:59", "filter": "NY", "expected": True},
        {"name": "Sunday open (Sun 22:00 UTC)", "ts": "2025-01-05 22:00", "filter": "London_NY", "expected": False},
        {"name": "Weekend (Sat 12:00 UTC)", "ts": "2025-01-04 12:00", "filter": "London", "expected": True, "weekend_warn": True},
        {"name": "DST transition (2025-03-30 12:00 UTC)", "ts": "2025-03-30 12:00", "filter": "London", "expected": True},
    ]

    rows = []
    warnings = []
    for c in cases:
        got = sf.passes(c["ts"], c["filter"])
        weekend = sf.is_weekend(c["ts"])
        passed = got == c["expected"]
        row = {
            "test": c["name"],
            "timestamp": c["ts"],
            "filter": c["filter"],
            "expected": c["expected"],
            "actual": got,
            "detected_session": sf.detect_session(c["ts"]),
            "weekend": weekend,
            "result": "PASS" if passed else "FAIL",
        }
        if c.get("weekend_warn") and weekend and got:
            warnings.append(
                "Session filter has no explicit weekend guard: %s returned %s. "
                "Dormant in practice because OHLCV panels contain no weekend bars." % (c["name"], got)
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    if verbose:
        print("=" * 26 + "  SESSION FILTER TEST REPORT  " + "=" * 26)
        print(df.to_string(index=False))
        for w in warnings:
            print("WARNING: " + w)
        overall = "PASS" if (df["result"] == "PASS").all() else "FAIL"
        print("Overall: " + overall)
    return df, warnings
