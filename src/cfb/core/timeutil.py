"""Time helpers. Every timestamp in core tables is datetime64[ns, UTC]."""
from __future__ import annotations

import numpy as np
import pandas as pd

# A game's results count as known this long after kickoff. Kickoff alone is not
# enough: with the postseason cutoff (kickoff minus 24 hours) a bowl that kicked
# off 23 hours 30 minutes before another one is still in progress at that cutoff.
RESULT_LAG = pd.Timedelta(hours=5)

# Postseason cutoff: this long before kickoff.
POSTSEASON_LEAD = pd.Timedelta(hours=24)

NS_MIN = np.iinfo("int64").min + 1  # always visible
NS_MAX = np.iinfo("int64").max  # never visible


def as_utc(value) -> pd.Timestamp:
    """Coerce a string, datetime or Timestamp to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.as_unit("ns")


def to_utc_series(values) -> pd.Series:
    """Parse a column of ISO strings or datetimes to datetime64[ns, UTC]."""
    out = pd.to_datetime(pd.Series(values), utc=True, errors="coerce", format="ISO8601")
    return out.dt.as_unit("ns")


def ns(values) -> np.ndarray:
    """Epoch nanoseconds as int64. NaT becomes NS_MAX (never visible)."""
    if isinstance(values, pd.Timestamp):
        return np.int64(as_utc(values).value)
    ser = pd.Series(values)
    if not isinstance(ser.dtype, pd.DatetimeTZDtype):
        ser = to_utc_series(ser)
    arr = ser.dt.as_unit("ns").to_numpy(dtype="datetime64[ns]")
    out = arr.astype("int64")
    out[np.isnat(arr)] = NS_MAX
    return out
