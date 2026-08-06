"""Timestamp helpers that are robust to pandas' datetime resolution (ns vs us).

pandas >= 3.0 defaults to microsecond-resolution datetimes, so the classic
``series.astype("int64") // 10 ** 6`` trick silently returns values that are
1000x too small. All epoch-ms conversions should go through these helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01")


def datetime_to_ms(values) -> np.ndarray:
    """Epoch-ms integers for datetime64 values, independent of resolution."""
    s = pd.to_datetime(pd.Series(values))
    base = pd.Timestamp("1970-01-01", tz="UTC") if s.dt.tz is not None else pd.Timestamp("1970-01-01")
    return (s.sub(base).dt.total_seconds() * 1000.0).astype("int64").to_numpy()


def series_to_ms(series) -> np.ndarray:
    """Epoch-ms integers for a Series that is either datetime64 or already epoch-ms ints."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return datetime_to_ms(series)
    return np.asarray(series, dtype="int64")


def col_to_datetime(series) -> pd.Series:
    """A datetime64 Series from either datetime64 values or epoch-ms integers."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)
    return pd.to_datetime(series, unit="ms", utc=True)
