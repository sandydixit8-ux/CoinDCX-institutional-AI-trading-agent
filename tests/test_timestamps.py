import numpy as np
import pandas as pd
import pytest

from ta_agent.timestamps import col_to_datetime, datetime_to_ms, series_to_ms


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "datetime64[us]"])
def test_datetime_to_ms_resolution_independent(dtype):
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    s = pd.Series(ts.tz_localize(None).astype(dtype).tz_localize("UTC"))
    ms = datetime_to_ms(s)
    assert ms.dtype.kind == "i"
    assert ms[0] == 1704067200000
    assert np.all(np.diff(ms) == 3_600_000)


def test_series_to_ms_handles_both_formats():
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    ms_int = datetime_to_ms(ts)
    dt = series_to_ms(pd.Series(ts))
    raw = series_to_ms(pd.Series(ms_int))
    assert dt[0] == 1704067200000
    assert list(raw) == [1704067200000, 1704070800000]
    assert list(dt) == list(raw)


def test_col_to_datetime_handles_both_formats():
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    ms_int = datetime_to_ms(ts)
    dt = col_to_datetime(pd.Series(ts))
    raw = col_to_datetime(pd.Series(ms_int))
    assert (dt.values.astype("datetime64[ns]") == raw.values.astype("datetime64[ns]")).all()
    assert dt.dt.tz is not None
