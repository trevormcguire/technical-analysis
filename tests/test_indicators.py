import pandas as pd
import pytest

from technical_analysis import indicators


@pytest.fixture
def input_df() -> pd.DataFrame:
    data = [
        {"open": 5.0, "high": 10.0, "low": 2.5, "close": 7.5, "volume": 100},
        {"open": 2.0, "high": 8.0, "low": 2.0, "close": 5.0, "volume": 200},
        {"open": 4.0, "high": 12.0, "low": 4.0, "close": 8.0, "volume": 300},
        {"open": 8.0, "high": 12.0, "low": 6.0, "close": 10.0, "volume": 400},
        {"open": 10.0, "high": 10.0, "low": 6.0, "close": 9.0, "volume": 500},
        {"open": 8.0, "high": 8.0, "low": 6.0, "close": 6.0, "volume": 600},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def perc_d_results() -> pd.Series:
    data = [1.0, 1.0, 0.5, 0.0]
    return pd.Series(data)


def test_perc_d(input_df, perc_d_results):
    d = indicators.perc_d(input_df["close"], period=3)
    d = d.dropna().reset_index(drop=True).round(2)
    assert (d == perc_d_results).all()
