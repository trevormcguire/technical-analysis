import numpy as np
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


@pytest.fixture
def rvol_results() -> pd.Series:
    return pd.Series([1.71])  # 6 / ((1 + 2 + 3 + 4 + 5 + 6) / 6)


def test_rvol(input_df, rvol_results):
    result = indicators.rvol(input_df["volume"], period=6).dropna().reset_index(drop=True).round(2)
    assert (result == rvol_results).all()


@pytest.fixture
def money_flow_volume_results() -> pd.Series:
    # [(Close  -  Low) - (High - Close)] /(High - Low)
    return pd.Series([0.333333 * 100, 0.0 * 200, 0.0 * 300, 0.333333 * 400, 0.5 * 500, -1.0 * 600])


def test_money_flow_volume(input_df, money_flow_volume_results):
    result = indicators.money_flow_volume(input_df["high"], input_df["low"], input_df["close"], input_df["volume"])
    result = result.round(4)
    assert np.isclose(result, money_flow_volume_results).all()


@pytest.fixture
def on_balance_volume_results():
    return pd.Series(np.cumsum([-200, 300, 400, -500, -600]))


def test_on_balance_volume(input_df, on_balance_volume_results):
    result = indicators.on_balance_volume(input_df["close"], volume=input_df["volume"]).dropna().reset_index(drop=True)
    assert (result == on_balance_volume_results).all()
