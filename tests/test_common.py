import pandas as pd
import pytest

from technical_analysis._common import _atr, _dbands, _true_range


@pytest.fixture
def input_df() -> pd.DataFrame:
    data = [
        {"open": 9.3, "high": 9.5, "low": 9.0, "close": 9.4, "volume": 1000},
        {"open": 9.5, "high": 10.0, "low": 9.2, "close": 9.7, "volume": 2000},
        {"open": 9.4, "high": 9.6, "low": 9.3, "close": 9.5, "volume": 4000},
        {"open": 9.9, "high": 9.9, "low": 9.3, "close": 9.4, "volume": 2000},
        {"open": 9.6, "high": 9.9, "low": 9.0, "close": 9.1, "volume": 1000},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def true_range_results() -> pd.Series:
    data = [0.5, 0.8, 0.4, 0.6, 0.9]
    return pd.Series(data)


def test_true_range(input_df, true_range_results):
    tr = _true_range(high=input_df["high"], low=input_df["low"], close=input_df["close"]).round(2)
    assert (tr == true_range_results).all()


@pytest.fixture
def atr_results() -> pd.Series:
    data = [0.65, 0.6, 0.5, 0.75]
    return pd.Series(data)


def test_atr(input_df, atr_results):
    avg_tr = _atr(high=input_df["high"], low=input_df["low"], close=input_df["close"], period=2, use_wilder_ma=False)
    avg_tr = avg_tr.dropna().reset_index(drop=True).round(2)
    assert (avg_tr == atr_results).all()


@pytest.fixture
def dbands_results():
    lower = [9.4, 9.4, 9.1]
    upper = [9.7, 9.7, 9.5]
    return pd.Series(lower), pd.Series(upper)


def test_dbands(input_df, dbands_results):
    lower, upper = _dbands(input_df["close"], period=3)
    assert (lower.dropna().reset_index(drop=True) == dbands_results[0]).all()
    assert (upper.dropna().reset_index(drop=True) == dbands_results[1]).all()
