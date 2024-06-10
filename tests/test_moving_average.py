import pandas as pd
import pytest

from technical_analysis import moving_average as ma


@pytest.fixture
def input_series() -> pd.Series:
    return pd.Series(list(range(21)))


@pytest.fixture
def sma_result() -> pd.Series:
    return pd.Series([9.5, 10.5])


def test_sma(input_series, sma_result):
    result = ma.sma(price=input_series, period=20).dropna().reset_index(drop=True)
    assert (result == sma_result).all()


def test_n_smoothed_ema(input_series):
    result = ma.n_smoothed_ema(price=input_series, period=5, n=2).round(2)
    assert (result.dropna() == ma.ema(ma.ema(price=input_series, period=5), period=5).round(2).dropna()).all()

    result = ma.n_smoothed_ema(price=input_series, period=(5, 4), n=2).round(2)
    assert (result.dropna() == ma.ema(ma.ema(price=input_series, period=5), period=4).round(2).dropna()).all()

    result = ma.double_ema(price=input_series, period=5).round(2).dropna()
    assert (result == ma.n_smoothed_ema(price=input_series, period=5, n=2).round(2).dropna()).all()

    result = ma.triple_ema(price=input_series, period=3).round(2).dropna()
    assert (result == ma.n_smoothed_ema(price=input_series, period=3, n=3).round(2).dropna()).all()
