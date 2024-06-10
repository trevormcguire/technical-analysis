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


@pytest.fixture
def ema_result() -> pd.Series:
    return pd.Series(
        [
            0.67,  # weighted avg between (0, 1) will be about 2/3
            1.56,  # (2/3)*2 + (1/3)*0.67
            2.52,  # (2/3)*3 + (1/3)*1.56
            3.51,  # (2/3)*4 + (1/3)*2.52
            4.5,  # (2/3)*5 + (1/3)*3.51
            5.5,
            6.5,
            7.5,
            8.5,
            9.5,
            10.5,
            11.5,
            12.5,
            13.5,
            14.5,
            15.5,
            16.5,
            17.5,
            18.5,
            19.5,
        ]
    )


def test_ema(input_series, ema_result):
    result = ma.ema(price=input_series, period=2).round(2).dropna().reset_index(drop=True)
    assert (result == ema_result).all()


def test_n_smoothed_ema(input_series):
    result = ma.n_smoothed_ema(price=input_series, period=5, n=2).round(2)
    assert (result.dropna() == ma.ema(ma.ema(price=input_series, period=5), period=5).round(2).dropna()).all()

    result = ma.n_smoothed_ema(price=input_series, period=(5, 4), n=2).round(2)
    assert (result.dropna() == ma.ema(ma.ema(price=input_series, period=5), period=4).round(2).dropna()).all()

    result = ma.double_ema(price=input_series, period=5).round(2).dropna()
    assert (result == ma.n_smoothed_ema(price=input_series, period=5, n=2).round(2).dropna()).all()

    result = ma.triple_ema(price=input_series, period=3).round(2).dropna()
    assert (result == ma.n_smoothed_ema(price=input_series, period=3, n=3).round(2).dropna()).all()
