import pandas as pd
import pytest

from technical_analysis.backtest import Backtest
from technical_analysis.backtest.strategy import MovingAverageCrossover


@pytest.fixture
def price_df() -> pd.DataFrame:
    data = [
        [100.0, 95.0],  # 0
        [101.0, 96.0],  # 1
        [102.0, 97.0],  # 2
        [103.0, 98.0],  # 3
        [104.0, 99.0],  # 4
        [101.0, 100.0],  # 5
        [99.0, 100.5],  # 6
        [97.0, 101.0],  # 7
        [98.0, 100.0],  # 8
        [98.0, 99.5],  # 9
        [100.0, 99.0],  # 10
        [102.0, 98.0],  # 11
        [101.0, 99.0],  # 12
        [102.5, 99.5],  # 13
        [103.0, 100.5],  # 14
    ]
    df = pd.DataFrame(data, columns=["ma_fast", "ma_slow"])
    return df


@pytest.fixture
def moving_avg_bullish_result() -> pd.Series:
    data = [
        False,  # 0
        False,  # 1
        False,  # 2
        False,  # 3
        False,  # 4
        False,  # 5
        False,  # 6
        False,  # 7
        False,  # 8
        False,  # 9
        True,  # 10
        False,  # 11
        False,  # 12
        False,  # 13
        False,  # 14
    ]
    return pd.Series(data)


@pytest.fixture
def moving_avg_bearish_result() -> pd.Series:
    data = [
        False,  # 0
        False,  # 1
        False,  # 2
        False,  # 3
        False,  # 4
        False,  # 5
        True,  # 6
        False,  # 7
        False,  # 8
        False,  # 9
        False,  # 10
        False,  # 11
        False,  # 12
        False,  # 13
        False,  # 14
    ]
    return pd.Series(data)


def test_moving_average_bullish_crossover(price_df, moving_avg_bullish_result):
    strategy = MovingAverageCrossover("ma_fast", "ma_slow", "bullish", confirmation_periods=0, lookback_periods=1)
    bullish_result = strategy._run_bullish(data=price_df, lookback=1)
    assert (bullish_result == moving_avg_bullish_result).all()


def test_moving_average_bearish_crossover(price_df, moving_avg_bearish_result):
    strategy = MovingAverageCrossover("ma_fast", "ma_slow", "bearish", confirmation_periods=0, lookback_periods=1)
    bearish_result = strategy._run_bearish(data=price_df, lookback=1)
    assert (bearish_result == moving_avg_bearish_result).all()
