import pandas as pd
import pytest

from technical_analysis.backtest.strategy import MovingAverageCrossover


@pytest.fixture
def moving_avg_bullish_result() -> pd.Series:
    data = [
        False,  # 0
        True,  # 1
        False,  # 2
        False,  # 3
        False,  # 4
        False,  # 5
        False,  # 6
        False,  # 7
        False,  # 8
        False,  # 9
        False,  # 10
        True,  # 11
        False,  # 12
        False,  # 13
        False,  # 14
        False,  # 15
        False,  # 16
        False,  # 17
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
        False,  # 6
        True,  # 7
        False,  # 8
        False,  # 9
        False,  # 10
        False,  # 11
        False,  # 12
        False,  # 13
        False,  # 14
        False,  # 15
        True,  # 16
        False,  # 17
    ]
    return pd.Series(data)


def test_moving_average_bullish_crossover(price_df, moving_avg_bullish_result):
    strategy = MovingAverageCrossover("ma_fast", "ma_slow", "bullish", lookback_periods=1)
    bullish_result = strategy._run_bullish(data=price_df, lookback=1)
    assert (bullish_result == moving_avg_bullish_result).all()


def test_moving_average_bearish_crossover(price_df, moving_avg_bearish_result):
    strategy = MovingAverageCrossover("ma_fast", "ma_slow", "bearish", lookback_periods=1)
    bearish_result = strategy._run_bearish(data=price_df, lookback=1)
    assert (bearish_result == moving_avg_bearish_result).all()
