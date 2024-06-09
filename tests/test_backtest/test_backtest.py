import pandas as pd
import pytest

from technical_analysis.backtest import Backtest
from technical_analysis.backtest.strategy import MovingAverageCrossover


@pytest.fixture
def backtest_results() -> dict:
    return {
        "benchmark": 6.0,
        "strategy": 0.5,
        "max_drawdown": -0.5,
        "max_profit": 1.0,
        "num_trades": 2,
        "returns": [-0.5, 1.0],
    }


def test_calculate_results(price_df, backtest_results):
    backtest = Backtest(
        entry_criteria=[MovingAverageCrossover("ma_fast", "ma_slow", "bullish", lookback_periods=1)],
        exit_criteria=[MovingAverageCrossover("ma_fast", "ma_slow", "bearish", lookback_periods=1)],
        max_positions=1,
        use_next_open=True,
    )
    entry = backtest._apply_criteria(price_df, exit=False)  # entry
    exit = backtest._apply_criteria(price_df, exit=True)
    results = backtest.calculate_results(price_df, entry=entry, exit=exit)
    for k in backtest_results.keys():
        assert results[k] == backtest_results[k]
