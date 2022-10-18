import numpy as np
import pandas as pd

from technical_analysis.utils import is_bullish_trend, is_bearish_trend
from technical_analysis.candles.single import is_long_body, negative_close, positive_close, body_outside_body


def dark_cloud(open: pd.Series,
               high: pd.Series,
               low: pd.Series,
               close: pd.Series,
               trend_lookback: int = 30,
               trend_threshold: float = 0.03,
               min_body_size: float = 0.75,
               new_high_periods: int = 100) -> pd.Series:
    """
    Bearish Reversal Pattern

    Candles:
    ---------
        1. a long bullish body
            - We define this as 75% of price action taking place between open and close
        2. next opens at a new high, then closes below midpoint of the body of (1)
            - We define new high as within the previous 100 periods
            - The midpoint is the timesteps (high(t-1) - low(t-1))
    """
    bullish_trend = is_bullish_trend(close, lookback=trend_lookback, threshold=trend_threshold)
    bullish_long_body = is_long_body(open, high, low, close, min_body_size=min_body_size) & positive_close(open, close)
    new_high_comparator = (high == high.rolling(new_high_periods).max())
    close_below_midpoint = close < (high.shift(1) + low.shift(1))/2
    return (bullish_trend & bullish_long_body.shift(1) & new_high_comparator & close_below_midpoint)



def bullish_engulfing(open: pd.Series,
                      high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      trend_lookback: int = 30,
                      trend_threshold: float = 0.03) -> pd.Series:
    """
    Bullish Englufing Pattern (Reversal)
    ---------

    Candles:
    ---------
    In a bearish trend:
        1. a small body
        2. a body that completely englufs the body of (1)
    """
    bearish_trend = is_bearish_trend(close, lookback=trend_lookback, threshold=trend_threshold)
    outisde_body = body_outside_body(open, close)
    return (bearish_trend & outisde_body & positive_close(open, close))


def bearish_engulfing(open: pd.Series,
                      high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      trend_lookback: int = 30,
                      trend_threshold: float = 0.03) -> pd.Series:
    """
    Bullish Englufing Pattern (Reversal)
    ---------

    Candles:
    ---------
    In a bearish trend:
        1. a small body
        2. a body that completely englufs the body of (1)
    """
    bearish_trend = is_bullish_trend(close, lookback=trend_lookback, threshold=trend_threshold)
    outisde_body = body_outside_body(open, close)
    return (bearish_trend & outisde_body & negative_close(open, close))