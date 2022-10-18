import numpy as np
import pandas as pd

from technical_analysis.candles.single import is_long_body


def dark_cloud(open: pd.Series,
               high: pd.Series,
               low: pd.Series,
               close: pd.Series,
               long_body_threshold: float = 0.75,
               new_high_periods: int = 100) -> pd.Series:
    """
    Bearish Reversal Pattern

    Candles:
    ---------
        1. a long bullish body
            - We define this as 75% of price action taking place between open and close
        2. next opens at a new high, then closes below midpoint of the body of (1)
            - We define new high as within the previous 100 periods
            - The midpoint is the timesteps (high - low)
    """
    bullish_long_body = is_long_body(open, high, low, close, long_body_threshold=long_body_threshold) & (close > open)
    new_high_comparator = (high == high.rolling(new_high_periods).max())
    close_below_midpoint = close < (high + low)/2
    return (bullish_long_body.shift(1) & new_high_comparator & close_below_midpoint)



def engulfing(open: pd.Series,
              high: pd.Series,
              low: pd.Series,
              close: pd.Series,
              small_body_threshold: float = 0.04) -> pd.Series:
    """
    Englufing Pattern (Reversal)
    ---------

    Candles:
    ---------
        1. a small body (close-open range)
            - defined by abs(close-open)/open less than 'small_body_threshold'
        2. a body that completely englufs the body of (1)
    """
    pass