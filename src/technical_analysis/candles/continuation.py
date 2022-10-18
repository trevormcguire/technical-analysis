import pandas as pd

from technical_analysis.utils import is_bearish_trend, is_bullish_trend
from technical_analysis.candles.single import (is_long_body,
                                               is_bearish_gap,
                                               is_bullish_gap,
                                               positive_close,
                                               negative_close)


def bearish_tasuki_gap(open: pd.Series,
                       high: pd.Series,
                       low: pd.Series,
                       close: pd.Series,
                       trend_lookback: int = 30,
                       trend_threshold: float = -0.03,
                       min_body_size: float = 0.75,
                       min_gap_size: float = 0.002) -> pd.Series:
    """
    Downside Tasuki Gap (Continuation Pattern)
    ---------
    |
   [ ]
   [ ]
   [ ]
    |
    
           [ ]
      [ ]  [ ]
      [ ]   |
       |
    Candles:
    ---------
        1. a long, bearish body
        2. a bearish gap
        3. a bullish candle that opens inside the body of (2) and closes inside (but does not fill) the gap
    """
    bearish_trend = is_bearish_trend(close, lookback=trend_lookback, threshold=trend_threshold)
    bearish_long_body = is_long_body(open, high, low, close, min_body_size=min_body_size) & negative_close(open, close)
    bearish_gap = is_bearish_gap(high, low, min_gap_size=min_gap_size)
    # open > close in previous body
    opened_in_prev_body = (open > close.shift(1)) & (open < open.shift(1))
    # bearish gap so high(i) < low(i-1)
    closed_inside_gap = (close > high.shift(1)) & (close < low.shift(2))
    return bearish_trend & bearish_long_body.shift(2) & bearish_gap.shift(1) & opened_in_prev_body & closed_inside_gap


def bullish_tasuki_gap(open: pd.Series,
                       high: pd.Series,
                       low: pd.Series,
                       close: pd.Series,
                       trend_lookback: int = 30,
                       trend_threshold: float = 0.03,
                       min_body_size: float = 0.75,
                       min_gap_size: float = 0.002) -> pd.Series:
    """
    Upside Tasuki Gap (Continuation Pattern)
    ---------
       |
      [ ]  
      [ ] [ ]
          [ ]
    
    |
   [ ]
   [ ]
   [ ]
    |
    Candles:
    ---------
        1. a long, bullish body
        2. a bullish gap
        3. a bearish candle that opens inside the body of (2) and closes inside (but does not fill) the gap
    """
    bullish_trend = is_bullish_trend(close, lookback=trend_lookback, threshold=trend_threshold)
    bullish_long_body = is_long_body(open, high, low, close, min_body_size=min_body_size) & positive_close(open, close)
    bullish_gap = is_bullish_gap(high, low, min_gap_size=min_gap_size)
    # close > open in previous body
    opened_in_prev_body = (open > open.shift(1)) & (open < close.shift(1))
    closed_inside_gap = (close < low.shift(1)) & (close > high.shift(2))
    return bullish_trend & bullish_long_body.shift(2) & bullish_gap.shift(1) & opened_in_prev_body & closed_inside_gap