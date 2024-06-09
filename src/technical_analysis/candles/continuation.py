import pandas as pd

from technical_analysis.candles.single import (
    body_inside_shadow,
    is_gap_down,
    is_gap_up,
    is_long_body,
    negative_close,
    positive_close,
)
from technical_analysis.utils import is_bearish_trend, is_bullish_trend, is_new_high, is_new_low


def rising_n(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int,
    lookback: int = 20,
) -> pd.Series:
    """
    Rising 'n' Method
    ------------


    Candles:
    ------------
        1. a long green body
        2. 'n' candles, each inside high-low range of (1)
        3. n+2 candle closes at a new high (determined by close > all closes 'lookback' periods ago)
    """
    bullish_trend = is_bullish_trend(close, lookback=lookback)
    long_green = is_long_body(open, high, low, close) & positive_close(open, close)
    insides = []
    shifts = list(range(n, 0, -1))
    lookback_periods = list(range(1, n + 1))
    for shift, period in list(zip(shifts, lookback_periods)):
        insides.append(body_inside_shadow(open, high, low, close, lookback=period).shift(shift))
    combined_insides = insides.pop(0)
    for series in insides:
        combined_insides = combined_insides & series
    return bullish_trend & long_green & combined_insides & is_new_high(close, lookback)


def rising_three(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """
    Rising Three Method
    ------------


    Candles:
    ------------
        1. a long green body
        2. three candles, each inside high-low range of (1)
        3. fifth candle closes at a new high (determined by close > all closes 'lookback' periods ago)
    """
    return rising_n(open, high, low, close, n=3, lookback=lookback)


def falling_n(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int,
    lookback: int = 20,
) -> pd.Series:
    """
    Rising 'n' Method
    ------------


    Candles:
    ------------
        1. a long red body
        2. 'n' candles, each inside high-low range of (1)
        3. n+2 candle closes at a new low (determined by close < all closes 'lookback' periods ago)
    """
    bearish_trend = is_bearish_trend(close, lookback=lookback)
    long_red = is_long_body(open, high, low, close) & negative_close(open, close)
    insides = []
    shifts = list(range(n, 0, -1))
    lookback_periods = list(range(1, n + 1))
    for shift, period in list(zip(shifts, lookback_periods)):
        insides.append(body_inside_shadow(open, high, low, close, lookback=period).shift(shift))
    combined_insides = insides.pop(0)
    for series in insides:
        combined_insides = combined_insides & series
    return bearish_trend & long_red & combined_insides & is_new_low(close, lookback)


def falling_three(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """
    Falling Three Method
    ------------


    Candles:
    ------------
        1. a long red body
        2. three candles, each inside high-low range of (1)
        3. fifth candle closes at a new low (determined by close < all closes 'lookback' periods ago)
    """
    return falling_n(open, high, low, close, n=3, lookback=lookback)


def bearish_tasuki_gap(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    trend_lookback: int = 30,
    trend_threshold: float = -0.03,
    min_body_size: float = 0.75,
    min_gap_size: float = 0.002,
) -> pd.Series:
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
    bearish_gap = is_gap_down(high, low, min_gap_size=min_gap_size)
    # open > close in previous body
    opened_in_prev_body = (open > close.shift(1)) & (open < open.shift(1))
    # bearish gap so high(i) < low(i-1)
    closed_inside_gap = (close > high.shift(1)) & (close < low.shift(2))
    return bearish_trend & bearish_long_body.shift(2) & bearish_gap.shift(1) & opened_in_prev_body & closed_inside_gap


def bullish_tasuki_gap(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    trend_lookback: int = 30,
    trend_threshold: float = 0.03,
    min_body_size: float = 0.75,
    min_gap_size: float = 0.002,
) -> pd.Series:
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
    bullish_gap = is_gap_up(high, low, min_gap_size=min_gap_size)
    # close > open in previous body
    opened_in_prev_body = (open > open.shift(1)) & (open < close.shift(1))
    closed_inside_gap = (close < low.shift(1)) & (close > high.shift(2))
    return bullish_trend & bullish_long_body.shift(2) & bullish_gap.shift(1) & opened_in_prev_body & closed_inside_gap
