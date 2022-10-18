import pandas as pd
import numpy as np

# https://school.stockcharts.com/doku.php?id=chart_analysis:candlestick_pattern_dictionary

# reversal patterns
BEARISH = ["dark_cloud"]
BULLISH = ["bullish_engulfing"]

# continuation patterns
CONTINUATION = ["tasuki_gap"]


def is_bearish_gap(high: pd.Series, low: pd.Series, threshold: float = 0.003) -> pd.Series:
    """
    Determines whether yesterday's low was greater than today's high (plus an optional threshold factor)
    """
    shifted_low = low.shift(1)
    return shifted_low > high + (high*threshold)


def is_bullish_gap(high: pd.Series, low: pd.Series, threshold: float = 0.003) -> pd.Series:
    """
    Determines whether yesterday's high was less than today's low (minus an optional threshold factor)
    """
    shifted_high = high.shift(1)
    return shifted_high < low - (low*threshold)


def is_gap(high: pd.Series, low: pd.Series, threshold: float = 0.003) -> pd.Series:
    """
    Determines whether a gap is present (direction-agnostic)
    """
    return (is_bullish_gap(high, low, threshold) | is_bearish_gap(high, low, threshold))


def is_long_body(open: pd.Series,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 long_body_threshold: float = 0.75) -> pd.Series:
    """
    Determines if a candle has a long body by comparing the open-close range to the high-low range
        (minus a thresholed)
    The idea here is that the open-close range should be at least 'long_body_threshold' percentage of the entire candle 
    Add price % change threshold to ensure its truly a 'long' body (must be a 1% change for example) ... ideally should base on volatilty. 
    """
    high_low_range = np.abs(high - low)
    close_open_range = np.abs(close - open) 
    return close_open_range > high_low_range - (high_low_range*long_body_threshold)


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


def doji(open: pd.Series,
         high: pd.Series,
         low: pd.Series,
         close: pd.Series,
         equality_threshold: float = 0.001,
         shadow_threshold: float = 0.005) -> pd.Series:
    """
    Doji represents indecision

    Candles:
    ---------
        The open and close are virtually equal
            - 'equal' means +/- some margin of error, determined by 'threshold'
        There should be a degree of shadows present (visually like some sort of cross or 'T')
    """
    close_minus_open = np.abs(close - open) / open
    high_minus_low = np.abs(high - low) / open
    return (close_minus_open < equality_threshold) & (high_minus_low > shadow_threshold)


def dragonfly_doji(open: pd.Series,
                   high: pd.Series,
                   low: pd.Series,
                   close: pd.Series,
                   equality_threshold: float = 0.003,
                   shadow_threshold: float = 0.005) -> pd.Series:
    """
    Dragonfly doji can be a reversal

    Candles:
    ---------
        Same as doji except, open/close are near high
    """
    is_doji = doji(open, high, low, close, equality_threshold, shadow_threshold)
    return is_doji & ((high - close) < equality_threshold)


def tasuki_gap(open: pd.Series,
               high: pd.Series,
               low: pd.Series,
               close: pd.Series,
               long_body_threshold: float = 0.75) -> pd.Series:
    """
    Downside Tasuki Gap (Continuation Pattern)
    ---------

    Candles:
    ---------
        1. a long, bearish body
        2. a bearish gap
        3. a bullish candle that opens inside the body of (2) and closes inside (but does not fill) the gap
    """
    bearish_long_body = is_long_body(open, high, low, close, long_body_threshold=long_body_threshold) & (close < open)
    bearish_gap = is_bearish_gap(high, low)
    opened_in_prev_body = (open > close.shift(1)) & (open < open.shift(1))
    closed_inside_gap = (close > high.shift(1)) & (close < low.shift(2))
    return bearish_long_body.shift(2) & bearish_gap.shift(1) & opened_in_prev_body & closed_inside_gap


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





