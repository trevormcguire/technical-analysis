"""
-------------
Single Candlestick Patterns
-------------
    "For any given, candle it is best to compare only recent price action (5-10 periods)" (Morris)

"""
import pandas as pd
import numpy as np


def positive_close(open: pd.Series, close: pd.Series) -> pd.Series:
    return close > open


def negative_close(open: pd.Series, close: pd.Series) -> pd.Series:
    return close < open


def is_bearish_gap(high: pd.Series, low: pd.Series, min_gap_size: float = 0.006) -> pd.Series:
    """
    Determines whether yesterday's low was greater than today's high (plus an optional threshold factor)

    Params:
    --------
        'min_gap_size' -> minimum size of the gap in % terms
    """
    shifted_low = low.shift(1)
    return shifted_low > high + (high*min_gap_size)


def is_bullish_gap(high: pd.Series, low: pd.Series, min_gap_size: float = 0.006) -> pd.Series:
    """
    Determines whether yesterday's high was less than today's low (minus an optional threshold factor)
    """
    shifted_high = high.shift(1)
    return shifted_high < low - (low*min_gap_size)


def is_gap(high: pd.Series, low: pd.Series, min_gap_size: float = 0.006) -> pd.Series:
    """
    Determines whether a gap is present (direction-agnostic)
    """
    return (is_bullish_gap(high, low, min_gap_size) | is_bearish_gap(high, low, min_gap_size))


def is_long_body(open: pd.Series,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 min_body_size: float = 0.75,
                 min_relative_size: float = 1.5,
                 lookback: int = 5) -> pd.Series:
    """
    Determines if a candle has a long body (based on open-close range)

    Params:
    --------
        'min_body_size' -> minimum percentage of the candle that the open-close range occupies
        'min_relative_size' -> minimum relative size of a candle compared to the previous 'lookback' candles
        'lookback' -> how many candles to lookback for 'relative_size' param
    """
    high_low_range = np.abs(high - low)
    close_open_range = np.abs(close - open) 
    has_large_body = close_open_range > high_low_range - (high_low_range*min_body_size)

    shifted_series = [close_open_range.shift(n) for n in range(1, lookback+1)]
    shifted_series = np.max(pd.concat(shifted_series, axis=1), axis=1)  # max open-close range of all lookback periods
    is_relatively_large = close_open_range > (shifted_series * min_relative_size)
    return is_relatively_large & has_large_body


def is_short_body(open: pd.Series,
                  high: pd.Series,
                  low: pd.Series,
                  close: pd.Series,
                  max_body_size: float = 0.25,
                  max_relative_size: float = 0.5,
                  lookback: int = 5) -> pd.Series:
    """
    Determines whether a candle has a short body (based on open-close range)

    Params:
    --------
        'max_body_size' -> max percentage of the candle that the open-close range occupies
        'max_relative_size' -> max relative size of a candle compared to the previous 'lookback' candles
        'lookback' -> how many candles to lookback for 'relative_size' param
    """
    high_low_range = np.abs(high - low)
    close_open_range = np.abs(close - open) 
    has_small_body = close_open_range < high_low_range - (high_low_range*max_body_size)

    shifted_series = [close_open_range.shift(n) for n in range(1, lookback+1)]
    shifted_series = np.min(pd.concat(shifted_series, axis=1), axis=1)  # min open-close range of all lookback periods
    is_relatively_small = close_open_range < (shifted_series * max_relative_size)
    return is_relatively_small & has_small_body


def is_doji(open: pd.Series,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            relative_threshold: float = 0.1) -> pd.Series:
    """
    Doji represents indecision

    Candles:
    ---------
        The open and close are virtually equal

    Calculation:
    ---------
        Open-close range (body) must be smaller than a % of the high-low range (shadow)
        > Where the % is determined by 'relative_threshold'
    """
    open_close_range = np.abs(close - open)
    high_low_range = np.abs(high - low)
    return open_close_range < (high_low_range * relative_threshold)  # body is smaller than a % of the shadow


def is_dragonfly_doji(open: pd.Series,
                      high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      relative_threshold: float = 0.1,
                      upper_threshold: float = 0.001) -> pd.Series:
    """
    Dragonfly doji can be a reversal

    Candles:
    ---------
        Same as doji except, open/close are near high
    """
    doji = is_doji(open, high, low, close, relative_threshold)
    near_high = close > (high - (high*upper_threshold))
    return doji & near_high


def is_gravestone_doji(open: pd.Series,
                       high: pd.Series,
                       low: pd.Series,
                       close: pd.Series,
                       relative_threshold: float = 0.1,
                       lower_threshold: float = 0.001) -> pd.Series:
    """
    Candles:
    ---------
        Same as doji except, open/close are near low
    """
    doji = is_doji(open, high, low, close, relative_threshold)
    near_low = close < (low + (low*lower_threshold))
    return doji & near_low


def is_marubozu(open: pd.Series,
                high: pd.Series,
                low: pd.Series,
                close: pd.Series,
                max_shadow_size: float = 0.05) -> pd.Series:
    """
    Determines if the candle has very small shadows on both ends(aka bald, shaven)
    ---------

    Calculation:
    ---------
        The body size is (1-max_shadow_size) percentage of the entire shadow size
    """
    high_low_range = np.abs(high - low)
    close_open_range = np.abs(close - open) 
    return close_open_range > high_low_range*(1-max_shadow_size)