"""
-------------
Single Candlestick Patterns
-------------
    "For any given, candle it is best to compare only recent price action (5-10 periods)" (Morris)

"""

import numpy as np
import pandas as pd

from technical_analysis.utils import get_body


def positive_close(open: pd.Series, close: pd.Series) -> pd.Series:
    return close > open


def negative_close(open: pd.Series, close: pd.Series) -> pd.Series:
    return close < open


def is_gap_down(high: pd.Series, low: pd.Series, min_gap_size: float = 0.003) -> pd.Series:
    """
    Determines whether yesterday's low was greater than today's high (plus an optional threshold factor)

    Params:
    --------
        'min_gap_size' -> minimum size of the gap in % terms
    """
    shifted_low = low.shift(1)
    return shifted_low > (high + (high * min_gap_size))


def is_gap_up(high: pd.Series, low: pd.Series, min_gap_size: float = 0.003) -> pd.Series:
    """
    Determines whether yesterday's high was less than today's low (minus an optional threshold factor)
    """
    shifted_high = high.shift(1)
    return shifted_high < (low - (low * min_gap_size))


def is_gap(high: pd.Series, low: pd.Series, min_gap_size: float = 0.003) -> pd.Series:
    """
    Determines whether a gap is present (direction-agnostic)
    """
    return is_gap_up(high, low, min_gap_size) | is_gap_down(high, low, min_gap_size)


def is_long_body(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    min_body_size: float = 0.75,
    min_relative_size: float = 1.5,
    lookback: int = 5,
) -> pd.Series:
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
    has_large_body = close_open_range > (high_low_range * min_body_size)
    if not lookback:
        return has_large_body
    shifted_series = [close_open_range.shift(n) for n in range(1, lookback + 1)]
    shifted_series = np.max(pd.concat(shifted_series, axis=1), axis=1)  # max open-close range of all lookback periods
    is_relatively_large = close_open_range > (shifted_series * min_relative_size)
    return is_relatively_large & has_large_body


def is_short_body(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    max_body_size: float = 0.25,
    max_relative_size: float = 0.5,
    lookback: int = 5,
) -> pd.Series:
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
    has_small_body = close_open_range < (high_low_range * max_body_size)
    if not lookback:
        return has_small_body
    shifted_series = [close_open_range.shift(n) for n in range(1, lookback + 1)]
    shifted_series = np.min(pd.concat(shifted_series, axis=1), axis=1)  # min open-close range of all lookback periods
    is_relatively_small = close_open_range < (shifted_series * max_relative_size)
    return is_relatively_small & has_small_body


def is_doji(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    relative_threshold: float = 0.1,
) -> pd.Series:
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


def is_dragonfly_doji(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    relative_threshold: float = 0.1,
    upper_threshold: float = 0.001,
) -> pd.Series:
    """
    Dragonfly doji can be a reversal

    Candles:
    ---------
        Same as doji except, open/close are near high
    """
    doji = is_doji(open, high, low, close, relative_threshold)
    near_high = close > (high - (high * upper_threshold))
    return doji & near_high


def is_gravestone_doji(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    relative_threshold: float = 0.1,
    lower_threshold: float = 0.001,
) -> pd.Series:
    """
    Candles:
    ---------
        Same as doji except, open/close are near low
    """
    doji = is_doji(open, high, low, close, relative_threshold)
    near_low = close < (low + (low * lower_threshold))
    return doji & near_low


def is_marubozu(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    max_shadow_size: float = 0.2,
) -> pd.Series:
    """
    Determines if the candle has very small shadows on both ends(aka bald, shaven)
    ---------

    Calculation:
    ---------
        The body size is (1-max_shadow_size) percentage of the entire shadow size
    """
    high_low_range = np.abs(high - low)
    close_open_range = np.abs(close - open)
    return close_open_range > (high_low_range * (1 - max_shadow_size))


def is_outside(high: pd.Series, low: pd.Series, threshold: float = 0.001, lookback: int = 1) -> pd.Series:
    """
    Determines if a candle's shadow is larger than shadow of prior candle (outside)
    Current shadow is outside previous shadow
    """
    if not threshold:
        return (high > high.shift(lookback)) & (low < low.shift(lookback))
    shifted_high = high.shift(lookback)
    shifted_high = shifted_high + (shifted_high * threshold)
    shifted_low = low.shift(lookback)
    shifted_low = shifted_low - (shifted_low * threshold)
    return (high > shifted_high) & (low < shifted_low)


def body_outside_shadow(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.001,
    lookback: int = 1,
) -> pd.Series:
    """
    Current body is outside previous shadow
    """
    shifted_high = high.shift(lookback)
    shifted_high = shifted_high + (shifted_high * threshold)
    shifted_low = low.shift(lookback)
    shifted_low = shifted_low - (shifted_low * threshold)

    lower_body, upper_body = get_body(open, close)

    return (upper_body > shifted_high) & (lower_body < shifted_low)


def body_outside_body(open: pd.Series, close: pd.Series, threshold: float = 0.001, lookback: int = 1) -> pd.Series:
    """
    Current body is outside previous bdoy
    """
    lower_body, upper_body = get_body(open, close)

    shifted_upper_body = upper_body.shift(lookback)
    shifted_upper_body = shifted_upper_body + (shifted_upper_body * threshold)
    shifted_lower_body = lower_body.shift(lookback)
    shifted_lower_body = shifted_lower_body - (shifted_lower_body * threshold)
    return (upper_body > shifted_upper_body) & (lower_body < shifted_lower_body)


def is_inside(high: pd.Series, low: pd.Series, threshold: float = 0.001, lookback: int = 1) -> pd.Series:
    """
    Determines if a candle's shadow is smaller than shadow of prior candle (outside)
    Current shadow is inside prior shadow
    """
    if not threshold:
        return (high < high.shift(lookback)) & (low > low.shift(lookback))
    shifted_high = high.shift(lookback)
    shifted_high = shifted_high - (shifted_high * threshold)
    shifted_low = low.shift(lookback)
    shifted_low = shifted_low + (shifted_low * threshold)
    return (high < shifted_high) & (low > shifted_low)


def shadow_inside_body(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.001,
    lookback: int = 1,
) -> pd.Series:
    """
    Current shadow is inside previous body
    """
    prev_lower_body, prev_upper_body = get_body(open, close)
    prev_upper_body = prev_upper_body - (prev_upper_body * threshold)
    prev_lower_body = prev_lower_body + (prev_lower_body * threshold)
    return (high < prev_upper_body.shift(lookback)) & (low > prev_lower_body.shift(lookback))


def body_inside_body(open: pd.Series, close: pd.Series, threshold: float = 0.001, lookback: int = 1) -> pd.Series:
    """
    Current body is inside previous body
    """
    lower_body, upper_body = get_body(open, close)

    shifted_upper_body = upper_body.shift(lookback)
    shifted_upper_body = shifted_upper_body - (shifted_upper_body * threshold)
    shifted_lower_body = lower_body.shift(lookback)
    shifted_lower_body = shifted_lower_body + (shifted_lower_body * threshold)
    return (upper_body < shifted_upper_body) & (lower_body > shifted_lower_body)


def body_inside_shadow(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float = 0.001,
    lookback: int = 1,
) -> pd.Series:
    """
    Current body is inside previous shadow
    """
    lower_body, upper_body = get_body(open, close)

    high = high.shift(lookback)
    high = high - (high * threshold)
    low = low.shift(lookback)
    low = low + (low * threshold)

    return (upper_body < high) & (lower_body > low)


def spinning_top(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    A spinning top has upper and lower shadows larger than the body
    Signals indecision
    """
    lower_body, upper_body = get_body(open, close)
    body_width = upper_body - lower_body
    return ((high - upper_body) > body_width) & ((lower_body - low) > body_width)


def hammer(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    upper_threshold: float = 0.05,
    body_size_multiplier: int = 2,
) -> pd.Series:
    """
    Hammer (Hanging Man)
    -------------
        Upper shadow is very small or nonexistent
        Lower shadow is 2-3x the size of the body

    Params
    -------
        'upper_threshold' -> % of candle that upper shadow occupies (float between (0, 1))
        'body_size_multiplier' -> lower shadow must be at least body_size*body_size_multiplier
    """
    lower_body, upper_body = get_body(open, close)
    lower_shadow = lower_body - low
    upper_shadow = high - upper_body
    total_size = high - low
    body_size = upper_body - lower_body
    upper_shadow_criteria = (upper_shadow / total_size) < upper_threshold
    return upper_shadow_criteria & (lower_shadow > (body_size * body_size_multiplier))


def inverted_hammer(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lower_threshold: float = 0.05,
    body_size_multiplier: int = 2,
) -> pd.Series:
    """
    Inverted Hammer (Shooting Star)
    -------------
        Lower shadow is very small or nonexistent
        Upper shadow is 2-3x the size of the body

    Params
    -------
        'lower' -> % of candle that lower shadow occupies (float between (0, 1))
        'body_size_multiplier' -> upper shadow must be at least body_size*body_size_multiplier
    """
    lower_body, upper_body = get_body(open, close)
    lower_shadow = lower_body - low
    upper_shadow = high - upper_body
    total_size = high - low
    body_size = upper_body - lower_body
    lower_shadow_criteria = (lower_shadow / total_size) < lower_threshold
    return lower_shadow_criteria & (upper_shadow > (body_size * body_size_multiplier))
