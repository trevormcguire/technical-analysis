import pandas as pd

from technical_analysis.candles.single import (is_long_body,
                                               is_bearish_gap,
                                               is_bullish_gap,
                                               positive_close,
                                               negative_close)


def bearish_tasuki_gap(open: pd.Series,
                       high: pd.Series,
                       low: pd.Series,
                       close: pd.Series,
                       min_body_size: float = 0.75) -> pd.Series:
    """
    Downside Tasuki Gap (Continuation Pattern)
    ---------

    Candles:
    ---------
        1. a long, bearish body
        2. a bearish gap
        3. a bullish candle that opens inside the body of (2) and closes inside (but does not fill) the gap
    """
    bearish_long_body = is_long_body(open, high, low, close, min_body_size=min_body_size) & negative_close(open, close)
    bearish_gap = is_bearish_gap(high, low)
    # open > close in previous body
    opened_in_prev_body = (open > close.shift(1)) & (open < open.shift(1))
    closed_inside_gap = (close > high.shift(1)) & (close < low.shift(2))
    return bearish_long_body.shift(2) & bearish_gap.shift(1) & opened_in_prev_body & closed_inside_gap


def bullish_tasuki_gap(open: pd.Series,
                       high: pd.Series,
                       low: pd.Series,
                       close: pd.Series,
                       min_body_size: float = 0.75) -> pd.Series:
    """
    Upside Tasuki Gap (Continuation Pattern)
    ---------

    Candles:
    ---------
        1. a long, bullish body
        2. a bullish gap
        3. a bearish candle that opens inside the body of (2) and closes inside (but does not fill) the gap
    """

    bullish_long_body = is_long_body(open, high, low, close, min_body_size=min_body_size) & positive_close(open, close)
    bullish_gap = is_bullish_gap(high, low)
    # close > open in previous body
    opened_in_prev_body = (open > open.shift(1)) & (open < close.shift(1))
    closed_inside_gap = (close > high.shift(1)) & (close < low.shift(2))
    return bullish_long_body.shift(2) & bullish_gap.shift(1) & opened_in_prev_body & closed_inside_gap