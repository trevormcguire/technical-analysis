import numpy as np
import pandas as pd

from technical_analysis.candles.single import is_long_body, is_bearish_gap


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