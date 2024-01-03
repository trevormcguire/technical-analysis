from typing import Tuple

import pandas as pd

from technical_analysis.indicators import atr
from technical_analysis.moving_average import ema


def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series]:
    """
    Pivot Point Calculation
    ```
        P = (High + Low + Close) / 3
        R1 = (P * 2) - Low
        R2 =P + (High - Low)
        S1 = (P * 2) - High
        S2 = P - (High - Low)
    ```
    """
    P = (high + close + low) / 3
    r1 = (P * 2) - low
    r2 = P + (high - low)
    s1 = (P * 2) - high
    s2 = P - (high - low)
    return r1, r2, s1, s2


def bbands(price: pd.Series, period: int = 20, num_std: int = 2) -> Tuple[pd.Series]:
    """
    Bollinger Bands Calculation
    
    1. Middle Band = 20-day simple moving average (SMA)
    2. Upper Band = 20-day SMA + (20-day standard deviation of price x 2) 
    3. Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
    """
    std = price.rolling(period).std()
    sma = price.rolling(period).mean()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return lower_band, upper_band


def dbands(price: pd.Series, period: int = 20) -> Tuple[pd.Series]:
    """
    Donchian Bands Calculation
    """
    upper_donchian = price.rolling(period).max()
    lower_donchian = price.rolling(period).min()
    return lower_donchian, upper_donchian


def kbands(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> Tuple[pd.Series]:
    """
    ------------
    Keltner Channels
    ------------
    Calculation:
        Middle Line: 20-day exponential moving average
        Upper Channel Line: 20-day EMA + (2 x ATR(10))
        Lower Channel Line: 20-day EMA - (2 x ATR(10))
    ------------
    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels
    """
    _ema = ema(close, period)
    _atr = atr(high, low, close, period=10)
    factor = _atr * 2
    lower_keltner = _ema - factor
    upper_keltner = _ema + factor
    return lower_keltner, upper_keltner


def chandalier_exit(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 22
) -> Tuple[pd.Series]:
    """
    Chandalier Exit -- Accounts for volatility, defined by Average True Range
    https://school.stockcharts.com/doku.php?id=technical_indicators:chandelier_exit
    """
    _atr = atr(high=high, low=low, close=close, period=period)
    long = high.rolling(period).max() - _atr * 3
    short = low.rolling(period).min() + _atr * 3
    return short, long


def ichimoku_clouds(
    high: pd.Series,
    low: pd.Series,
    conv_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    span_lag: int = 26,
    return_all: bool = True
) -> Tuple[pd.Series]:
    """
    ## Ichimoku Clouds

    ## Params
    1. `high`: pd.Series representing high prices
    2. `low`: pd.Series representing low prices
    3. `conv_period`: rolling window length for conversion line
    4. `base_period`: rolling window length for base line
    5. `span_b_period`: rolling window length for span_b calculation
    6. `span_lag`: how many periods to shift span_a and span_b
    7. `return_all`: if true, will return conversion line, base line, in addition to span A, span B.

    ## Returns
    - if `return_all`: conversion line, base line, span A, span B
    - otherwise: span A, span B

    ## Calculation
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
    - Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    - Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2)) shifted 26 days
    - Senkou Span B (Leading Span B): (52-period high + 52-period low)/2)) shifted 26 days

    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:ichimoku_cloud
    """
    conversion_line = (high.rolling(conv_period).max() + low.rolling(conv_period).min()) / 2
    base_line = (high.rolling(base_period).max() + low.rolling(base_period).min()) / 2
    span_a = ((conversion_line + base_line) / 2).shift(span_lag)
    span_b = ((high.rolling(span_b_period).max() + low.rolling(span_b_period).min()) / 2).shift(span_lag)
    if return_all:
        return conversion_line, base_line, span_a, span_b
    return span_a, span_b

ichimoku = ichimoku_clouds  # alias