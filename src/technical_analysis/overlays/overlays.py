from typing import Tuple

import numpy as np
import pandas as pd

from technical_analysis.indicators import atr
from technical_analysis.overlays.moving_average import ema


def pivot_points(high: pd.Series,
                 low: pd.Series,
                 close: pd.Series) -> Tuple[pd.Series]:
    """
    -----------
    Pivot Point Calculation
    -----------
        P=(High+Low+Close) / 3
        R1=(P*2)-Low
        R2=P+(High-Low)
        S1=(P*2)-High
        S2=P-(High-Low)
    """
    P = (high + close + low) / 3
    r1 = (P*2) - low
    r2 = P + (high - low)
    s1 = (P*2) - high
    s2 = P - (high - low)
    return r1, r2, s1, s2


def bbands(price: pd.Series, period: int = 20) -> Tuple[pd.Series]:
    """
    Bollinger Bands Calculation
    """
    std = price.rolling(period).std()
    upper_band = price + (std * 2)
    lower_band = price - (std * 2)
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
    ema_ = ema(close, period)
    atr_ = atr(high ,low, close, period=10)
    factor = (atr_ * 2)
    lower_keltner = ema_ - factor
    upper_keltner = ema_ + factor
    return lower_keltner, upper_keltner
