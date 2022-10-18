import numpy as np
import pandas as pd


def slope(x):
    m, _ = np.polyfit(range(len(x)), x, 1)
    return m


def get_trend(price: pd.Series, lookback: int) -> pd.Series:
    """
    Uses slope as a proxy for trend
    """
    return price.rolling(lookback, min_periods=lookback).apply(slope)


def is_bullish_trend(price: pd.Series, lookback: int, threshold: float = 0.0005) -> pd.Series:
    """
    Determines whether the trend (slope) as a % of 'price' is > threshold
    """
    threshold = max(threshold, threshold*-1)
    return (get_trend(price, lookback) / price) > threshold


def is_bearish_trend(price: pd.Series, lookback: int, threshold: float = -0.0005) -> pd.Series:
    """
    Determines whether the trend (slope) as a % of 'price' is < threshold
    """
    threshold = min(threshold, threshold*-1)
    return (get_trend(price, lookback) / price) < threshold