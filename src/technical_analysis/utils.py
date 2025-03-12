import numpy as np
import pandas as pd


def log_returns(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1))


def get_body(open: pd.Series, close: pd.Series) -> tuple[pd.Series]:
    """
    gets upper and lower bounds of candle body, direction-agnostic
    """
    body = pd.concat([open, close], axis=1)
    upper_body = body.max(axis=1)
    lower_body = body.min(axis=1)
    return lower_body, upper_body


def get_trend(price: pd.Series, lookback: int) -> pd.Series:
    """
    Uses percentage change from 'lookback' periods ago as a proxy for trend
    """
    shifted = price.shift(lookback)
    return (price - shifted) / shifted


def is_bullish_trend(price: pd.Series, lookback: int, threshold: float = None) -> pd.Series:
    """
    Determines whether the trend is > threshold

    Params:
    --------
        'lookback':
            number of periods ago to compare current price to
        'threshold':
            min percentage change from 'lookback' periods ago to count as bullish trend
            > defaults to (lookback/1000)
    """
    if threshold is None:
        threshold = lookback / 1000
    else:
        threshold = max(threshold, threshold * -1)
    return get_trend(price, lookback) > threshold


def is_bearish_trend(price: pd.Series, lookback: int, threshold: float = None) -> pd.Series:
    """
    Determines whether the trend is < threshold

    Params:
    --------
        'lookback':
            number of periods ago to compare current price to
        'threshold':
            max percentage change from 'lookback' periods ago to count as bearish trend
            > defaults to (lookback/1000) * -1
    """
    if threshold is None:
        threshold = (lookback / 1000) * -1
    else:
        threshold = min(threshold, threshold * -1)
    return get_trend(price, lookback) < threshold


def is_new_high(price: pd.Series, lookback: int):
    return price == price.rolling(lookback).max()


def is_new_low(price: pd.Series, lookback: int):
    return price == price.rolling(lookback).min()


def cum_pct_change(price: pd.Series) -> pd.Series:
    return price.pct_change().cumsum()
