import numpy as np
import pandas as pd

from technical_analysis.moving_average import ema, sma, wilder_ma


def _bbands(price: pd.Series, period: int = 20, num_std: int = 2, ma_type: str = "sma") -> tuple[pd.Series]:
    """
    Bollinger Bands Calculation

    1. Middle Band = 20-day simple moving average (SMA)
    2. Upper Band = 20-day SMA + (20-day standard deviation of price x 2)
    3. Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
    """
    std = price.rolling(period).std()
    if ma_type == "sma":
        ma = price.rolling(period).mean()
    elif ma_type == "ema":
        ma = ema(price, period=period)
    else:
        raise NotImplementedError
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return lower_band, upper_band


def _dbands(price: pd.Series, period: int = 20) -> tuple[pd.Series]:
    """
    Donchian Bands Calculation
    """
    upper_donchian = price.rolling(period).max()
    lower_donchian = price.rolling(period).min()
    return lower_donchian, upper_donchian


def _true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    True Range
    -----------
        ```
            max(
                high - low
                abs(high - prev_close)
                abs(low - prev_close)
            )
        ```
    """
    high_low = high - low
    high_cp = np.abs(high - close.shift())
    low_cp = np.abs(low - close.shift())
    df = pd.concat([high_low, high_cp, low_cp], axis=1)
    return np.max(df, axis=1)


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    use_wilder_ma: bool = True,
) -> pd.Series:
    """
    Average True Range
    --------------------
    Measures volatility by taking the 14 day moving average `true_range`

    """
    tr = _true_range(high=high, low=low, close=close)
    if use_wilder_ma:
        average_tr = wilder_ma(tr, period)
    else:
        average_tr = sma(tr, period)
    return average_tr
