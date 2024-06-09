from typing import Callable, Tuple

import numpy as np
import pandas as pd

from technical_analysis.overlays import bbands, dbands
from technical_analysis.moving_average import ema, sma, wilder_ma
from technical_analysis.utils import log_returns



def volatility(price: pd.Series, period: int, use_log: bool = True) -> pd.Series:
    if use_log:
        price = log_returns(price)
    else:
        price = price.pct_change()
    return price.rolling(period).std()


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    use_wilder_ma: bool = True,
) -> pd.Series:
    """
    Average True Range (measures volatility)
    and is the 14 day moving average of the following:
    ```
        max(
            high - low
            abs(high - prev_close)
            abs(low - prev_close)
        )
    ```
    """
    high_low = high - low
    high_cp = np.abs(high - close.shift(1))
    low_cp = np.abs(low - close.shift())
    df = pd.concat([high_low, high_cp, low_cp], axis=1)
    true_range = np.max(df, axis=1)
    if use_wilder_ma:
        average_true_range = wilder_ma(true_range, period)
    else:
        average_true_range = sma(true_range, period)
    return average_true_range


def rsi(price: pd.Series, period: int, ma_fn: Callable = sma, use_wilder_ma: bool = True) -> pd.Series:
    """
    Relative Strength Index

    Calculation:
    -----------
        Average Gain = sum(gains over period) / period
        Average Loss = sum(losses over period) / period
        RS = Average Gain / Average Loss
        RSI = 100 - (100/(1+RS))
    """
    if use_wilder_ma:
        ma_fn = wilder_ma

    delta = price.diff()[1:]  # first row is nan
    gains, losses = delta.clip(lower=0), delta.clip(upper=0).abs()
    gains = ma_fn(gains, period)
    losses = ma_fn(losses, period)
    rs = gains / losses
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:] = np.select([losses == 0, gains == 0, True], [100, 0, rsi])
    return rsi


def perc_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    -------
    Williams %R
    Reflects the level of the close relative to the high-low range over a given period of time
    Oscillator between 0 and 1
    -------
    Calculation:
        %R = (Highest High - Close)/(Highest High - Lowest Low)

    Params:
        1. 'period' -> lookback period for highest high and lowest low
    -------
    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r
    """
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    return (highest_high - close) / (highest_high - lowest_low)


def perc_b(price: pd.Series, period: int = 20, num_std: int = 2) -> pd.Series:
    """
    %B measures a security's price in relation to the Bollinger Bands
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce
    %B = (Price - Lower Band) / (Upper Band - Lower Band)
    """
    lower_band, upper_band = bbands(price, period=period, num_std=num_std)
    return (price - lower_band) / (upper_band - lower_band)


def perc_d(price: pd.Series, period: int = 20) -> pd.Series:
    """
    %D measures a security's price in relation to the Donchian Bands
    """
    lower_band, upper_band = dbands(price, period=period)
    return (price - lower_band) / (upper_band - lower_band)


def tsi(price: pd.Series, period1: int = 25, period2: int = 13) -> pd.Series:
    """
    True Strength Index
    ------------

    Calculation:
    ------------
        Double Smoothed PC
        ------------------
        PC = Current Price minus Prior Price
        First Smoothing = 25-period EMA of PC
        Second Smoothing = 13-period EMA of 25-period EMA of PC

        Double Smoothed Absolute PC
        ---------------------------
        Absolute Price Change |PC| = Absolute Value of Current Price minus Prior Price
        First Smoothing = 25-period EMA of |PC|
        Second Smoothing = 13-period EMA of 25-period EMA of |PC|

        TSI = 100 x (Double Smoothed PC / Double Smoothed Absolute PC)

    Reference:
    ------------
        https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index
    """
    shifted_price = price.shift(1)
    pc = price - shifted_price
    double_smoothed_pc = ema(ema(pc, period1), period2)

    pc = np.abs(pc)
    double_smoothed_abs_pc = ema(ema(pc, period1), period2)
    return 100 * (double_smoothed_pc / double_smoothed_abs_pc)


def trix(price: pd.Series, period: int = 15) -> pd.Series:
    """
    Displays percent rate of change as a triply smoothed moving average
        > similar to MACD, but smoother

    Calculation:
    ------------
        1. Single-Smoothed EMA = 15-period EMA of the closing price
        2. Double-Smoothed EMA = 15-period EMA of Single-Smoothed EMA
        3. Triple-Smoothed EMA = 15-period EMA of Double-Smoothed EMA
        4. TRIX = 1-period percent change in Triple-Smoothed EMA

    Reference:
    -----------
        https://school.stockcharts.com/doku.php?id=technical_indicators:trix
    """
    trix = ema(ema(ema(price, period), period), period)
    return trix.pct_change(1)


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    perc_k_smoothing: int = 0,
    perc_d_smoothing: int = 3,
) -> Tuple[pd.Series]:
    """
    Stochastic Oscillator
    ----------

    Calculation:
    ----------
        %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
        %D = 3-day SMA of %K

    Three modes:
    ----------
        1. Fast
            perc_k_smoothing = 0
            perc_d_smoothing = 3

        2. Slow
            perc_k_smoothing = 3
            perc_d_smoothing = 3

        3. Full
            perc_k_smoothing > 3
            perc_d_smoothing > 3

    Reference:
    -----------
        https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

    """
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()

    perc_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    if perc_k_smoothing:
        perc_k = sma(perc_k, perc_k_smoothing)
    perc_d = sma(perc_k, perc_d_smoothing)  # the trigger line
    return perc_k, perc_d


def macd(
    price: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    return_histogram: bool = True,
) -> pd.Series:
    """
    Moving Average Convergence/Divergence (MACD)

    Calculation:
    -----------
        MACD Line: (12-day EMA - 26-day EMA)
        Signal Line: 9-day EMA of MACD Line
        MACD Histogram: MACD Line - Signal Line

    Reference:
    -----------
        https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd

    """
    macd_line = ema(price, period=fast_period) - ema(price, period=slow_period)
    signal_line = ema(macd_line, signal_period)
    if return_histogram:
        return macd_line - signal_line
    return signal_line


def rvol(volume: pd.Series, period: int) -> pd.Series:
    """
    Relative Volume
    
    `RVOL = current volume / average volume over the look-back period`
    https://school.stockcharts.com/doku.php?id=technical_indicators:rvol
    """
    return volume / volume.rolling(period).mean()
