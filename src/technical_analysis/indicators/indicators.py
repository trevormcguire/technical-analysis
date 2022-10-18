import numpy as np
import pandas as pd

from technical_analysis.overlays.moving_average import sma, ema



def std(price: pd.Series, period: int) -> pd.Series:
    """
    Rolling standard deviation
    """
    return price.rolling(period).std()


def roc(price: pd.Series, period: int) -> pd.Series:
    """
    Acceleration (rate of change)
    """
    shifted_price = price.shift(period)
    return (price - shifted_price) / shifted_price


def atr(high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    Average True Range (measures volatility)
    and is the 14 day moving average of the following:

        max(
            high - low
            abs(high - prev_close)
            abs(low - prev_close)
        )

    """
    high_low = high - low
    high_cp = np.abs(high - close.shift(1))
    low_cp = np.abs(low - close.shift())
    df = pd.concat([high_low, high_cp, low_cp], axis=1)
    true_range = np.max(df, axis=1)
    average_true_range = true_range.rolling(period).mean()
    return average_true_range


def rsi(price: pd.Series, period: int) -> pd.Series:
    """
    Relative Strength Index

    Calculation:
    -----------
        Average Gain = sum(gains over period) / period
        Average Loss = sum(losses over period) / period
        RS = Average Gain / Average Loss
        RSI = 100 - (100/(1+RS))
    """
    delta = price.diff()[1:] #first row is nan
    gains, losses = delta.clip(lower=0), delta.clip(upper=0).abs()
    gains = sma(gains, period)
    losses = sma(losses, period)
    rs = gains / losses
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:] = np.select([losses == 0, gains == 0, True], [100, 0, rsi])
    valid_rsi = rsi[period - 1:]
    assert ((0 <= valid_rsi) & (valid_rsi <= 100)).all()
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
