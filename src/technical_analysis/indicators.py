from typing import Callable

import numpy as np
import pandas as pd

from technical_analysis._common import _atr, _bbands, _dbands, _true_range
from technical_analysis.moving_average import ema, sma, wilder_ma
from technical_analysis.utils import log_returns


def volatility(price: pd.Series, period: int, use_log: bool = True) -> pd.Series:
    if use_log:
        price = log_returns(price)
    else:
        price = price.pct_change()
    return price.rolling(period).std()


true_range = _true_range
atr = _atr


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
    lower_band, upper_band = _bbands(price, period=period, num_std=num_std)
    return (price - lower_band) / (upper_band - lower_band)


def perc_d(price: pd.Series, period: int = 20) -> pd.Series:
    """
    %D measures a security's price in relation to the Donchian Bands
    """
    lower_band, upper_band = _dbands(price, period=period)
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
) -> tuple[pd.Series]:
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


def positive_directional_movement(high: pd.Series) -> pd.Series:
    return np.clip(high.diff(), a_min=0, a_max=None)


def negative_directional_movement(low: pd.Series) -> pd.Series:
    return np.clip(low.diff(), a_min=None, a_max=0)


def directional_movement(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> tuple[pd.Series]:
    """
    Directional Movement
    ------------
    Measures "trending quality" of the market, as used by Wilder. It is the largest part of today's range
    that is *outside* yesterday's range.

    - +DM = H[t] - H[t-1]
    - -DM = L[t] - L[t-1]

    Reference: CMT Level III Cirriculum (2020); page 205
    """
    # calculate +DM (PDM)
    pdm = positive_directional_movement(high=high)
    # calculate -DM (MDM)
    mdm = negative_directional_movement(low=low)
    # calculate true range
    tr = true_range(high=high, low=low, close=close)

    pdm_smoothed = wilder_ma(pdm, period=period)
    mdm_smoothed = wilder_ma(mdm, period=period)
    tr_smoothed = wilder_ma(tr, period=period)

    pdm_indicator = pdm_smoothed / tr_smoothed
    pdm_indicator = ((1 / period) * pdm_indicator.shift()) + pdm_indicator

    mdm_indicator = mdm_smoothed / tr_smoothed
    mdm_indicator = ((1 / period) * mdm_indicator.shift()) + mdm_indicator

    tr_smoothed_indicator = ((1 / period) * tr_smoothed.shift()) + tr_smoothed
    return pdm_indicator, mdm_indicator, tr_smoothed_indicator


def directional_indicators(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> tuple[pd.Series]:
    """
    Directional Indicators
    --------------
    Calculates Positive and Minus Directional Indicators

    Reference: CMT Level III Cirriculum (2020); page 205.
    """
    pdm, mdm, tr = directional_movement(high=high, low=low, close=close, period=period)
    pdi = pdm / tr
    mdi = mdm / tr
    return pdi, mdi


def true_directional_movement(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    """
    True Directional Movement Index (DX)
    -----------
    The difference betweeen the Positive Directional Indicator and the Minus Directional Indicator.
    When an upward trend is sustaineed, MDI will be zero, so the DX grows.

    Reference: CMT Level III Cirriculum (2020); page 206.
    """
    pdi, mdi = directional_indicators(high=high, low=low, close=close, period=period)
    return 100 * (np.abs(pdi - mdi) / pdi + mdi)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, dx_period: int = 10, period: int = 10) -> pd.Series:
    """
    Average Directional Movement Index
    --------------

    Reference: CMT Level III Cirriculum (2020); page 206.
    """
    dxi = true_directional_movement(high=high, low=low, close=close, period=dx_period)
    return wilder_ma(dxi, peeriod=period)


def adx_rating(high: pd.Series, low: pd.Series, close: pd.Series, dx_period: int = 10, period: int = 10) -> pd.Series:
    """
    Average Directional Movement Index Rating (ADXR)
    ------------
    Takes extreme variance of ADX into account.
    The distance between ADX and ADXR measrues overbought/oversold conditions.

    Reference: CMT Level III Cirriculum (2020); page 206.
    """
    average_dx = adx(high=high, low=low, close=close, dx_period=dx_period, period=period)
    return (average_dx + average_dx.shift(period)) / 2


def efficiency_ratio(price: pd.Series, period: int) -> pd.Series:
    """
    Efficiency Ratio
    ----------
    Measures noise in the market over a given number of periods.
    Calculated as the absolute value of the net price change divided by
    the sum of the absoluet individual price changes over the same period.
    NOTE: Lower values indicate more noise.

    Reference: CMT Level III Cirriculum (2020); page 210.
    """

    def _er(p: pd.Series) -> float:
        return abs(p.iloc[-1] - p.iloc[0]) / p.diff().abs().sum()

    return price.rolling(period).apply(_er)


def money_flow_volume(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Money Flow Volume
    ------
    Used in both Chaikin Money Flow (CMF) and Accumulation Distribution Line (ADL)

    Calculation
    ------
    ```
    1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low)
    2. Money Flow Volume = Money Flow Multiplier x Volume for the Period
    ```
    """
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_volume = mf_multiplier * volume
    return mf_volume


def money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF)
    ------
    Measures the amount of money flowing into an asset over a specific period.

    Calculation
    ------
    ```
    N-period CMF = N-period Sum of Money Flow Volume / N-period Sum of Volume
    ```
    """
    mf_volume = money_flow_volume(high=high, low=low, close=close, volume=volume)
    return mf_volume.rolling(period).sum() / volume.rolling(period).sum()


def adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Accumulation Distribution Line (ADL)
    ------
    The ADL is calculated as the running sum of Money Flow Volume
    """
    return np.cumsum(money_flow_volume(high=high, low=low, close=close, volume=volume))


def on_balance_volume(price: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume (OBV)
    ------
    Calculates the running total of positive and negative volume, where the sign of the volume is determined by daily returns.

    Calculation
    ------
    - If the closing price is above the prior close price then:
        `Current OBV = Previous OBV + Current Volume`

    - If the closing price is below the prior close price then:
        `Current OBV = Previous OBV  -  Current Volume`

    - If the closing prices equals the prior close price then:
        `Current OBV = Previous OBV (no change)`

    Reference: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/on-balance-volume-obv
    """
    return np.cumsum(np.sign(price.diff()) * volume)
