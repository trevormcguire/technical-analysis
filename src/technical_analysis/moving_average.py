import numpy as np
import pandas as pd


def sma(price: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average (SMA)
    ---------
    """
    return price.rolling(period).mean()


def lwma(price: pd.Series, period: int) -> pd.Series:
    """
    Linearly Weighted Moving Average (LWMA)
    ---------
    """
    weights = np.arange(1, period + 1)
    return price.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def ema(price: pd.Series, period: int) -> pd.Series:
    """
    Exponentially Smoothed Moving Average (EMA)
    ---------
    """
    transformed_series = price.ewm(span=period, adjust=False).mean()
    transformed_series[: period - 1] = np.nan
    return transformed_series


def n_smoothed_ema(price: pd.Series, period: int | tuple[int], n: int) -> pd.Series:
    """
    Variably Smoothed EMA
    """
    if isinstance(period, tuple):
        assert len(period) == n
    else:
        period = [period] * n
    res = price.copy()
    for p in period:
        res = ema(res, p)
    return res


def double_ema(price: pd.Series, period: int) -> pd.Series:
    """
    Doubly-Smoothed EMA (Mulloy)
    """
    return n_smoothed_ema(price, period=period, n=2)


def triple_ema(price: pd.Series, period: int) -> pd.Series:
    """
    Triply-Smoothed EMA (Mulloy, 1994)
    """
    return n_smoothed_ema(price, period=period, n=3)


def wilder_ma(price: pd.Series, period: int) -> pd.Series:
    """
    Welles Wilder (1978) method
    ---------

    Notes:
    ---------
        > Used in all of Wilder's Indicators (ATR, RSI, DMI)
        > Is more responsive than SMA (more recent prices are weighed more heavily)

    Calculation:
    ---------
        MA(i) = ((n-1) * MA(i-1) + Price(i)) / n
    """
    ma = sma(price, period)
    shifted_ma = ma.shift(1)
    return ((period - 1) * shifted_ma + price) / period


def gma(price: pd.Series, period: int) -> pd.Series:
    """
    Geometric Moving Average (GMA)
    ---------

    Notes:
    ---------
        > Mainly used in Indexes
        > Is the SMA of the daily percent change over some period (n)
    """
    return sma(price.pct_change(), period)


def tma(price: pd.Series, period: int):
    """
    Triangular Moving Average (TMA)
    ---------

    Notes:
    ---------
        > Takes an SMA with period n
        > Then takes another SMA of period n/2
        > This weighs the middle of the series more heavily
    """
    return sma(sma(price, period), period // 2)


def kama(
    price: pd.Series,
    period: int,
    min_smoothing_constant: int,
    max_smoothing_constant: int,
) -> pd.Series:
    """
    Kaufman Adaptive Moving Average
    ---------

    Notes:
    ---------
        > Like other Adaptive (Variable) EMAs, KAMA accounts for volatility
        > Closely follows price when volatility is low, lags more when volatility is higher
        > Efficiency Ratio (ER) fluctuates between (0, 1)

    Params:
    ---------
        'price' -> a pandas series of prices
        'period' -> period to use in efficiency ratio (ER)
        'min_smoothing_constant' -> smoothing constant for the fastest ema (ex: 2)
        'max_smoothing_constant' -> smoothing constant for the slowest ema (ex: 30)

    Ref:
    ---------
        https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average
    """
    change = np.abs(price - price.shift(period))
    volatiltiy = np.abs(price - price.shift(1)).rolling(period).sum()
    efficiency_ratio = change / volatiltiy

    min_smoothing_constant = 2 / (min_smoothing_constant + 1)
    max_smoothing_constant = 2 / (max_smoothing_constant + 1)
    smoothing_constant = (
        efficiency_ratio * (min_smoothing_constant - max_smoothing_constant) + min_smoothing_constant
    ) ** 2

    # Current KAMA = Prior KAMA + SC x (Price - Prior KAMA)
    kama = sma(price, period)  # the first value of kama is just the SMA
    kama_shifted = kama.shift(1)
    kama = kama_shifted + smoothing_constant * (price - kama_shifted)
    return kama


def crossover_signal(ma1: pd.Series, ma2: pd.Series):
    """
    Moving Average Crossover Signal
    ---------

    Returns:
    ---------
        the indexes at which ma1 and ma2 cross (direction-agnostic)
    """
    return np.where(np.diff(np.sign(ma1 - ma2)))[0]


def bullish_crossover_signal(ma1: pd.Series, ma2: pd.Series):
    """
    Bullish Moving Average Crossover Signal
    ---------

    Returns:
    ---------
        the indexes at which ma1 crosses below ma2
        - Generally, this means ma1 should be a faster moving average than ma2
    """
    return np.where(np.diff(np.sign(ma1 - ma2)) > 0)[0]


def bearish_crossover_signal(ma1: pd.Series, ma2: pd.Series):
    """
    Bearish Moving Average Crossover Signal
    ---------

    Returns:
    ---------
        the indexes at which ma1 crosses below ma2
        - Generally, this means ma1 should be a faster moving average than ma2
    """
    return np.where(np.diff(np.sign(ma1 - ma2)) < 0)[0]
