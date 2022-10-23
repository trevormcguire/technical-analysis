from typing import Union, Tuple

import pandas as pd
import numpy as np


def linear_regression(x: np.ndarray, y: np.ndarray = None) -> Tuple[float]:
    if y is None:
        return np.polyfit(np.arange(len(x)), x, 1)
    return np.polyfit(x, y, 1)


def get_slope(arr: np.ndarray) -> float:
    return linear_regression(arr)[0]


def get_yintercept(arr: np.ndarray) -> float:
    return linear_regression(arr)[1]


def rescaled_range(arr: np.ndarray):
    """
    Calculates Hursts' Rescaled Range,
        a measure of variability introduced by Hurst to
        shows how variability changes over time in a series

    Reference:
    --------------
        https://en.wikipedia.org/wiki/Rescaled_range

    """
    mean = np.mean(arr)
    mean_adjusted = arr - mean
    Z = np.cumsum(mean_adjusted)
    R = np.max(Z) - np.min(Z)
    S = np.std(arr, ddof=1)
    return R / S


def hurst_exp(price: pd.Series,
              window_sizes: Tuple[int] = (50, 200),
              window_increment: int = 10,
              return_c: bool = False) -> Union[float, Tuple[float]]:
    """
    The Hurst Exponent calculates the degree of persistance (think 'long-term memory')

    Calculation:
    ---------------
        1. For each window of size n:
            - calculate R/S
            - take the mean
            - Calculate linear regression on (log(R/S), log(n))
            - m, b = (H, c)


    Hurst Thresholds:
    ---------------
    H < 0.5 - Mean Reverting (anti-persistent)
              The closer to 0., the more mean-reverting
              https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process

    H ~ 0.5 - A geometric random walk (Brownian Motion)

    H > 0.5 - Trending (persistent) 
              The closer to 1., the stronger the trend
    
    Params:
    ---------------
        'price' -> pd.Series; 
        'window_sizes' -> tuple; min and max window sizes to use
        'window_increment' -> int; the 'step' in the 'window_sizes' to create windows
        'return_c' -> bool; returns H, c rather than just H

    Reference:
    ---------------
        https://en.wikipedia.org/wiki/Hurst_exponent
    """
    returns = price.pct_change()[1:]  # return over 1 period; first is nan
    series_length = len(returns)
    rs_values = []
    window_sizes = range(window_sizes[0], window_sizes[1]+1, window_increment)
    for window_size in window_sizes:
        window_rs_vals = []
        for start in range(0, series_length, window_size):
            window_rs_vals.append(rescaled_range([returns[start:start+window_size]]))
        rs_values.append(np.mean(window_rs_vals))
    H, c = linear_regression(np.log(window_sizes), np.log(rs_values))
    if return_c:
        return H, c
    return H


def fft_period(arr: pd.Series, top_n: int = 1):
    """
    Estimates the period using the FFT

    Returns the indexes of the largest amplitudes in the frequency domain,
        which corresponds to the period in terms of timesteps
    """
    arr -= np.mean(arr)  # remove dc component
    amps = np.fft.rfft(arr)
    largest_amp_indexes = np.argsort(amps)[::-1]
    return largest_amp_indexes[:top_n]


def autocorr(x: np.ndarray) -> np.ndarray:
    """
    Autocorrelation of x against itself
    """
    assert len(x.shape) == 1
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]


def autocorr_coef(x: np.ndarray, lags: tuple = (1, 200)) -> float:
    """
    Normalized covariance statistic between x(t) and x(t-n), where n is a lag in range 'lags'
    """
    min_lag = min(lags[0], len(x))
    max_lag = min(lags[1], len(x))
    results = []
    for lag in range(min_lag, max_lag+1):
        results.append(np.corrcoef(x[:-lag], x[lag:])[:,1][0])
    return np.array(results)

