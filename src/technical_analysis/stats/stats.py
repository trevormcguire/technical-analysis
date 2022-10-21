from typing import Union, Tuple

import pandas as pd
import numpy as np


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
            - find the slope as a result of linear regression on (log(R/S), log(n))

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
    A = np.vstack([np.log(window_sizes), np.ones(len(rs_values))]).T
    H, c = np.linalg.lstsq(A, np.log(rs_values), rcond=-1)[0]
    return H if not return_c else H, c
