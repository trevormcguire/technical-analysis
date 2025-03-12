import numpy as np
import pandas as pd


def linear_regression(x: np.ndarray, y: np.ndarray = None) -> tuple[float]:
    if y is None:
        return np.polyfit(np.arange(len(x)), x, 1)
    return np.polyfit(x, y, 1)


def get_slope(arr: np.ndarray) -> float:
    return linear_regression(arr)[0]


def get_yintercept(arr: np.ndarray) -> float:
    return linear_regression(arr)[1]


def period(arr: pd.Series, top_n: int = 1):
    """
    Estimates the period using the FFT

    Returns the indexes of the 'top_n' largest amplitudes in the frequency domain,
    which corresponds to the period of a signal
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
    result = np.correlate(x, x, mode="full")
    return result[result.size // 2 :]


def autocorr_coef(x: np.ndarray, lags: tuple = (1, 200)) -> float:
    """
    Normalized covariance statistic between x(t) and x(t-n), where n is a lag in range 'lags'
    """
    min_lag = min(lags[0], len(x))
    max_lag = min(lags[1], len(x))
    results = []
    for lag in range(min_lag, max_lag + 1):
        results.append(np.corrcoef(x[:-lag], x[lag:])[:, 1][0])
    return np.array(results)
