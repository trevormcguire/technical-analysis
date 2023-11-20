import numpy as np


def random_walk(
    length: int,
    persistance: float = 0.5,
    mean: float = 0.0,
    std: float = 1.0,
    start: float = 0.0,
    lookback_range: tuple = (2, 100),
    distribution_size: int = 50000,
    seed: int = None,
) -> np.ndarray:
    """
    Generate an array of values that follows a random walk

    Params
    --------
        'length' -> int; size of the array generated
        'persistance' -> float, default 0.5
                         represents the probability that values will follow the trend
                         probability > 0.5 is a persistent random walk
                         probability < 0.5 is a mean-reverting random walk
        'mean' -> float; default 0
                  the mean of the distribution being sampled from
        'std' -> float; default 1
                 the standard deviation of the distribution being sampled from
        'start' -> float; starting price of random walk
        'distribution_size' -> int; default 50000
                               size of the distribution to sample from (remember: law of large numbers)
        'seed' -> random seed
    """

    def get_current_trend(returns: np.ndarray, lag: int) -> int:
        cumulative_returns = np.cumsum(returns)
        return np.sign(cumulative_returns[-1] - cumulative_returns[-lag])

    def sample_until_sign(dist: np.ndarray, sign: int):
        sample = np.random.choice(dist)
        if np.sign(sample) == sign:
            return sample
        return sample_until_sign(dist, sign)

    min_lookback, max_lookback = lookback_range
    lookback_periods = list(range(min_lookback, max_lookback))

    if seed is not None:
        np.random.seed(seed)
    dist = np.random.normal(loc=mean, scale=std, size=distribution_size)
    values = [np.random.choice(dist)]
    for _ in range(1, length):  # start at 1 because first value is calculated above
        proba = np.random.random()  # odds this point will follow the trend
        trend_lookback = np.random.choice(lookback_periods)
        if trend_lookback < len(values):
            trend = get_current_trend(values, trend_lookback)
            if proba < persistance:
                new_value = sample_until_sign(dist, trend)  # persistant
            else:
                new_value = sample_until_sign(dist, trend * -1)  # mean-reverting
        else:
            new_value = np.random.choice(dist)
        values.append(new_value)
    return np.cumsum(np.array(values)) + start
