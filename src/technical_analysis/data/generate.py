import numpy as np


def random_walk(length: int,     
                probability: float = 0.5,
                mean: float = 0.,
                std: float = 1.,
                distribution_size: int = 50000,
                seed: int = None) -> np.ndarray:
    """
    Generate an array of values that follow a random walk

    Params
    --------
        'length' -> int; size of the array generated
        'probability' -> float, default 0.5
                         represents the probability that values will follow the trend
                         probability > 0.5 is a persistent random walk
                         probability < 0.5 is a mean-reverting random walk
        'mean' -> float; default 0
                  the mean of the distribution being sampled from
        'std' -> float; default 1
                 the standard deviation of the distribution being sampled from
        'distribution_size' -> int; default 50000
                               size of the distribution to sample from (remember: law of large numbers)
        'seed' -> random seed
    """
    if seed is not None:
        np.random.seed(seed)
    dist = np.random.normal(loc=mean, scale=std, size=distribution_size)
    values = [np.random.choice(dist)]
    for n in range(1, length):  # start at 1 because first value is calculated above
        new_value = np.random.choice(dist)
        values.append(new_value)
    return np.array(values)

    