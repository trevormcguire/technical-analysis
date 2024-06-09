import pandas as pd


class Strategy(object):
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return self.run(data)

    def run(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class MovingAverageCrossover(Strategy):
    """
    Parameters:
    ------------
    - `ma1_name`: str; column name of faster moving average
    - `ma2_name`: str; column name of slower moving average
    - `lookback_periods`: int; number of periods to look back to validate crossover
    - `kind`: str; one of ['bullish', 'bearish']
    """

    allowed_kinds = ["bullish", "bearish"]

    def __init__(
        self,
        ma1_name: str,
        ma2_name: str,
        kind: str,
        lookback_periods: int = 1,
    ):
        self.ma1_name = ma1_name
        self.ma2_name = ma2_name
        self.lookback_periods = lookback_periods

        if kind not in self.allowed_kinds:
            raise ValueError(f"`kind` must be one of {self.allowed_kinds}")
        self.kind = kind

    def _run_bullish(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        # faster > slower
        above = data[self.ma1_name] > data[self.ma2_name]

        # faster < slower
        prior_below = data[self.ma1_name].shift(lookback) < data[self.ma2_name].shift(lookback)
        return prior_below & above

    def _run_bearish(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        below = data[self.ma1_name] < data[self.ma2_name]

        prior_above = data[self.ma1_name].shift(lookback) > data[self.ma2_name].shift(lookback)
        return prior_above & below

    def run(self, data: pd.DataFrame) -> pd.Series:
        if len(data) <= self.lookback_periods:
            raise ValueError(f"`data` must have length > {self.lookback_periods}")

        if self.kind == "bullish":
            return self._run_bullish(data, self.lookback_periods)
        return self._run_bearish(data, self.lookback_periods)  # guaranteed by assertion in init
