from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BacktestBase(object):
    """
    Backtesting Base Object
    ------------

    Parameters:
    ------------
        'entry_criteria':
                a list or tuple of entry criteria in the following formats:
                    1.  String;
                        '{column_name} {condition}'
                        where {condition} can be a value or another column name
                        Examples:
                        ---------
                            'rsi > 70'
                            'sma9 < sma20'

                    2. Callable;
                       An object that takes in a pandas DataFrame and returns
                       a boolean pandas Series
                    
                    3. Tuple;
                       A tuple of conditions that are merged with a logical 'or' operation
                       Examples:
                       ---------
                           ('rsi > 70', 'sma9 < sma20', some_callable_fn)  # merged with 'or'
                           equivelent to pandas:
                                (data['rsi'] > 70) | (data['sma9'] < data['sma20']) | (some_callable_fn(data))
                           in english:
                                'rsi is greater than 70,
                                OR sma9 is less than sma20,
                                OR the result of the function some_callable_fn'

                    4. List;
                       A list of conditions that are merged with a logical 'and' operation
                       Examples:
                       ---------
                           ['rsi > 70', 'sma9 < sma20', some_callable_fn]  # merged with 'and'
                           equivelent to pandas:
                                (data['rsi'] > 70) & (data['sma9'] < data['sma20']) & (some_callable_fn(data))
                           in english:
                                'rsi is greater than 70,
                                AND sma9 is less than sma20,
                                AND the result of the function some_callable_fn'

        'exit_criteria'-> a list or tuple of exit criteria in same format as 'entry_criteria'
    
    Example Usage:
    -------------
        >>> def criteria_fn(data: pd.DataFrame):
        ...     # test if rsi14 has a positive slope over the past 10 periods
        ...     return data["rsi14"] > data["rsi14"].shift(10)
        ...
        >>> class HigherTrend(object):
        ...     # abstracts 'criteria_fn' to use any column and lag
        ...     def __init__(self, column_name: str, lag: int):
        ...         self.column_name = column_name
        ...         self.lag = lag
        ...     
        ...     def __call__(self, data):
        ...         col = data[self.column_name]
        ...         return col > col.shift(self.lag) 
        ...
        >>> entry_criteria = ["rsi14 > 70",
        ...                   criteria_fn]  # list means logical 'and' operation
        ...
        >>> # we can also use nested tuples and lists for nested logical operations
        >>> exit_criteria = (
        ...     "rsi14 < rsi28",
        ...     [HigherTrend(column_name="atr", lag=20),
        ...      HigherTrend(column_name="rsi14", lag=50)],
        ... ) # tuple means logical 'or' operation
        ...
        >>> spy["rsi14"] = indicators.rsi(spy.close, period=14)
        >>> spy["rsi28"] = indicators.rsi(spy.close, period=28)
        >>> backtest = BacktestBase(entry_criteria, exit_criteria)
        >>> backtest.run(spy)

    """
    def __init__(self,
                 entry_criteria: Union[list, tuple],
                 exit_criteria: Union[list, tuple]):

        assert type(entry_criteria) in [list, tuple], \
            f"Entry criteria type must be a list or tuple"
        assert type(exit_criteria) in [list, tuple], \
            "Exit criteria type must be a list or tuple"
   
        for condition in entry_criteria:
            assert isinstance(condition, str) \
                   or isinstance(condition, list) \
                   or isinstance(condition, tuple) \
                   or isinstance(condition, Callable), \
                   "Entry criteria conditions must a list, tuple, str, or callable"

        for condition in exit_criteria:
            assert isinstance(condition, str) \
                   or isinstance(condition, list) \
                   or isinstance(condition, tuple) \
                   or isinstance(condition, Callable), \
                   "Exit criteria conditions must a list, tuple, str, or callable"
    
        self.entry_criteria = entry_criteria
        self.exit_criteria = exit_criteria
        self.feature_columns = []
        self.results = {}

    def __repr__(self):
        return f"""Backtest(num_entry_conditions={len(self.entry_criteria)},
         n_exit_conditions={len(self.exit_criteria)},
         has_results={len(self.results) > 0})"""

    def _parse_criteria(self, condition: str) -> str:
        """
        Parses and evaluates string conditions
        """
        column, operator, value = condition.strip().split(" ")
        assert column in self.feature_columns, \
            f"'data' must have columns corresponding to criteria. Column '{column}' not found."
        if not value.isnumeric():  # assume its a column name
            assert value in self.feature_columns, \
                f"'data' must have columns corresponding to criteria. Column '{value}' not found."
            value = f"data['{value}']"
        condition = f"data['{column}'] {operator} {value}"
        return condition

    def _apply_criteria(self,
                        data: pd.DataFrame,
                        exit: bool = False,
                        criteria: Union[list, tuple] = None) -> pd.Series:
        """
        Recursively applies criteria specified
        Handles the cases whereby:
            - the criteria is a list or tuple
            - the conditions are a list, tuple, str, or callable
        """
        if not self.feature_columns:
            self.feature_columns = list(data.columns)
        criteria_states = []

        if criteria is None:
            criteria = self.exit_criteria if exit else self.entry_criteria  # assume entry if not exit

        if isinstance(criteria, tuple):
            logical = "or"
        elif isinstance(criteria, list):
            logical = "and"
        else:
            raise ValueError("'criteria' must be a list or tuple")

        for c in criteria:
            # criteria in tuples are recursively merged with a logical 'or' operation
            if isinstance(c, tuple):
                criteria_states.append(self._apply_criteria(data, exit=exit, criteria=c))
            # criteria in lists are recursively merged with a logical 'and' operation
            elif isinstance(c, list):
                criteria_states.append(self._apply_criteria(data, exit=exit, criteria=c))
            # Callable must return a single boolean pd.Series
            elif isinstance(c, Callable):
                callable_result = c(data)
                assert isinstance(callable_result, pd.Series) \
                       and callable_result.dtype == bool, \
                        "if using a callable criteria, the callable must return one pandas boolean Series"
                criteria_states.append(callable_result)
            # string conditions are parsed and then evalulated
            else:
                condition = self._parse_criteria(c)
                criteria_states.append(eval(condition))

        if logical == "or":
            return pd.concat(criteria_states, axis=1).any(axis=1)  # column-wise logical 'or'
        return pd.concat(criteria_states, axis=1).all(axis=1)  # column-wise logical 'and'

    def run(self, data: pd.DataFrame):
        if not self.feature_columns:
            self.feature_columns = list(data.columns)

        entry = self._apply_criteria(data, exit=False)  # entry
        exit = self._apply_criteria(data, exit=True)
        return entry

    def save(self):
        pass

    def plot(self):
        pass
