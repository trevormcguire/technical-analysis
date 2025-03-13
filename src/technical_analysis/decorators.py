import functools
import pandas as pd


def df_ohlc_to_series(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], pd.DataFrame):
            df = args[0]
            args = args[1:]
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                raise ValueError("df must have columns 'open', 'high', 'low', 'close'")
            kwargs["open"] = df["open"]
            kwargs["close"] = df["close"]
            kwargs["high"] = df["high"]
            kwargs["low"] = df["low"]
        return func(*args, **kwargs)

    return wrapper

def df_price_to_series(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], pd.DataFrame):
            df = args[0]
            args = args[1:]
            if not {"close"}.issubset(df.columns):
                raise ValueError("df must have columns 'close'")
            kwargs["price"] = df["close"]
        return func(*args, **kwargs)

    return wrapper