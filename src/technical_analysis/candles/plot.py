from typing import Tuple

import mplfinance as mpf
import pandas as pd


def plot(
    ohlc_data: pd.DataFrame,
    kind: str = "candle",
    volume: bool = False,
    figsize: Tuple[int] = (8, 6),
    style: str = "yahoo",
    **kwargs,
):
    """
    mplfinance wrapper to plot ticker data
    ----------

    Params:
    ----------
        'ohlc_data':
            a pandas DataFrame with columns [open, high, low, close], and optionally, volume

        'kind':
            Available options:
                - line
                - candle
                - renko
                - pnf (point and figure)

        'volume':
            whether or not to include volume bars

        'figsize':
            (height, width) tuple

        'style':
            Available options:
                - yahoo
                - binance
                - blueskies
                - brasil
                - charles
                - checkers
                - classic
                - default
                - ibd
                - kenan
                - mike
                - nightclouds
                - sas
                - starsandstripes

        **'kwargs'
            1. 'tight_layout' -> removes gaps at ends of graph
            2. 'title' -> adds a title
            3. 'xlabel' -> x-axis label
            4. 'ylabel' -> y-axis label
            5. 'savefig' -> filepath to save the image
    ----------
    """
    columns_to_keep = ["open", "high", "low", "close"]
    if volume:
        columns_to_keep += ["volume"]
    for column in columns_to_keep:
        assert column in ohlc_data.columns, f"'ohlc_data' must include column {column}"

    mpf.plot(
        ohlc_data[columns_to_keep],
        type=kind,
        volume=volume,
        show_nontrading=False,
        figsize=figsize,
        style=style,
        **kwargs,
    )
