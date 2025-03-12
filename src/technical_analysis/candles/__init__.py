"""
-------------
Detect Japanese Candlestick Patterns in Python
-------------

    "Japanese Candlestick Patterns offer a quick picture into the psychology of
    short-term trading, studying the effect, not the cause" (Morris)

-------------
Candlestick Terms:
-------------
    1. Body
        > price action between (open, close)
    2. Upper Shadow
        > price action between (max(open, close), high)
    3. Lower Shadow
        > price action between (min(open, close), low)
-------------
References:
-------------
    1. Candlestick Charting Exaplined, 3rd Edition (Morris)
    2. CMT Level II Cirriculum (2019) -- Wiley
    3. https://school.stockcharts.com/doku.php?id=chart_analysis:candlestick_pattern_dictionary
-------------
"""

from technical_analysis.candles.single import *  # noqa
from technical_analysis.candles.reversal import *  # noqa
from technical_analysis.candles.continuation import *  # noqa
