"""
-------------
Detecting Japanese Candlestick Patterns in Python
-------------

    "Japanese Candlestick Patterns offer a quick picture into the psychology of
    short-term trading, studying the effect, not the cause" (Morris)

-------------
Candlestick Terms:
-------------
    1. Body
        > the open-close range
    2. Upper Shadow
        > the high
    3. Lower Shadow
        > the low
-------------
References:
-------------
    1. Candlestick Charting Exaplined, 3rd Edition (Morris) -- 
    2. CMT Level II Cirriculum (2019) -- Wiley
    3. https://school.stockcharts.com/doku.php?id=chart_analysis:candlestick_pattern_dictionary

"""
from technical_analysis.candles.plot import plot
from technical_analysis.candles.reversal import *
from technical_analysis.candles.continuation import *
from technical_analysis.candles.single import *

BEARISH_REVERSAL = {}

BEARISH = ["dark_cloud"]
BULLISH = ["bullish_engulfing"]

# continuation patterns
CONTINUATION = ["tasuki_gap"]