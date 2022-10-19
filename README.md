# Technical Analysis for Python

## Example Usage
```
import pandas as pd

from technical_analysis import candles
from technical_analysis import overlays
from technical_analysis import indicators


spy = pd.read_csv(filepath)

# exponentially-weighted moving average
spy["ema9"] = overlays.ema(spy.close, 9)
spy["ema20"] = overlays.ema(spy.close, 20)
spy["ema50"] = overlays.ema(spy.close, 50)

# triangular moving average
spy["tma9"] = overlays.tma(spy.close, 9)
spy["tma20"] = overlays.tma(spy.close, 20)
spy["tma50"] = overlays.tma(spy.close, 50)

# linearly-weighted moving average
spy["lwma9"] = overlays.lwma(spy.close, 9)
spy["lwma20"] = overlays.lwma(spy.close, 20)
spy["lwma50"] = overlays.lwma(spy.close, 50)

# kaufman adaptive moving average
spy["kama9"] = overlays.kama(spy.close, 9, min_smoothing_constant=3, max_smoothing_constant=30)
spy["kama20"] = overlays.kama(spy.close, 20, min_smoothing_constant=3, max_smoothing_constant=30)
spy["kama50"] = overlays.kama(spy.close, 50, min_smoothing_constant=3, max_smoothing_constant=30)

# bollinger band
spy["bband_lower"], spy["bband_upper"] = overlays.bbands(spy.close, period=20)
# donchian band
spy["dband_lower"], spy["dband_upper"] = overlays.dbands(spy.close, period=20)
# keltner band
spy["kband_lower"], spy["kband_upper"] = overlays.kbands(spy.high, spy.low, spy.close, period=20)
# average true range
spy["atr"] = indicators.atr(spy.high, spy.low, spy.close, period=14)
# relative strength index
spy["rsi"] = indicators.rsi(spy.close, period=14)
# Williams' %R
spy["perc_r"] = indicators.perc_r(spy.high, spy.low, spy.close, period=14)
# true strength index
spy["tsi"] = indicators.tsi(spy.close, period1=25, period2=13)
# TRIX
spy["trix"] = indicators.trix(spy.close, period=15)
# stochastic %k, %d (fast, slow, or full)
spy["stoch_k"], spy["stoch_d"] = indicators.stochastic(spy.high, spy.low, spy.close, period=14, perc_k_smoothing=3)
# macd histogram
spy["macd_histogram"] = indicators.macd(spy.close, return_histogram=True)
spy.dropna(inplace=True)
```