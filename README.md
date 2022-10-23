# Technical Analysis for Python

Technical Analysis (TA) is the study of price movements.

This package aims to provide an extensible framework for working with various TA tools. This includes, but is not limited to: candlestick patterns, technical overlays, technical indicators, statistical analysis, and strategy backtesting.

## Why Use This Library?

The Technical Analysis Library is still in its early days, but already has the following:

1. Recognition for 30+ Candlestick Patterns
3. 10+ technical indicators
4. 10+ moving average types (including adaptive)
5. Technical overlays
6. Automated backtests and strategies
7. Statistical tools to analyze price action
8. A highly extensible framework to create custom indicators, backtests, and strategies


## Installation
Pypi link: https://pypi.org/project/technical-analysis/

```
pip install technical-analysis
```

## Technical Overlays Example Usage
Technical Overlays are indicators placed directly on a chart.
These include moving averages and volatility bands.
```
>>> import pandas as pd
>>> from technical_analysis import candles
>>> from technical_analysis import overlays
>>> from technical_analysis import indicators
>>>
>>> spy = pd.read_csv(filepath)
>>>
>>> # exponentially-weighted moving average
>>> spy["ema9"] = overlays.ema(spy.close, 9)
>>> spy["ema20"] = overlays.ema(spy.close, 20)
>>> spy["ema50"] = overlays.ema(spy.close, 50)
>>>
>>> # triangular moving average
>>> spy["tma9"] = overlays.tma(spy.close, 9)
>>> spy["tma20"] = overlays.tma(spy.close, 20)
>>> spy["tma50"] = overlays.tma(spy.close, 50)
>>>
>>> # linearly-weighted moving average
>>> spy["lwma9"] = overlays.lwma(spy.close, 9)
>>> spy["lwma20"] = overlays.lwma(spy.close, 20)
>>> spy["lwma50"] = overlays.lwma(spy.close, 50)
>>>
>>> # kaufman adaptive moving average
>>> spy["kama9"] = overlays.kama(spy.close, 9, min_smoothing_constant=3, max_smoothing_constant=30)
>>> spy["kama20"] = overlays.kama(spy.close, 20, min_smoothing_constant=3, max_smoothing_constant=30)
>>> spy["kama50"] = overlays.kama(spy.close, 50, min_smoothing_constant=3, max_smoothing_constant=30)
>>>
>>> # bollinger bands
>>> spy["bband_lower"], spy["bband_upper"] = overlays.bbands(spy.close, period=20)
>>>
>>> # donchian bands
>>> spy["dband_lower"], spy["dband_upper"] = overlays.dbands(spy.close, period=20)
>>>
>>> # keltner bands
>>> spy["kband_lower"], spy["kband_upper"] = overlays.kbands(spy.high, spy.low, spy.close, period=20)

```

## Technical Indicators Example Usage
```
>>> # average true range
>>> spy["atr"] = indicators.atr(spy.high, spy.low, spy.close, period=14)
>>>
>>> # relative strength index
>>> spy["rsi"] = indicators.rsi(spy.close, period=14)
>>>
>>> # Williams' %R
>>> spy["perc_r"] = indicators.perc_r(spy.high, spy.low, spy.close, period=14)
>>>
>>> # true strength index
>>> spy["tsi"] = indicators.tsi(spy.close, period1=25, period2=13)
>>>
>>> # TRIX
>>> spy["trix"] = indicators.trix(spy.close, period=15)
>>>
>>> # stochastic %k, %d (fast, slow, or full)
>>> spy["stoch_k"], spy["stoch_d"] = indicators.stochastic(spy.high, spy.low, spy.close, period=14, perc_k_smoothing=3)
>>>
>>> # macd histogram
>>> spy["macd_histogram"] = indicators.macd(spy.close, return_histogram=True)
```

## Candlestick Pattern Recognition Example Usage
```
>>> spy["gap_down"] = candles.is_gap_down(spy.high, spy.low, min_gap_size=0.003)
>>> spy["gap_up"] = candles.is_gap_down(spy.high, spy.low, min_gap_size=0.003)
>>> spy["long_body"] = candles.is_long_body(spy.open, spy.high, spy.low, spy.close, min_body_size=0.7)
>>> spy["doji"] = candles.is_doji(spy.open, spy.high, spy.low, spy.close, relative_threshold=0.1)
>>> spy["outside"] = candles.is_outside(spy.high, spy.low)
>>> spy["inside"] = candles.is_inside(spy.high, spy.low)
>>> spy["spinning_top"] = candles.spinning_top(spy.open, spy.high, spy.low, spy.close)
>>> spy["marubozu"] = candles.is_marubozu(spy.open, spy.high, spy.low, spy.close, max_shadow_size=0.2)
>>> spy["dark_cloud"] = candles.dark_cloud(spy.open,
...                                        spy.high,
...                                        spy.low,
...                                        spy.close,
...                                        min_body_size=0.65,
...                                        new_high_periods=30)
...
>>> spy["bullish_engulfing"] = candles.bullish_engulfing(spy.open, spy.high, spy.low, spy.close)
>>> spy["bearish_engulfing"] = candles.bearish_engulfing(spy.open, spy.high, spy.low, spy.close)

```

## Automatic Backtesting Example Usage
The technical-analysis library comes with an extensible framework to backtest trading strategies.
```
>>> import pandas as pd
>>> from technical_analysis.backtest import Backtest
>>> from technical_analysis.backtest.strategy import MovingAverageCrossover
>>>
>>> df = pd.read_csv(filepath)
>>> # test an exponential moving average crossover strategy
>>> df["ema9"] = overlays.ema(spy.close, period=9)
>>> df["ema20"] = overlays.ema(spy.close, period=20)
>>> 
>>> backtest = Backtest(entry_criteria=[MovingAverageCrossover("sma9", "sma20", "bullish")],
...                     exit_criteria=[MovingAverageCrossover("sma9", "sma20", "bearish")])
...
>>> backtest.run(df)
>>> backtest.results
{'benchmark': 5.56607215019379,
 'strategy': 1.39245960527215,
 'max_drawdown': -0.10934780434711658,
 'max_profit': 0.2002025942258056,
 'avg_return': 0.01832183691147566,
 'std_return': 0.05842269396587131,
 'returns': [0.1079530513709391, ...]
```
