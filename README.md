# Technical Analysis for Python

Technical Analysis (TA) is the study of price movements.

This package aims to provide an extensible framework for working with various TA tools. This includes, but is not limited to: candlestick patterns, technical overlays, technical indicators, statistical analysis, and automated strategy backtesting.

## Why Use This Library?

The Technical Analysis Library is still in its early days, but already has the following:

1. Recognition for Candlestick Patterns
2. Technical indicators
3. Technical overlays
4. Moving averages (including adaptive)
5. Automated backtests and strategies
6. Tools for timeseries analysis
7. Artificial data generation
8. A highly extensible framework to create custom indicators, backtests, and strategies


## Installation
Pypi link: https://pypi.org/project/technical-analysis/

```
pip install technical-analysis
```

## Overview
This package assumes you're working with pandas dataframes. If you're not familiar with pandas, see the docs here https://pandas.pydata.org/docs/

## Technical Overlays Example Usage
Technical Overlays are indicators placed directly on a chart.
These include moving averages and volatility bands.

### Moving Averages
```
>>> from technical_analysis import moving_average
>>> # simple moving average
>>> df["sma9"] = moving_average.sma(df.close, 9)
>>>
>>> # exponentially-weighted moving average
>>> df["ema9"] = moving_average.ema(df.close, period=9)
>>>
>>> # triangular moving average
>>> df["tma9"] = moving_average.tma(df.close, 9)
>>>
>>> # linearly-weighted moving average
>>> df["lwma9"] = moving_average.lwma(df.close, 9)
>>>
>>> # kaufman adaptive moving average
>>> df["kama9"] = moving_average.kama(df.close, 9, min_smoothing_constant=3, max_smoothing_constant=30)
>>>
>>> # wilder moving average
>>> df["wilder9"] = moving_average.wilder_ma(df.close, 9)
```

### Bands
```
>>> from technical_analysis import overlays
>>> # bollinger bands
>>> df["bband_lower"], df["bband_upper"] = overlays.bbands(df.close, period=20)
>>>
>>> # donchian bands
>>> df["dband_lower"], df["dband_upper"] = overlays.dbands(df.close, period=20)
>>>
>>> # keltner bands
>>> df["kband_lower"], df["kband_upper"] = overlays.kbands(df.high, df.low, df.close, period=20)
```

## Technical Indicators Example Usage
```
>>> from technical_analysis import indicators
>>> # average true range
>>> df["atr"] = indicators.atr(df.high, df.low, df.close, period=14)
>>>
>>> # relative strength index
>>> df["rsi"] = indicators.rsi(df.close, period=14)
>>>
>>> # Williams' %R
>>> df["perc_r"] = indicators.perc_r(df.high, df.low, df.close, period=14)
>>>
>>> # true strength index
>>> df["tsi"] = indicators.tsi(df.close, period1=25, period2=13)
>>>
>>> # TRIX
>>> df["trix"] = indicators.trix(df.close, period=15)
>>>
>>> # stochastic %k, %d (fast, slow, or full)
>>> df["stoch_k"], df["stoch_d"] = indicators.stochastic(df.high, df.low, df.close, period=14, perc_k_smoothing=3)
>>>
>>> # macd histogram
>>> df["macd_histogram"] = indicators.macd(df.close, return_histogram=True)
```

## Candlestick Pattern Recognition Example Usage
```
>>> from technical_analysis import candles
>>> df["gap_down"] = candles.is_gap_down(df.high, df.low, min_gap_size=0.003)
>>> df["gap_up"] = candles.is_gap_down(df.high, df.low, min_gap_size=0.003)
>>> df["long_body"] = candles.is_long_body(df.open, df.high, df.low, df.close, min_body_size=0.7)
>>> df["doji"] = candles.is_doji(df.open, df.high, df.low, df.close, relative_threshold=0.1)
>>> df["outside"] = candles.is_outside(df.high, df.low)
>>> df["inside"] = candles.is_inside(df.high, df.low)
>>> df["spinning_top"] = candles.spinning_top(df.open, df.high, df.low, df.close)
>>> df["marubozu"] = candles.is_marubozu(df.open, df.high, df.low, df.close, max_shadow_size=0.2)
>>> df["bullish_engulfing"] = candles.bullish_engulfing(df.open, df.high, df.low, df.close)
>>> df["bearish_engulfing"] = candles.bearish_engulfing(df.open, df.high, df.low, df.close)
```

## Automatic Backtesting Example Usage
The technical-analysis library comes with an extensible framework to backtest trading strategies.
```
>>> from technical_analysis.backtest import Backtest
>>> from technical_analysis.backtest.strategy import MovingAverageCrossover
>>> from technical_analysis import overlays
>>>
>>> # test an exponential moving average crossover strategy
>>> df["ema9"] = overlays.ema(df.close, period=9)
>>> df["ema20"] = overlays.ema(df.close, period=20)
>>> df = df.dropna().reset_index(drop=True)
>>> entry_criteria=[MovingAverageCrossover("ema9", "ema20", "bullish")]
>>> exit_criteria=[MovingAverageCrossover("ema9", "ema20", "bearish")]
>>> backtest = Backtest(entry_criteria, exit_criteria, max_positions=1, use_next_open=True)
>>> backtest.run(df)
>>> backtest.results
{'benchmark': 3.925821463626707,
 'strategy': 1.2970321301363634,
 'max_drawdown': -0.10934780434803487,
 'max_profit': 0.20020259422562683,
 'avg_return': 0.015817465001662968,
 'std_return': 0.057687131745236445,
 'returns': [0.01751003732275545, ...]}
```

## Timeseries Analysis
The technical-analysis library comes with useful timeseries analysis tools.
```
>>> from technical_analysis.stats import autocorr_coef, period
>>> # auto-correlation
>>> corr = autocorr_coef(df.close.pct_change())
>>> np.argsort(corr)[::-1][:10]
array([199,  62,  72,  71,  70,  69,  68,  67,  66,  65])
>>>
>>> # periodicity
>>> period(df.close, top_n=10)
array([ 1,  2,  5,  4,  3,  7, 16, 10, 25, 38])
>>>
>>> # hurst exponent
>>> hurst_exp(df.close)
0.3238867311092554
```

# Development

If you'd like to contribute please feel free to raise an issue or open a PR.

## Tests
```
pytest tests
```

## Linting
```
python -m black src/technical_analysis -l 120 --check 
```