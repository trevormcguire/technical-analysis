import os
import pandas as pd
from technical_analysis.indicators import trend_down, adx, tr, sma, ema, atr
from technical_analysis.candles import rising_three, rising_n
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("./src/pkn.csv")
df = df[["date", "open", "high", "low", "val", "vol"]]
df = df.rename(columns={"val": "close"})
df["date"] = pd.to_datetime(df["date"])
resample_agg = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "vol": "sum",
}
df = df.resample(rule="ME", on="date").agg(resample_agg)
df.reset_index(inplace=True)
#df=pd.read_csv('adx.csv', skiprows=1)


#go.Figure(data=go.Candlestick(x=df['date'],open=df['open'],high=df['high'],low=df['low'],close=df['close'])).show()
# px.bar(df, x='date',y=['open', 'high', 'low', 'close'],title='candles', barmode='group').show()

rt = rising_three(df['open'], df['high'], df['low'], df['close'], lookback=20)
rt = rising_n(df['open'], df['high'], df['low'], df['close'], n =6,lookback=20)


df['adx']=adx(df, output=['adx'])


high = df['high']
low = df['low']
price = df['close']

plusDM = pd.Series(0.0, index=high.index, dtype=float)
minusDM = pd.Series(0.0, index=high.index, dtype=float)

up_move = high - high.shift(1)
down_move = low.shift(1) - low
trs = tr(high, low, price)
trs=ema(trs, 14)

plusDM[up_move > down_move] = pd.DataFrame({'up':up_move,'zer':0}).max(axis=1)
minusDM[down_move > up_move] = pd.DataFrame({'down':down_move,'zer':0}).max(axis=1)

plusDI = ema(plusDM, period=14) / trs *100
plusDI.ffill(inplace=True)
minusDI = ema(minusDM, period=14) / trs *100
minusDI.ffill(inplace=True)

dx= 100 * abs((plusDI - minusDI)/(plusDI + minusDI))

adx= ema(dx,14)


df['exADX'] = adx
df['minusDI'] = minusDI
df['plusDI'] = plusDI
df['trs'] = trs
df['tr'] = tr(high, low, price)


#px.line(x=df.loc[~df['plusDM'].isna(),'date'],y=df.loc[~df['plusDM'].isna(),'plusDM'],title='plusDM').show()
px.line(x=df['date'],y=df['adx'],title='adx').show()

#fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

#axs[1].plot(df["ta"])
# axs[2].plot(df["obv"], label="obv")
# axs[2].plot(df["ad"], label="ad")
#axs[1].plot(df["adx"], label="adx")
#axs[1].plot(df["exADX"], label="my_ex_adx")
#axs[1].plot(df["minusDI"], label="minusDI")
#axs[1].plot(df["plusDI"], label="plusDI")
# axs[1].plot(df["trs"], label="trs")
# axs[1].plot(df["tr"], label="tr")
# #axs[1].plot(df["ADX"], label="ex_adx")
# # axs[2].plot(df["+DI"], label="+DI")
# # axs[2].plot(df["-DI"], label="-DI")
# # axs[2].plot(df["ema20"])
# # axs[2].plot(df["ema50"])
# # axs[2].plot(df["ema200"])
# axs[0].plot(df["close"])
# axs[1].legend()
# plt.show(block=True)
