import yfinance as yf
import pandas as pd
import os

os.chdir(r"/Users/zhengxuanzhao/vs code/stock/data/source")

###### BTC data ######
# btc = yf.download("BTC-USD", start="2022-06-23", end="2025-06-23")
# btc["Prev Close"] = btc["Close"].shift(1)
# btc["Weekday"] = btc.index.weekday

# btc.to_csv("bitcoin_2022-06_2025-06.csv")
# print(btc.head())

###### NASDAQ data ######
nasdaq = yf.download("^IXIC", start="2022-06-23", end="2025-06-23")

nasdaq.to_csv("nasdaq_2022-06_2025-06.csv")

print(nasdaq.head())