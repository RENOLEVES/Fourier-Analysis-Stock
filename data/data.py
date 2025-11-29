import ccxt
import pandas as pd
import time

exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '5m'
since = exchange.parse8601('2023-01-01T00:00:00Z')

all_data = []

while since < exchange.milliseconds():
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
    if not data:
        break
    since = data[-1][0] + 1
    all_data.extend(data)
    time.sleep(exchange.rateLimit / 1000)

df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

df.to_csv(r"E:\IntelliJ IDEA 2020.2\IdeaProjects\stock\data\source\5m_btc.csv")
print(df.head())