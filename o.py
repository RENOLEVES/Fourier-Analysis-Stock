import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from ta.volatility import AverageTrueRange

# === Assume you already have this:
# btc_5min = <your loaded 5-min BTC/USD dataframe>
# If not, get it via yfinance:
# import yfinance as yf
# btc_5min = yf.download("BTC-USD", interval="5m", period="7d")

btc_5min = pd.read_csv(r"E:\IntelliJ IDEA 2020.2\IdeaProjects\stock\data\source\5m_btc.csv")

btc = btc_5min.copy()
btc.dropna(inplace=True)

btc = btc.tail(10000)

# === Indicator Settings (5-min version)
short_window = 12   # 12 x 5min = 1 hour
long_window = 48    # 48 x 5min = 4 hours
atr_window = 48     # 4-hour ATR
take_profit = 0.05  # +3% take profit
stop_loss = -0.05   # -1% stop loss

# === Indicators
btc["SMA_short"] = btc["close"].rolling(short_window).mean()
btc["SMA_long"] = btc["close"].rolling(long_window).mean()
btc["ATR"] = AverageTrueRange(high=btc["high"], low=btc["low"], close=btc["close"], window=atr_window).average_true_range()

btc["golden_cross"] = (btc["SMA_short"] > btc["SMA_long"]) & (btc["SMA_short"].shift(1) <= btc["SMA_long"].shift(1))
btc["death_cross"] = (btc["SMA_short"] < btc["SMA_long"]) & (btc["SMA_short"].shift(1) >= btc["SMA_long"].shift(1))

# === Portfolio Initialization
btc["position"] = 0.0
btc["cash"] = 1.0
btc["btc_held"] = 0.0
btc["portfolio_value"] = 1.0

entry_price = None
last_position = 0.0

# === QP Optimizer
def optimize_position(r_t, sigma2, prev_w, force_sell=False, force_buy=False):
    w = cp.Variable()

    # === 调整惩罚权重
    penalty_weight = 0.001 if abs(r_t) < 0.3 else 0.0

    # === 目标函数
    objective = cp.Minimize(0.5 * sigma2 * w**2 - r_t * w + penalty_weight * cp.abs(w - prev_w))

    # === 基本约束
    constraints = [w >= 0, w <= 1]

    # === 强制信号处理
    if force_sell:
        constraints.append(w == 0)
    elif force_buy:
        constraints.append(w >= 0.2)

    try:
        cp.Problem(objective, constraints).solve(solver=cp.SCS)
    except:
        return 0.0

    w_val = w.value if w.value is not None else 0.0

    # === 抑制极小仓位波动（可选）
    return w_val if w_val >= 0.1 else 0.0


# === Backtest
for i in range(1, len(btc)):
    prev = btc.iloc[i - 1]
    now = btc.iloc[i]
    price = now["close"]
    sigma2 = now["ATR"]**2 if not np.isnan(now["ATR"]) else 0

    unrealized_pnl = (price - entry_price) / entry_price if entry_price else 0

    # === Improved r_t: short-term momentum scaled by ATR
    momentum = np.log(now["close"] / btc.iloc[i - 12]["close"])
    sma_spread = now["SMA_short"] - now["SMA_long"]
    atr = now["ATR"]

    r_t = (100 * momentum + 0.5 * sma_spread / atr) if atr > 0 else 0

    force_sell = now["death_cross"] or unrealized_pnl < stop_loss or unrealized_pnl > take_profit
    force_buy = now["golden_cross"] and price > now["SMA_long"] + 0.2 * now["ATR"]

    w = optimize_position(r_t, sigma2, last_position, force_sell=force_sell, force_buy=force_buy)

    total_value = prev["btc_held"] * price + prev["cash"]
    btc_held = w * total_value / price
    cash = total_value - btc_held * price

    btc.at[btc.index[i], "btc_held"] = btc_held
    btc.at[btc.index[i], "cash"] = cash
    btc.at[btc.index[i], "position"] = w
    btc.at[btc.index[i], "portfolio_value"] = btc_held * price + cash

    if w > 0 and entry_price is None:
        entry_price = price
    elif w == 0:
        entry_price = None

    last_position = w

    print(f"{i}: price={price:.2f}, SMA_short={now['SMA_short']:.2f}, SMA_long={now['SMA_long']:.2f}, ATR={now['ATR']:.5f}, r_t={r_t:.5f}, force_buy={force_buy}, force_sell={force_sell}, w={w:.2f}")

# === Plot
btc["portfolio_value"].plot(figsize=(14, 6), label="Portfolio")
btc["close"].plot(secondary_y=True, alpha=0.3, label="BTC Price")
plt.title("5-Minute BTC Strategy with QP + Golden/Death Cross + ATR + Stop Loss")
plt.grid()
plt.legend()
plt.savefig("0.png")
plt.show()
