import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

corn_df = pd.read_csv("crops_data/iowa_corn_temps_10y.csv", index_col="Date", parse_dates=True)
coffee_df = pd.read_csv("crops_data/varginha_coffee_temps_10y.csv", index_col="Date", parse_dates=True)
soybean_df = pd.read_csv("crops_data/iowa_soybean_temps_10y.csv", index_col="Date", parse_dates=True)

corn_prices = yf.download("ZC=F", start="2015-01-01", end="2025-11-24", auto_adjust=True)["Close"]
if isinstance(corn_prices.columns, pd.MultiIndex):
    corn_prices.columns = corn_prices.columns.droplevel(1)

coffee_prices = yf.download("KC=F", start="2015-01-01", end="2025-11-24", auto_adjust=True)["Close"]
if isinstance(coffee_prices.columns, pd.MultiIndex):
    coffee_prices.columns = coffee_prices.columns.droplevel(1)

soybean_prices = yf.download("ZS=F", start="2015-01-01", end="2025-11-24", auto_adjust=True)["Close"]
if isinstance(soybean_prices.columns, pd.MultiIndex):
    soybean_prices.columns = soybean_prices.columns.droplevel(1)

#combine commodity prices into one dataframe
close_prices = pd.concat([corn_prices, coffee_prices, soybean_prices], axis=1)
close_prices.columns = ["corn", "coffee", "soybean"]
print(close_prices.head())


# Import the helper functions (not the variables)
from corn import get_corn_buy_signals
from coffee import get_coffee_buy_signals
from soybeans import get_soybean_buy_signals

#sorting buy signals for each commodity then creating combined dataframe for backtesting
corn_buy_signals = sorted(get_corn_buy_signals())
coffee_buy_signals = sorted(get_coffee_buy_signals())
soybean_buy_signals = sorted(get_soybean_buy_signals())

df_corn = pd.DataFrame({'commodity type': 'corn', 'date': corn_buy_signals})
df_coffee = pd.DataFrame({'commodity type': 'coffee', 'date': coffee_buy_signals})
df_soybean = pd.DataFrame({'commodity type': 'soybean', 'date': soybean_buy_signals})

combined_df = pd.concat([df_corn, df_coffee, df_soybean], ignore_index=True)
combined_df['date'] = pd.to_datetime(combined_df['date'])
result_df = combined_df.sort_values(by='date').reset_index(drop=True)
print(result_df.head())


cash = 10000
portfolio_value = pd.Series(index=close_prices.index, data=cash, dtype=float)
corn_busy_until_date = None
coffee_busy_until_date = None
soybean_busy_until_date = None

for buy_date in result_df['date']:
    print(buy_date)
    """
    if buy_date not in close_prices.index:
        continue

    if busy_until_date is not None and buy_date < busy_until_date:
        continue

    buy_price = prices.loc[buy_date]["Close"]
    corn_shares = cash / buy_price
    target_sell_date = buy_date + pd.DateOffset(months=holding_period)
    idx = prices.index.get_indexer([target_sell_date], method="nearest")[0]
    sell_date = prices.index[idx]

    if sell_date > prices.index[-1]:
        break

    period_prices = prices.loc[buy_date:sell_date]["Close"]
    portfolio_value.loc[buy_date:sell_date] = corn_shares * period_prices
    sell_price = prices.loc[sell_date]["Close"]
    cash = corn_shares * sell_price
    portfolio_value.loc[sell_date:] = cash
    busy_until_date = sell_date

total_return = (cash - 10000) / 10000
years = (prices.index[-1] - prices.index[0]).days / 365.25
annualized_return = (1 + total_return) ** (1 / years) - 1
print(f"Final Portfolio Value: ${cash:.2f}")
print(f"Annualized Return: {annualized_return * 100:.2f}%")

print(cash, annualized_return, portfolio_value)
"""