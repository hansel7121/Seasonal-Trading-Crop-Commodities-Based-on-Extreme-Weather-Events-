import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# Import corn and coffee prices
corn_prices = yf.download("ZC=F", start="2015-01-01", end="2025-11-24", auto_adjust=True)["Close"]
if isinstance(corn_prices.columns, pd.MultiIndex):
    corn_prices.columns = corn_prices.columns.droplevel(1)

coffee_prices = yf.download("KC=F", start="2015-01-01", end="2025-11-24", auto_adjust=True)["Close"]
if isinstance(coffee_prices.columns, pd.MultiIndex):
    coffee_prices.columns = coffee_prices.columns.droplevel(1)

# Combine commodity prices into one dataframe
close_prices = pd.concat([corn_prices, coffee_prices], axis=1)
close_prices.columns = ["corn", "coffee"]
print(close_prices.head())


# Import corn and coffee buy signals
from corn.corn_roll_yield import get_corn_buy_signals
from coffee.coffee_roll_yield import get_coffee_buy_signals

corn_buy_signals = sorted(get_corn_buy_signals())
coffee_buy_signals = sorted(get_coffee_buy_signals())

df_corn = pd.DataFrame({'commodity type': 'corn', 'date': corn_buy_signals})
df_coffee = pd.DataFrame({'commodity type': 'coffee', 'date': coffee_buy_signals})

combined_df = pd.concat([df_corn, df_coffee], ignore_index=True)
combined_df['date'] = pd.to_datetime(combined_df['date'])
result_df = combined_df.sort_values(by='date').reset_index(drop=True)
print(result_df.head())


holding_period = 10

# Rolling yield functions
corn_estimated_drag = 0.02
coffee_estimated_drag = 0.015

def get_roll_months(current_date):
    month = current_date.month
    if month in [3, 5, 7, 9, 12]:
        return True
    return False

def get_estimated_drag(buy_date, contract_drag):
    roll_months = []
    for i in range(1, holding_period + 1):
        if get_roll_months(buy_date + pd.DateOffset(months=i)):
            roll_months.append(i)
    total_drag = 1
    for i in range(len(roll_months)):
        total_drag *= 1 - contract_drag
    return total_drag


# Backtesting

total_cash = 10000
cash_series = pd.Series(index=close_prices.index, data=total_cash, dtype=float)
corn_portfolio_value = pd.Series(index=close_prices.index, data=0, dtype=float)
coffee_portfolio_value = pd.Series(index=close_prices.index, data=0, dtype=float)
corn_busy_until_date = None
coffee_busy_until_date = None
corn_cash = 0
coffee_cash = 0
corn_shares = 0
coffee_shares = 0

for i in range(len(result_df)):
    buy_date = result_df['date'].iloc[i]
    buy_type = result_df["commodity type"].iloc[i]
    

    if buy_date not in close_prices.index:
        continue

    holding_corn = corn_busy_until_date is not None and buy_date < corn_busy_until_date
    holding_coffee = coffee_busy_until_date is not None and buy_date < coffee_busy_until_date


    if buy_type == "corn":
        if holding_corn:
            continue 


        # buy process
        total_drag = get_estimated_drag(buy_date, corn_estimated_drag)
        buy_price = close_prices.loc[buy_date, "corn"]

        current_cash = cash_series.loc[buy_date]
        
        if holding_coffee:   
            trade_cash = current_cash 
        else:
            trade_cash = current_cash / 2

        corn_shares = trade_cash / buy_price
        
        # sell process
        target_sell_date = buy_date + pd.DateOffset(months=holding_period)
        idx = close_prices.index.get_indexer([target_sell_date], method="nearest")[0]
        sell_date = close_prices.index[idx]

        if sell_date > close_prices.index[-1]:
            continue

        cash_series.loc[buy_date:] -= trade_cash

        corn_period_prices = close_prices.loc[buy_date:sell_date]["corn"]
        corn_portfolio_value.loc[buy_date:sell_date] = corn_shares * corn_period_prices 
        
        corn_sell_price = close_prices.loc[sell_date, "corn"]
        sell_proceeds = corn_shares * corn_sell_price * total_drag
        
        cash_series.loc[sell_date:] += sell_proceeds

        corn_portfolio_value.loc[sell_date:] = 0
        corn_busy_until_date = sell_date



    if buy_type == "coffee":
        if holding_coffee:
            continue

        total_drag = get_estimated_drag(buy_date, coffee_estimated_drag)
        buy_price = close_prices.loc[buy_date, "coffee"]
        
        current_cash = cash_series.loc[buy_date]

        if holding_corn:
            trade_cash = current_cash
        else:
            trade_cash = current_cash / 2

        coffee_shares = trade_cash / buy_price

        target_sell_date = buy_date + pd.DateOffset(months=holding_period)
        idx = close_prices.index.get_indexer([target_sell_date], method="nearest")[0]
        sell_date = close_prices.index[idx]

        if sell_date > close_prices.index[-1]:
            continue

        cash_series.loc[buy_date:] -= trade_cash

        coffee_period_prices = close_prices.loc[buy_date:sell_date]["coffee"]
        coffee_portfolio_value.loc[buy_date:sell_date] = coffee_shares * coffee_period_prices
        
        coffee_sell_price = close_prices.loc[sell_date, "coffee"]
        sell_proceeds = coffee_shares * coffee_sell_price * total_drag
        
        cash_series.loc[sell_date:] += sell_proceeds

        coffee_portfolio_value.loc[sell_date:] = 0
        coffee_busy_until_date = sell_date


# add up portfolio series
portfolio_value = corn_portfolio_value + coffee_portfolio_value + cash_series

final_portfolio_value = portfolio_value.iloc[-1]
total_return = (final_portfolio_value - 10000) / 10000
years = (close_prices.index[-1] - close_prices.index[0]).days / 365.25
annualized_return = (1 + total_return) ** (1 / years) - 1
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Annualized Return: {annualized_return * 100:.2f}%")
print(f"Total Return: {total_return * 100:.2f}%")



# plot portfolio value
plt.figure(figsize=(10, 5))
plt.plot(portfolio_value.index, portfolio_value, label="Portfolio Value")
plt.plot(corn_portfolio_value.index, corn_portfolio_value, label="Corn Portfolio Value")
plt.plot(coffee_portfolio_value.index, coffee_portfolio_value, label="Coffee Portfolio Value")
plt.title(f"Portfolio Value Over {holding_period} Months (Initial Cash: $10,000)")
plt.xlabel("Date")
plt.ylabel("Value ($)")
plt.grid(True)
plt.legend()
plt.show()
