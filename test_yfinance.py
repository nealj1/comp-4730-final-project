import os
import yfinance as yf


# Clear the screen on Unix/Linux
if os.name == 'posix':
    os.system('clear')
# Clear the screen on Windows
elif os.name == 'nt':
    os.system('cls')


data = yf.download("TSLA", start="2023-10-01", end="2023-10-30")
print(data)

tsla = yf.Ticker("TSLA")
print(tsla)

dividends = tsla.dividends
print(dividends)

splits = tsla.splits
print(splits)

actions = tsla.actions
print(actions)

financials = tsla.financials
print(financials)

# recommendations = tsla.recommendations
# print(recommendations)

options_data = tsla.options
print(options_data)

# info = tsla.info
# print(info)
