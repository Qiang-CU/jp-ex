import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np


ticker = yf.Ticker("AAPL")

balance_sheet = ticker.quarterly_balance_sheet
cash_flow = ticker.quarterly_cashflow
income_statement = ticker.quarterly_income_stmt

total_assets = balance_sheet.loc['Total Assets']
total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest']
equity = balance_sheet.loc['Stockholders Equity']

# deal with missing value
total_assets_cleaned = total_assets.dropna()
total_liabilities_cleaned = total_liabilities.dropna()
equity_cleaned = equity.dropna()

# scale data
scaler = MinMaxScaler(feature_range=(0, 1))
total_assets_scaled = scaler.fit_transform(total_assets_cleaned.values.reshape(-1, 1))
total_liabilities_scaled = scaler.fit_transform(total_liabilities_cleaned.values.reshape(-1, 1))
equity_scaled = scaler.fit_transform(equity_cleaned.values.reshape(-1, 1))

# stack features
data = np.column_stack([total_assets_scaled, total_liabilities_scaled, equity_scaled])

# split train/test set
train_size = int(len(data) * 0.6)
train_data = data[:train_size]
test_data = data[train_size:]