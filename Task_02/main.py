import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load historical data
ticker = 'AAPL'  # You can change this to 'TSLA', 'GOOG', etc.
df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
print("Data Loaded:", df.shape)

# 2. Prepare the features and target
df = df.dropna()
df['Next_Close'] = df['Close'].shift(-1)

features = ['Open', 'High', 'Low', 'Volume']
df = df.dropna() 
X = df[features]
y = df['Next_Close']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train the model
model = LinearRegression() 
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# 6. Plot actual vs predicted close prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Close Price', color='blue')
plt.plot(y_pred, label='Predicted Close Price', color='red')
plt.title(f'{ticker} - Actual vs Predicted Close Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
