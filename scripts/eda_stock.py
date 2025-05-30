import pandas as pd
import matplotlib.pyplot as plt

def load_stock_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def describe_stock(df):
    print("\nStock Price Summary:")
    print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

def plot_stock(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.title('📉 Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()