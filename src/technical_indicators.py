import pandas as pd
import talib
import matplotlib.pyplot as plt


def load_price_data(path):
    # Ensure the correct column name for parsing and index
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df


def calculate_indicators(df):
    """Calculate technical indicators using TA-Lib."""
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['EMA_10'] = talib.EMA(df['Close'], timeperiod=10)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    return df


def visualize_indicators(df, output_dir='plots'):
    """Visualize stock prices and technical indicators."""
    
    # Plot Close Price and SMA
    plt.figure(figsize=(12, 5))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['SMA_20'], label='20-Day SMA')
    plt.title('Close Price vs 20-Day SMA')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot RSI
    plt.figure(figsize=(12, 3))
    plt.plot(df['RSI'], label='RSI')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title('Relative Strength Index (RSI)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot MACD
    plt.figure(figsize=(12, 4))
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['MACD_signal'], label='MACD Signal')
    plt.title('MACD vs Signal Line')
    plt.legend()
    plt.grid(True)
    plt.show()


