import pandas as pd
import talib
import pynance as pn
import matplotlib.pyplot as plt


def load_price_data(path):
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    return df


def calculate_indicators(df):
    """Calculate technical indicators using TA-Lib."""
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['EMA_10'] = talib.EMA(df['Close'], timeperiod=10)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    return df

def visualize_indicators(df, indicators, output_dir='plots'):
    """Visualize stock prices and technical indicators."""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df['Close'], label='Close Price')
    plt.plot(indicators['SMA'], label='20-Day SMA')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(indicators['RSI'], label='RSI')
    plt.axhline(70, color='r', linestyle='--')
    plt.axhline(30, color='g', linestyle='--')
    plt.legend()
    plt.savefig(f'{output_dir}/technical_indicators.png')
    plt.close()
