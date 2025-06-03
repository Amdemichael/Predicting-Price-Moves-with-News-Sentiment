import pandas as pd

def load_news_data(file_path):
    """Load FNSPID news data into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    return df

def load_stock_data(file_path):
    """Load stock price data into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['Date'], utc=True)
    return df