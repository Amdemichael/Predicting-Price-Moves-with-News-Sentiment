# src/data_loader.py
import pandas as pd

def load_news_data(file_path: str) -> pd.DataFrame:
    """Load FNSPID news data into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    return df

def load_stock_data(file_path: str) -> pd.DataFrame:
    """Load stock price data into a pandas DataFrame."""
    df = pd.read_csv(file_path)

    # Use flexible handling for 'Date' column
    date_col = None
    for col in df.columns:
        if col.lower() == 'date':
            date_col = col
            break

    if not date_col:
        raise ValueError(f"No 'Date' column found in {file_path}. Columns: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    return df
