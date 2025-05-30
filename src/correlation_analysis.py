import pandas as pd
from scipy.stats import pearsonr

def align_dates(news_df, stock_df):
    """Align news and stock data by date."""
    news_df['date_only'] = news_df['date'].dt.date
    stock_df['date_only'] = stock_df['date'].dt.date
    merged_df = pd.merge(news_df, stock_df, on=['date_only', 'stock'], how='inner')
    return merged_df

def calculate_daily_returns(stock_df):
    """Calculate daily percentage returns."""
    stock_df['daily_return'] = stock_df['Close'].pct_change()
    return stock_df

def correlation_analysis(news_df, stock_df):
    """Calculate correlation between sentiment and stock returns."""
    merged_df = align_dates(news_df, stock_df)
    merged_df = calculate_daily_returns(merged_df)
    daily_sentiment = merged_df.groupby('date_only')['sentiment'].mean()
    daily_returns = merged_df.groupby('date_only')['daily_return'].mean()
    correlation, p_value = pearsonr(daily_sentiment, daily_returns)
    return correlation, p_value