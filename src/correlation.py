import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

def calculate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sentiment polarity using TextBlob and categorize."""
    df['sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_category'] = df['sentiment'].apply(
        lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'
    )
    return df

def aggregate_daily_sentiment(df: pd.DataFrame, by_stock=False) -> pd.DataFrame:
    """Aggregate sentiment daily, optionally by stock."""
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    group_cols = ['date_only']
    if by_stock and 'stock' in df.columns:
        group_cols.append('stock')
    daily_sentiment = df.groupby(group_cols)['sentiment'].mean().reset_index()
    return daily_sentiment

def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns for each stock."""
    if 'stock' in df.columns:
        df = df.sort_values(['stock', 'date'])
        df['daily_return'] = df.groupby('stock')['Close'].pct_change()
    else:
        df = df.sort_values('date')
        df['daily_return'] = df['Close'].pct_change()
    return df

def merge_sentiment_returns(sentiment_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily sentiment and daily stock returns on date and stock if applicable."""
    returns_df = returns_df.reset_index(drop=True)
    returns_df['date_only'] = returns_df['date'].dt.date
    sentiment_df['date_only'] = pd.to_datetime(sentiment_df['date_only']).dt.date

    if 'stock' in sentiment_df.columns and 'stock' in returns_df.columns:
        merged = pd.merge(returns_df, sentiment_df, on=['date_only', 'stock'], how='inner')
    else:
        merged = pd.merge(returns_df, sentiment_df, on='date_only', how='inner')
    return merged

def compute_correlation(merged_df: pd.DataFrame):
    """Compute Pearson correlation between sentiment and daily return."""
    corr, p_value = pearsonr(merged_df['sentiment'], merged_df['daily_return'])
    return corr, p_value

def plot_sentiment_vs_returns(merged_df: pd.DataFrame, title_suffix=""):
    """Plot sentiment vs daily return with regression line."""
    plt.figure(figsize=(10,6))
    sns.regplot(x='sentiment', y='daily_return', data=merged_df, scatter_kws={'alpha':0.6})
    plt.title(f"Correlation between News Sentiment and Stock Daily Returns {title_suffix}")
    plt.xlabel("Average Daily Sentiment Score")
    plt.ylabel("Daily Stock Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def rolling_correlation_analysis(merged_df: pd.DataFrame, window=7):
    """Calculate and plot rolling correlation of sentiment and returns."""
    if 'stock' in merged_df.columns:
        for stock in merged_df['stock'].unique():
            subset = merged_df[merged_df['stock'] == stock].copy()
            subset = subset.sort_values('date_only')
            subset['rolling_corr'] = subset['sentiment'].rolling(window).corr(subset['daily_return'])
            plt.plot(subset['date_only'], subset['rolling_corr'], label=stock)
        plt.title(f"Rolling {window}-Day Correlation between Sentiment and Returns by Stock")
    else:
        merged_df = merged_df.sort_values('date_only')
        merged_df['rolling_corr'] = merged_df['sentiment'].rolling(window).corr(merged_df['daily_return'])
        plt.plot(merged_df['date_only'], merged_df['rolling_corr'], label='All Stocks')
        plt.title(f"Rolling {window}-Day Correlation between Sentiment and Returns")

    plt.xlabel("Date")
    plt.ylabel("Rolling Correlation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    


def lagged_sentiment_analysis(merged_df: pd.DataFrame, max_lag=5):
    """Calculate correlations with lagged sentiment to detect lead-lag effects."""
    results = []
    df = merged_df.copy()
    if 'stock' in df.columns:
        stocks = df['stock'].unique()
    else:
        stocks = [None]

    for lag in range(1, max_lag+1):
        if stocks[0] is not None:
            for stock in stocks:
                sub = df[df['stock'] == stock].copy()
                sub = sub.sort_values('date_only')
                sub[f'sentiment_lag_{lag}'] = sub['sentiment'].shift(lag)
                sub = sub.dropna(subset=[f'sentiment_lag_{lag}', 'daily_return'])
                if len(sub) > 2:
                    corr, pval = pearsonr(sub[f'sentiment_lag_{lag}'], sub['daily_return'])
                    results.append({'stock': stock, 'lag': lag, 'correlation': corr, 'p_value': pval})
        else:
            df = df.sort_values('date_only')
            df[f'sentiment_lag_{lag}'] = df['sentiment'].shift(lag)
            sub = df.dropna(subset=[f'sentiment_lag_{lag}', 'daily_return'])
            if len(sub) > 2:
                corr, pval = pearsonr(sub[f'sentiment_lag_{lag}'], sub['daily_return'])
                results.append({'stock': None, 'lag': lag, 'correlation': corr, 'p_value': pval})

    return pd.DataFrame(results)