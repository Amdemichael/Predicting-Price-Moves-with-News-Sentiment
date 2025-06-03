from textblob import TextBlob
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Tuple, Dict

# External module
from data_loader import load_news_data, load_stock_data


def calculate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sentiment polarity using TextBlob and categorize."""
    df['sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_category'] = df['sentiment'].apply(
        lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'
    )
    return df


def align_and_merge(news_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    """Align news and stock data by date and merge."""
    news_df['date_only'] = news_df['date'].dt.date
    stock_df['date_only'] = stock_df['date'].dt.date
    return pd.merge(news_df, stock_df[['date_only', 'Close']], on='date_only', how='inner')


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from Close prices."""
    df['daily_return'] = df['Close'].pct_change()
    return df


def correlation_analysis(news_df: pd.DataFrame, stock_df: pd.DataFrame) -> Tuple[Tuple[float, float], pd.DataFrame]:
    """Compute Pearson correlation between sentiment and daily returns."""
    merged_df = align_and_merge(news_df, stock_df)
    merged_df = calculate_daily_returns(merged_df)

    daily_metrics = merged_df.groupby('date_only').agg({
        'sentiment': 'mean',
        'daily_return': 'last'
    }).dropna()

    if daily_metrics.shape[0] < 2:
        raise ValueError("Insufficient data points for correlation analysis")

    correlation = pearsonr(daily_metrics['sentiment'], daily_metrics['daily_return'])
    return correlation, daily_metrics


def plot_correlation_matrix(data: pd.DataFrame, ticker: str) -> None:
    """Plot heatmap of correlation matrix."""
    corr_matrix = data.corr(method='pearson')
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix for {ticker}')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_labels: pd.Series, predicted_labels: pd.Series, ticker: str) -> None:
    """Plot confusion matrix comparing sentiment and returns."""
    cm = confusion_matrix(true_labels, predicted_labels, labels=[1, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Gain', 'Loss'])
    disp.plot()
    plt.title(f'Confusion Matrix for {ticker}')
    plt.tight_layout()
    plt.show()


def run_analysis(news_file: str, stock_files: Dict[str, str]) -> None:
    """Main analysis pipeline."""
    news_df = load_news_data(news_file)
    news_df = calculate_sentiment(news_df)

    for ticker, stock_file in stock_files.items():
        print(f"🔍 Analyzing {ticker}...")
        stock_df = load_stock_data(stock_file)
        stock_news_df = news_df[news_df['stock'] == ticker]

        try:
            (corr_coef, p_value), daily_metrics = correlation_analysis(stock_news_df, stock_df)

            print(f"📈 Correlation between sentiment and daily returns for {ticker}:")
            print(f"   Coefficient: {corr_coef:.4f}, P-value: {p_value:.4f}\n")

            plot_correlation_matrix(daily_metrics[['sentiment', 'daily_return']], ticker)

            true_labels = (daily_metrics['daily_return'] > 0).astype(int)
            predicted_labels = (daily_metrics['sentiment'] > 0).astype(int)

            plot_confusion_matrix(true_labels, predicted_labels, ticker)

        except ValueError as e:
            print(f"⚠️ Skipping {ticker} — {str(e)}\n")


if __name__ == "__main__":
    stock_files = {
        'AAPL': '../data/AAPL_historical_data.csv',
        'AMZN': '../data/AMZN_historical_data.csv',
        'GOOG': '../data/GOOG_historical_data.csv',
        'META': '../data/META_historical_data.csv',
        'MSFT': '../data/MSFT_historical_data.csv',
        'NVDA': '../data/NVDA_historical_data.csv',
        'TSLA': '../data/TSLA_historical_data.csv'
    }

    run_analysis('../raw_analyst_ratings.csv', stock_files)
