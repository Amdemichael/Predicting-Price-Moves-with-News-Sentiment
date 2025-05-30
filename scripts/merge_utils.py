import pandas as pd
import matplotlib.pyplot as plt

def merge_news_stock(stock_df, daily_sentiment_df):
    merged = pd.merge(stock_df, daily_sentiment_df, on='Date', how='left')
    merged.dropna(subset=['Avg_Sentiment', 'Daily_Return'], inplace=True)
    return merged

def plot_correlation(merged_df):
    correlation = merged_df['Avg_Sentiment'].corr(merged_df['Daily_Return'])
    print(f"📌 Pearson Correlation: {correlation:.4f}")

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Avg Sentiment', color='blue')
    ax1.plot(merged_df['Date'], merged_df['Avg_Sentiment'], color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Daily Return', color='orange')
    ax2.plot(merged_df['Date'], merged_df['Daily_Return'], color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('🔁 Sentiment vs. Stock Returns')
    plt.show()