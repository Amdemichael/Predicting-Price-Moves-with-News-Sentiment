import pandas as pd
import matplotlib.pyplot as plt


def merge_and_correlate(news_sentiment, stock_data):
    stock_data = stock_data.copy()
    stock_data['date'] = stock_data.index.date
    stock_data['daily_return'] = stock_data['Close'].pct_change()

    merged = pd.merge(news_sentiment, stock_data[['date', 'daily_return']], on='date')
    print("Correlation matrix:")
    print(merged.corr())

    merged.plot.scatter(x='sentiment', y='daily_return', title='Sentiment vs Stock Return')
    plt.show()