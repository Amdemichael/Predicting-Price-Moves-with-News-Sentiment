from textblob import TextBlob
import pandas as pd

def calculate_sentiment(df):
    """Calculate sentiment scores for headlines."""
    df['sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df