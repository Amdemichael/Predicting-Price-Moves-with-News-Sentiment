from textblob import TextBlob
import pandas as pd

def compute_sentiment(df, text_column='headline'):
    df['sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

def get_daily_sentiment(df, date_column='date_only'):
    daily = df.groupby(date_column)['sentiment'].mean().reset_index()
    daily.columns = ['Date', 'Avg_Sentiment']
    return daily