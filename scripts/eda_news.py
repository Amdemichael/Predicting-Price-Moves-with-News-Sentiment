import pandas as pd
import matplotlib.pyplot as plt

def load_news_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df['headline_length'] = df['headline'].apply(len)
    df['date_only'] = df['date'].dt.date
    return df

def describe_news(df):
    print("Headline Length Stats:")
    print(df['headline_length'].describe())

    print("\nTop Publishers:")
    print(df['publisher'].value_counts().head())

def plot_news_timeline(df):
    daily_counts = df.groupby('date_only').size()
    plt.figure(figsize=(12, 5))
    daily_counts.plot()
    plt.title('🕒 News Articles Over Time')
    plt.xlabel('Date')
    plt.ylabel('Articles')
    plt.show()