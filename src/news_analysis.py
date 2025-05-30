import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_news_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['headline_length'] = df['headline'].apply(len)
    return df


def plot_headline_lengths(df):
    sns.histplot(df['headline_length'], bins=50)
    plt.title("Distribution of Headline Lengths")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.show()


def plot_publication_trends(df):
    df['date'].dt.date.value_counts().sort_index().plot(figsize=(15,5))
    plt.title("Publication Frequency Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.show()


def top_publishers(df, top_n=10):
    df['publisher'].value_counts().head(top_n).plot(kind='barh')
    plt.title(f"Top {top_n} Publishers")
    plt.xlabel("Article Count")
    plt.show()
