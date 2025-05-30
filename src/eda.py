import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def descriptive_stats(df):
    """Compute basic statistics for headline lengths and article counts."""
    df['headline_length'] = df['headline'].apply(len)
    stats = {
        'headline_length_stats': df['headline_length'].describe(),
        'articles_per_publisher': df['publisher'].value_counts(),
        'articles_per_stock': df['stock'].value_counts()
    }
    return stats

def time_series_analysis(df, output_dir='plots'):
    """Analyze publication frequency over time."""
    df['date_only'] = df['date'].dt.date
    daily_counts = df.groupby('date_only').size()
    
    plt.figure(figsize=(10, 6))
    daily_counts.plot()
    plt.title('Article Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.savefig(f'{output_dir}/publication_frequency.png')
    plt.close()

def publisher_analysis(df):
    """Analyze publisher contribution."""
    publisher_counts = df['publisher'].value_counts()
    return publisher_counts.head(10)  # Top 10 publishers

def topic_modeling(df, num_keywords=10):
    """Extract common keywords from headlines."""
    headlines = df['headline'].str.lower()
    tokens = headlines.apply(word_tokenize)
    all_tokens = [token for sublist in tokens for token in sublist]
    freq = nltk.FreqDist(all_tokens)
    return freq.most_common(num_keywords)