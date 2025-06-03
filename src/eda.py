import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
import os # Import the os module for path manipulation

# --- NLTK Downloads (Ensure these are run once before using NLTK functions) ---
def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    print("Checking NLTK data...")
    # List of NLTK packages needed
    nltk_packages = ['punkt', 'stopwords', 'punkt_tab'] # Explicitly include punkt_tab

    for package in nltk_packages:
        try:
            # Try to find the resource, if not found, download it
            nltk.data.find(f'tokenizers/{package}')
            print(f"'{package}' NLTK data already present.")
        except LookupError:
            print(f"Downloading '{package}' NLTK data...")
            nltk.download(package)
            print(f"Downloaded '{package}' NLTK data.")

    # --- Optional: Verify NLTK Data Path ---
    # Sometimes the issue is where NLTK looks for the data.
    # You can add your virtual environment's nltk_data path if it's not picked up automatically.
    # This might require checking your 'taenv' directory structure.
    # For example, if your 'nltk_data' is directly under 'taenv', you could add:
    # current_env_nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'taenv', 'nltk_data')
    # if current_env_nltk_data_path not in nltk.data.path:
    #     nltk.data.path.append(current_env_nltk_data_path)
    #     print(f"Added '{current_env_nltk_data_path}' to NLTK data paths.")

# Call the download function at the start of your script or notebook
download_nltk_data()

# --- EDA Functions (rest of your code, unchanged from previous good version) ---

def descriptive_stats(df):
    """
    Compute basic statistics for headline lengths and article counts.
    """
    df['headline_length'] = df['headline'].astype(str).apply(len)
    stats = {
        'headline_length_stats': df['headline_length'].describe(),
        'articles_per_publisher': df['publisher'].value_counts(),
        'articles_per_stock': df['stock'].value_counts()
    }
    return stats

def time_series_analysis(df):
    """
    Analyze publication frequency over time and generate a plot.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df.dropna(subset=['date'], inplace=True)

    df['date_only'] = df['date'].dt.date
    daily_counts = df.groupby('date_only').size()

    plt.figure(figsize=(12, 7))
    daily_counts.plot(kind='line')
    plt.title('Article Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def publisher_analysis(df):
    """
    Analyze publisher contribution and visualize the top 10 publishers.
    """
    publisher_counts = df['publisher'].value_counts()
    top_10_publishers = publisher_counts.head(10)

    plt.figure(figsize=(12, 7))
    sns.barplot(x=top_10_publishers.index, y=top_10_publishers.values, palette='viridis')
    plt.title('Top 10 Publishers by Article Count')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return top_10_publishers

def clean_text(text):
    """
    Performs basic text cleaning: lowercase, remove non-alphabetic, tokenize, remove stopwords.
    """
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return filtered_tokens

def topic_modeling(df, num_keywords=10, num_bigrams=5):
    """
    Extract common keywords and bigrams from headlines after cleaning.
    """
    all_cleaned_tokens = []
    for headline in df['headline'].dropna():
        all_cleaned_tokens.extend(clean_text(headline))

    word_freq = Counter(all_cleaned_tokens)
    top_single_keywords = word_freq.most_common(num_keywords)

    from nltk.util import ngrams
    bigrams = Counter(list(ngrams(all_cleaned_tokens, 2)))
    top_two_word_phrases = bigrams.most_common(num_bigrams)

    return {
        'top_keywords': top_single_keywords,
        'top_bigrams': top_two_word_phrases
    }