import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def clean_text(text):
    if text is None:
        return []
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [w for w in tokens if w not in stop_words and len(w) > 1]

def topic_modeling(df, num_keywords=10, num_bigrams=5):
    all_tokens = []
    for text in df['headline'].dropna():
        all_tokens.extend(clean_text(text))
    
    word_freq = Counter(all_tokens)
    bigrams = Counter(ngrams(all_tokens, 2))
    
    return {
        'top_keywords': word_freq.most_common(num_keywords),
        'top_bigrams': bigrams.most_common(num_bigrams)
    }
