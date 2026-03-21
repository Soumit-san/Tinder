"""
Keyword Generator
Generates keyword frequency data for the Word Cloud and Top Keywords dashboard screens.
Uses TF-IDF term scores and NLTK FreqDist, split by sentiment class.
"""
import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
stop_words = set(stopwords.words('english'))


def minimal_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_top_tfidf_words(texts, n=50):
    """Extract top-N words by mean TF-IDF score across the corpus."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[::-1][:n]
    return [
        {'word': feature_names[i], 'tfidf': round(float(mean_scores[i]), 6)}
        for i in top_indices
    ]


def get_freq_dist(texts, n=50):
    """Get top-N words by raw frequency, excluding stopwords."""
    all_tokens = []
    for text in texts:
        tokens = text.lower().split()
        filtered = [t for t in tokens if t not in stop_words and len(t) > 2]
        all_tokens.extend(filtered)
    fd = FreqDist(all_tokens)
    return [{'word': word, 'count': count} for word, count in fd.most_common(n)]


def main():
    os.chdir(SCRIPT_DIR)

    # Try to load aspect_results (which has sentiment labels) or fall back to mismatches
    if os.path.exists('data/aspect_results.csv'):
        print("Loading aspect_results.csv (has sentiment labels)...")
        df = pd.read_csv('data/aspect_results.csv')
        text_col = 'clean_text' if 'clean_text' in df.columns else 'review_text'
        sentiment_col = 'sentiment'
    elif os.path.exists('data/mismatches.csv'):
        print("Loading mismatches.csv (has sentiment labels)...")
        df = pd.read_csv('data/mismatches.csv')
        text_col = 'review_text'
        sentiment_col = 'sentiment_label'
        df['clean_text'] = df[text_col].apply(minimal_clean)
        text_col = 'clean_text'
    else:
        print("No labelled dataset found. Run aspect_sentiment.py or mismatch_detector.py first.")
        return

    df = df.dropna(subset=[text_col, sentiment_col])
    # Deduplicate by review text to avoid inflating word clouds
    df = df.drop_duplicates(subset=[text_col])

    pos_texts = df[df[sentiment_col] == 'Positive'][text_col].tolist()
    neg_texts = df[df[sentiment_col] == 'Negative'][text_col].tolist()
    all_texts = df[text_col].tolist()

    print(f"Positive reviews: {len(pos_texts)}, Negative reviews: {len(neg_texts)}")

    # TF-IDF keywords per sentiment
    print("Computing TF-IDF keywords...")
    pos_tfidf = get_top_tfidf_words(pos_texts) if pos_texts else []
    neg_tfidf = get_top_tfidf_words(neg_texts) if neg_texts else []

    # Frequency counts per sentiment
    print("Computing frequency distributions...")
    pos_freq = get_freq_dist(pos_texts)
    neg_freq = get_freq_dist(neg_texts)

    # Combined top keywords
    top_freq = get_freq_dist(all_texts)
    top_tfidf = get_top_tfidf_words(all_texts)

    # Merge tfidf + freq for each word
    def merge_keywords(tfidf_list, freq_list):
        freq_map = {item['word']: item['count'] for item in freq_list}
        merged = []
        for item in tfidf_list:
            word = item['word']
            merged.append({
                'word': word,
                'tfidf': item['tfidf'],
                'count': freq_map.get(word, 0),
            })
        return merged

    result = {
        'positive': merge_keywords(pos_tfidf, pos_freq),
        'negative': merge_keywords(neg_tfidf, neg_freq),
        'top': merge_keywords(top_tfidf, top_freq),
    }

    out_path = 'data/keywords.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved keyword data to {out_path}")

    # Print preview
    print("\n-- Top 10 Positive Keywords --")
    for kw in result['positive'][:10]:
        print(f"  {kw['word']:20s}  tfidf={kw['tfidf']:.5f}  count={kw['count']}")
    print("\n-- Top 10 Negative Keywords --")
    for kw in result['negative'][:10]:
        print(f"  {kw['word']:20s}  tfidf={kw['tfidf']:.5f}  count={kw['count']}")


if __name__ == '__main__':
    main()
