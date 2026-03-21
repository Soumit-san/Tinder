"""
Fake / Spam Review Detection
Flags potentially fake or bot-generated reviews using:
  1. TF-IDF cosine similarity > 0.90 (duplicate detection)
  2. Length heuristic: reviews with < 3 words
  3. Repetition ratio: reviews where > 50% of tokens are repeated
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_short_review(text):
    """Flag reviews with fewer than 3 words."""
    if not isinstance(text, str):
        return True
    return len(text.split()) < 3


def check_repetition(text, threshold=0.5):
    """Flag reviews where > threshold fraction of tokens are repeated."""
    if not isinstance(text, str) or len(text.split()) == 0:
        return False
    tokens = text.lower().split()
    if len(tokens) <= 1:
        return False
    unique = set(tokens)
    repetition_ratio = 1 - (len(unique) / len(tokens))
    return repetition_ratio > threshold


def find_near_duplicates(texts, threshold=0.90, max_samples=5000):
    """Find pairs of reviews with cosine similarity > threshold.
    Uses TF-IDF vectorization. Raises ValueError if len(texts) exceeds
    the hard memory cap (max_samples); reduce --max-rows or increase
    max_samples to proceed."""
    if len(texts) > max_samples:
        raise ValueError(
            f"Input size ({len(texts)}) exceeds the hard memory cap "
            f"({max_samples}). Reduce --max-rows to <= {max_samples} "
            f"or increase max_samples in find_near_duplicates()."
        )
    sample_indices = np.arange(len(texts))
    sampled_texts = texts

    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(sampled_texts)

    # Compute similarity in chunks to avoid memory issues
    duplicate_indices = set()
    chunk_size = 500
    for i in range(0, tfidf_matrix.shape[0], chunk_size):
        chunk = tfidf_matrix[i:i + chunk_size]
        sim = cosine_similarity(chunk, tfidf_matrix)
        # Zero out self-similarity and lower triangle
        for row_idx in range(sim.shape[0]):
            global_row = i + row_idx
            sim[row_idx, :global_row + 1] = 0
        high_sim = np.argwhere(sim > threshold)
        for row_idx, col_idx in high_sim:
            duplicate_indices.add(sample_indices[i + row_idx])
            duplicate_indices.add(sample_indices[col_idx])

    return duplicate_indices


def main():
    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
        return ivalue

    parser = argparse.ArgumentParser(description="Fake / Spam Review Detection")
    parser.add_argument('--max-rows', type=positive_int, default=5000,
                        help="Max reviews to process (default: 5000)")
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)

    print("Loading raw reviews...")
    df = pd.read_csv('data/raw_reviews.csv')
    df = df.dropna(subset=['review_text'])

    total_available = len(df)
    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)
        print(f"Sampled {len(df)} / {total_available} reviews (--max-rows {args.max_rows})")

    texts = df['review_text'].tolist()

    # Flag reasons
    df['is_short'] = df['review_text'].apply(check_short_review)
    df['is_repetitive'] = df['review_text'].apply(check_repetition)

    print("Computing TF-IDF cosine similarity for duplicate detection...")
    # Only check non-short reviews for duplicates
    valid_texts = [t if isinstance(t, str) else "" for t in texts]
    dup_indices = find_near_duplicates(valid_texts)
    df['is_duplicate'] = False
    df.loc[df.index.isin(dup_indices), 'is_duplicate'] = True

    # Combine flags
    df['is_spam'] = df['is_short'] | df['is_repetitive'] | df['is_duplicate']
    df['flag_reason'] = ''
    reasons = []
    for _, row in df.iterrows():
        r = []
        if row['is_short']:
            r.append('too_short')
        if row['is_repetitive']:
            r.append('repetitive')
        if row['is_duplicate']:
            r.append('near_duplicate')
        reasons.append(', '.join(r) if r else '')
    df['flag_reason'] = reasons

    spam_count = df['is_spam'].sum()
    total = len(df)
    print(f"\nSpam/fake reviews flagged: {spam_count} / {total} ({spam_count/total*100:.1f}%)")
    print(f"  Too short (< 3 words): {df['is_short'].sum()}")
    print(f"  Repetitive (> 50% repeated tokens): {df['is_repetitive'].sum()}")
    print(f"  Near-duplicates (cosine > 0.90): {df['is_duplicate'].sum()}")

    # Save output
    cols = ['review_id', 'app_name', 'review_text', 'star_rating', 'flag_reason', 'is_spam']
    out_df = df[[c for c in cols if c in df.columns]]
    out_path = 'data/flagged_reviews.csv'
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} reviews (with spam flags) to {out_path}")


if __name__ == '__main__':
    main()
