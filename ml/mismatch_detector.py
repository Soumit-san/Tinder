"""
Rating vs Sentiment Mismatch Detector
Flags reviews where star rating and BERT-predicted sentiment are inconsistent:
  - star_rating >= 4 AND sentiment == Negative
  - star_rating <= 2 AND sentiment == Positive
"""

import argparse
import os

import pandas as pd
from onnx_inference import LABEL_MAP, minimal_clean, predict_sentiment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
        return ivalue

    parser = argparse.ArgumentParser(
        description="Rating vs Sentiment Mismatch Detector"
    )
    parser.add_argument(
        "--max-rows",
        type=positive_int,
        default=2000,
        help="Max reviews to process through ONNX inference (default: 2000)",
    )
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)

    print("Loading raw reviews...")
    df = pd.read_csv("data/raw_reviews.csv")
    df = df.dropna(subset=["review_text", "star_rating"])
    df["clean_text"] = df["review_text"].apply(minimal_clean)
    df = df[df["clean_text"].str.split().str.len() >= 4].reset_index(drop=True)

    if len(df) == 0:
        print("No reviews remaining after filtering. Writing empty mismatches.csv.")
        cols = [
            "review_id",
            "app_name",
            "review_text",
            "star_rating",
            "sentiment_label",
            "sentiment_score",
            "is_mismatch",
        ]
        pd.DataFrame(columns=cols).to_csv("data/mismatches.csv", index=False)
        return

    total_available = len(df)
    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)
        print(
            f"Sampled {len(df)} / {total_available} reviews (--max-rows {args.max_rows})"
        )

    print(f"Predicting sentiment for {len(df)} reviews (single-sample ONNX)...")
    texts = df["clean_text"].tolist()
    labels, confidences = predict_sentiment(texts)
    df["sentiment_label"] = [LABEL_MAP[l] for l in labels]
    df["sentiment_score"] = confidences

    # Detect mismatches
    df["is_mismatch"] = False
    df.loc[
        (df["star_rating"] >= 4) & (df["sentiment_label"] == "Negative"), "is_mismatch"
    ] = True
    df.loc[
        (df["star_rating"] <= 2) & (df["sentiment_label"] == "Positive"), "is_mismatch"
    ] = True

    mismatch_count = df["is_mismatch"].sum()
    total = len(df)
    print(
        f"\nMismatches found: {mismatch_count} / {total} ({mismatch_count/total*100:.1f}%)"
    )

    # Save all reviews with mismatch flag
    cols = [
        "review_id",
        "app_name",
        "review_text",
        "star_rating",
        "sentiment_label",
        "sentiment_score",
        "is_mismatch",
    ]
    out_df = df[[c for c in cols if c in df.columns]]
    out_path = "data/mismatches.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} reviews (with mismatch flags) to {out_path}")

    # Print breakdown
    print("\n-- Mismatch Breakdown --")
    mismatched = df[df["is_mismatch"]]
    print(
        f"  High rating + Negative sentiment: {len(mismatched[(mismatched['star_rating'] >= 4) & (mismatched['sentiment_label'] == 'Negative')])}"
    )
    print(
        f"  Low rating + Positive sentiment:  {len(mismatched[(mismatched['star_rating'] <= 2) & (mismatched['sentiment_label'] == 'Positive')])}"
    )


if __name__ == "__main__":
    main()
