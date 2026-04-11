"""
Aspect-Based Sentiment Analysis (ABSA)
Assigns aspect tags to reviews using keyword matching, then uses the ONNX BERT
model to predict binary sentiment per review. Outputs per-aspect sentiment breakdown.
"""

import argparse
import os
import re

import pandas as pd
from onnx_inference import LABEL_MAP, minimal_clean, predict_sentiment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Aspect keyword dictionaries (from PRD F-04) ──────────────────────────────
ASPECT_KEYWORDS = {
    "UI/UX": [
        "interface",
        "design",
        "layout",
        "navigation",
        "simple",
        "confusing",
        "ui",
        "ux",
        "screen",
        "button",
        "menu",
        "theme",
        "dark mode",
        "look",
        "beautiful",
        "ugly",
        "clean",
        "cluttered",
        "intuitive",
        "user friendly",
    ],
    "Pricing": [
        "price",
        "subscription",
        "premium",
        "gold",
        "expensive",
        "free",
        "cost",
        "refund",
        "pay",
        "money",
        "charge",
        "billing",
        "plan",
        "worth",
        "cheap",
        "overpriced",
        "paid",
        "purchase",
        "tinder plus",
        "tinder gold",
        "platinum",
    ],
    "Matches": [
        "match",
        "like",
        "algorithm",
        "profile",
        "swipe",
        "boost",
        "super like",
        "connection",
        "date",
        "partner",
        "compatible",
        "suggestion",
        "recommend",
        "discover",
        "passport",
        "explore",
    ],
    "Bugs": [
        "crash",
        "bug",
        "glitch",
        "slow",
        "freeze",
        "load",
        "error",
        "fix",
        "update",
        "broken",
        "laggy",
        "stuck",
        "unresponsive",
        "not working",
        "force close",
        "hang",
        "lag",
    ],
    "Safety": [
        "fake",
        "bot",
        "scam",
        "privacy",
        "data",
        "report",
        "block",
        "verify",
        "catfish",
        "spam",
        "harassment",
        "safety",
        "secure",
        "identity",
        "verification",
        "real",
        "genuine",
        "suspicious",
    ],
}

# Pre-compile word-boundary regex per aspect for efficient matching
_ASPECT_PATTERNS = {
    aspect: re.compile(
        r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
        re.IGNORECASE,
    )
    for aspect, keywords in ASPECT_KEYWORDS.items()
}


def detect_aspects(text):
    """Return list of aspect names that match keywords in the text using word boundaries."""
    matched = []
    for aspect, pattern in _ASPECT_PATTERNS.items():
        if pattern.search(text):
            matched.append(aspect)
    return matched


def main():
    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
        return ivalue

    parser = argparse.ArgumentParser(description="Aspect-Based Sentiment Analysis")
    parser.add_argument(
        "--max-rows",
        type=positive_int,
        default=2000,
        help="Max reviews to process through ONNX inference (default: 2000)",
    )
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)

    # Load data
    print("Loading raw reviews...")
    df = pd.read_csv("data/raw_reviews.csv")
    df = df.dropna(subset=["review_text"])
    df["clean_text"] = df["review_text"].apply(minimal_clean)
    df = df[df["clean_text"].str.split().str.len() >= 4].reset_index(drop=True)

    total_available = len(df)
    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)
        print(
            f"Sampled {len(df)} / {total_available} reviews (--max-rows {args.max_rows})"
        )

    # Predict sentiment for sampled reviews
    print(f"Predicting sentiment for {len(df)} reviews (single-sample ONNX)...")
    texts = df["clean_text"].tolist()
    labels, confidences = predict_sentiment(texts)
    df["sentiment_label"] = [LABEL_MAP[l] for l in labels]
    df["sentiment_score"] = confidences

    # Detect aspects
    print("Detecting aspects...")
    df["aspects"] = df["clean_text"].apply(detect_aspects)

    # Explode into per-aspect rows
    rows = []
    for _, row in df.iterrows():
        aspects = row["aspects"]
        if not aspects:
            aspects = ["General"]
        for aspect in aspects:
            rows.append(
                {
                    "review_id": row.get("review_id", ""),
                    "app_name": row.get("app_name", ""),
                    "review_text": row["review_text"],
                    "clean_text": row["clean_text"],
                    "star_rating": row.get("star_rating", None),
                    "aspect": aspect,
                    "sentiment": row["sentiment_label"],
                    "confidence": row["sentiment_score"],
                }
            )

    result_df = pd.DataFrame(rows)
    out_path = "data/aspect_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Saved {len(result_df)} aspect-tagged rows to {out_path}")

    # Print summary
    if result_df.empty or not {"aspect", "sentiment"}.issubset(result_df.columns):
        print("\nNo aspect-tagged rows to summarize.")
    else:
        print("\n-- Aspect Sentiment Summary --")
        summary = (
            result_df.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
        )
        print(summary)


if __name__ == "__main__":
    main()
