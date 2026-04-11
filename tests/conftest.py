"""
Shared pytest fixtures for Sentix Tinder test suite.
Provides a TestClient with synthetic in-memory data (no disk/model dependencies).
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Synthetic data builders ──────────────────────────────────────────────────


def _build_raw_reviews():
    """10 synthetic reviews across 2 apps."""
    return pd.DataFrame(
        {
            "review_id": [str(i) for i in range(10)],
            "app_name": ["Tinder"] * 5 + ["Bumble"] * 5,
            "review_text": [
                "Great app, love the matches!",
                "Terrible experience, full of bugs",
                "OK app, nothing special",
                "Amazing design, very intuitive UI",
                "Waste of money, too expensive",
                "Best dating app ever",
                "Crashes every time I open it",
                "Nice but the subscription is overpriced",
                "Love swiping, found my partner here",
                "Fake profiles everywhere, not safe",
            ],
            "star_rating": [5, 1, 3, 5, 1, 5, 1, 2, 5, 1],
            "review_date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
                "2025-01-07",
                "2025-01-08",
                "2025-01-09",
                "2025-01-10",
            ],
        }
    )


def _build_aspect_results():
    return pd.DataFrame(
        {
            "review_id": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "app_name": ["Tinder"] * 5 + ["Bumble"] * 5,
            "aspect": [
                "Matches",
                "Bugs",
                "General",
                "UI/UX",
                "Pricing",
                "General",
                "Bugs",
                "Pricing",
                "Matches",
                "Safety",
            ],
            "sentiment": [
                "Positive",
                "Negative",
                "Positive",
                "Positive",
                "Negative",
                "Positive",
                "Negative",
                "Negative",
                "Positive",
                "Negative",
            ],
        }
    )


def _build_mismatches():
    """Reviews 0 and 9 are mismatches (high rating + negative, or low rating + positive)."""
    return pd.DataFrame(
        {
            "review_id": [str(i) for i in range(10)],
            "review_text": [
                "Great app, love the matches!",
                "Terrible experience, full of bugs",
                "OK app, nothing special",
                "Amazing design, very intuitive UI",
                "Waste of money, too expensive",
                "Best dating app ever",
                "Crashes every time I open it",
                "Nice but the subscription is overpriced",
                "Love swiping, found my partner here",
                "Fake profiles everywhere, not safe",
            ],
            "star_rating": [5, 1, 3, 5, 1, 5, 1, 2, 5, 1],
            "sentiment_label": [
                "Positive",
                "Negative",
                "Positive",
                "Positive",
                "Negative",
                "Positive",
                "Negative",
                "Negative",
                "Positive",
                "Positive",
            ],
            "sentiment_score": [
                0.95,
                0.88,
                0.72,
                0.91,
                0.85,
                0.93,
                0.89,
                0.78,
                0.96,
                0.65,
            ],
            "is_mismatch": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )


def _build_flagged_reviews():
    return pd.DataFrame(
        {
            "review_id": [str(i) for i in range(10)],
            "is_spam": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )


def _build_keywords():
    return {
        "positive": [
            {"word": "love", "tfidf": 0.15, "count": 5},
            {"word": "great", "tfidf": 0.12, "count": 4},
        ],
        "negative": [
            {"word": "crash", "tfidf": 0.18, "count": 3},
            {"word": "expensive", "tfidf": 0.14, "count": 2},
        ],
        "top": [
            {"word": "app", "tfidf": 0.20, "count": 8},
            {"word": "tinder", "tfidf": 0.10, "count": 6},
        ],
    }


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def synthetic_data():
    """Returns all synthetic DataFrames and keyword dict."""
    return {
        "raw_reviews": _build_raw_reviews(),
        "aspect_results": _build_aspect_results(),
        "mismatches": _build_mismatches(),
        "flagged_reviews": _build_flagged_reviews(),
        "keywords": _build_keywords(),
    }


@pytest.fixture(scope="session")
def client(synthetic_data):
    """
    FastAPI TestClient with synthetic data injected into the store.
    Model is marked as loaded but inference calls must be individually mocked.
    """
    # Patch _load_data and inference.load_model before importing app
    with patch("backend.main._load_data") as mock_load, patch(
        "backend.inference.load_model", return_value=True
    ):

        from fastapi.testclient import TestClient

        from backend.main import app, store

        # Inject synthetic data into the global store
        store.raw_reviews = synthetic_data["raw_reviews"]
        store.aspect_results = synthetic_data["aspect_results"]
        store.mismatches = synthetic_data["mismatches"]
        store.flagged_reviews = synthetic_data["flagged_reviews"]
        store.keywords = synthetic_data["keywords"]

        # Build merged reviews (same logic as _load_data)
        merged = store.raw_reviews.copy()
        sent_df = store.mismatches[
            ["review_id", "sentiment_label", "sentiment_score", "is_mismatch"]
        ].drop_duplicates("review_id")
        merged = merged.merge(sent_df, on="review_id", how="left")
        spam_df = store.flagged_reviews[["review_id", "is_spam"]].drop_duplicates(
            "review_id"
        )
        merged = merged.merge(spam_df, on="review_id", how="left")
        merged["is_mismatch"] = merged["is_mismatch"].fillna(False)
        merged["is_spam"] = merged["is_spam"].fillna(False)
        store.reviews_merged = merged

        app.state.model_loaded = True

        with TestClient(app) as tc:
            yield tc
