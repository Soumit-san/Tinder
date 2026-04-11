"""Tests for ml/spam_detector.py — check_short_review, check_repetition, find_near_duplicates."""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml")
)

from spam_detector import check_repetition, check_short_review, find_near_duplicates


class TestCheckShortReview:
    def test_short_review_flagged(self):
        assert check_short_review("good app") is True  # 2 words
        assert check_short_review("ok") is True  # 1 word

    def test_adequate_review_not_flagged(self):
        assert check_short_review("This app is great") is False  # 4 words

    def test_exactly_3_words_not_flagged(self):
        assert check_short_review("app is good") is False

    def test_non_string_returns_true(self):
        assert check_short_review(None) is True
        assert check_short_review(42) is True


class TestCheckRepetition:
    def test_highly_repetitive(self):
        # "good good good good good" → 1 unique / 5 total = 0.8 repetition
        assert check_repetition("good good good good good") is True

    def test_normal_text_not_repetitive(self):
        assert check_repetition("this app has a great design") is False

    def test_empty_string(self):
        assert check_repetition("") is False

    def test_single_word(self):
        assert check_repetition("hello") is False

    def test_non_string(self):
        assert check_repetition(None) is False


class TestFindNearDuplicates:
    def test_identical_texts_detected(self):
        texts = [
            "This app is absolutely amazing and I love it",
            "This app is absolutely amazing and I love it",
            "Terrible experience with bugs and crashes",
        ]
        dups = find_near_duplicates(texts, threshold=0.90, max_samples=100)
        # Indices 0 and 1 should be flagged
        assert 0 in dups
        assert 1 in dups

    def test_unique_texts_not_flagged(self):
        texts = [
            "This app is absolutely amazing and wonderful",
            "Terrible experience with bugs and crashes everywhere",
            "The pricing model needs significant improvements",
        ]
        dups = find_near_duplicates(texts, threshold=0.90, max_samples=100)
        assert len(dups) == 0

    def test_exceeds_max_samples_raises(self):
        import pytest

        texts = ["text"] * 20
        with pytest.raises(ValueError, match="exceeds the hard memory cap"):
            find_near_duplicates(texts, max_samples=10)
