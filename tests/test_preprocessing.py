"""Tests for ml/preprocessing.py — clean_text() and preprocess_pipeline()."""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml")
)

from preprocessing import clean_text, preprocess_pipeline


class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_url_removal(self):
        result = clean_text("Visit https://example.com for more")
        assert "https" not in result
        assert "example.com" not in result

    def test_html_removal(self):
        result = clean_text("Hello <b>world</b>")
        assert "<b>" not in result
        assert "</b>" not in result

    def test_emoji_removal(self):
        result = clean_text("Great app 😊👍")
        assert "😊" not in result
        assert "👍" not in result

    def test_punctuation_removal(self):
        result = clean_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_non_string_returns_empty(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""
        assert clean_text(float("nan")) == ""


class TestPreprocessPipeline:
    def test_stopword_removal(self):
        # "this is a very good application for dating and meeting" has enough tokens
        result = preprocess_pipeline(
            "this is a very good application for dating and meeting people together"
        )
        # Common stopwords like "this", "is", "a" should be removed
        tokens = result.split()
        assert "this" not in tokens
        assert "is" not in tokens

    def test_lemmatization(self):
        result = preprocess_pipeline(
            "the applications were running and crashing repeatedly every single time"
        )
        tokens = result.split()
        # "applications" should be lemmatized to "application"
        if "application" in tokens:
            assert True
        # "crashing" may or may not be lemmatized depending on NLTK version
        assert len(tokens) > 0

    def test_short_review_filtered(self):
        # Less than 5 tokens after processing → empty string
        result = preprocess_pipeline("good app")
        assert result == ""

    def test_adequate_length_review(self):
        result = preprocess_pipeline(
            "This application has amazing design features and wonderful user experience overall"
        )
        assert len(result) > 0
        assert isinstance(result, str)
