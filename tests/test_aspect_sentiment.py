"""Tests for ml/aspect_sentiment.py — detect_aspects()."""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml")
)

from aspect_sentiment import detect_aspects


class TestDetectAspects:
    def test_bugs_aspect(self):
        aspects = detect_aspects("The app has a crash bug every time I open it")
        assert "Bugs" in aspects

    def test_pricing_aspect(self):
        aspects = detect_aspects(
            "The subscription is too expensive and not worth the price"
        )
        assert "Pricing" in aspects

    def test_matches_aspect(self):
        aspects = detect_aspects("I got great matches and found my date here")
        assert "Matches" in aspects

    def test_ui_ux_aspect(self):
        aspects = detect_aspects("The interface design is clean and intuitive")
        assert "UI/UX" in aspects

    def test_safety_aspect(self):
        aspects = detect_aspects("Too many fake profiles and scam accounts")
        assert "Safety" in aspects

    def test_multiple_aspects(self):
        aspects = detect_aspects(
            "The app has a crash bug and the subscription price is ridiculous"
        )
        assert "Bugs" in aspects
        assert "Pricing" in aspects

    def test_no_aspects(self):
        aspects = detect_aspects("Thanks for everything")
        assert len(aspects) == 0

    def test_case_insensitive(self):
        aspects = detect_aspects("CRASH BUG ERROR")
        assert "Bugs" in aspects
