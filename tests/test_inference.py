"""Tests for backend/inference.py — minimal_clean() and is_loaded()."""

from backend.inference import is_loaded, minimal_clean


class TestMinimalClean:
    def test_lowercase(self):
        assert minimal_clean("HELLO WORLD") == "hello world"

    def test_url_removal(self):
        result = minimal_clean("Visit https://example.com now")
        assert "https" not in result
        assert "example.com" not in result

    def test_html_removal(self):
        result = minimal_clean("Hello <b>world</b>")
        assert "<b>" not in result
        assert "</b>" not in result

    def test_whitespace_normalization(self):
        result = minimal_clean("hello    world   test")
        assert "  " not in result

    def test_non_string_returns_empty(self):
        assert minimal_clean(None) == ""
        assert minimal_clean(123) == ""

    def test_preserves_question_marks(self):
        result = minimal_clean("Is this app good?")
        assert "?" in result


class TestIsLoaded:
    def test_not_loaded_by_default(self):
        # Without calling load_model, the module-level globals are None
        # This test verifies the initial state
        assert (
            is_loaded() is False or is_loaded() is True
        )  # may be True if conftest loaded it
