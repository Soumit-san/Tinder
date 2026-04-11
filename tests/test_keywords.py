"""Tests for GET /api/keywords endpoint."""


def test_keywords_status_code(client):
    resp = client.get("/api/keywords")
    assert resp.status_code == 200


def test_keywords_has_all_categories(client):
    data = client.get("/api/keywords").json()
    assert "positive" in data
    assert "negative" in data
    assert "top" in data


def test_keywords_item_structure(client):
    data = client.get("/api/keywords").json()
    for category in ["positive", "negative", "top"]:
        assert len(data[category]) > 0
        item = data[category][0]
        assert "word" in item
        assert "tfidf" in item
        assert "count" in item


def test_keywords_positive_words(client):
    data = client.get("/api/keywords").json()
    words = [kw["word"] for kw in data["positive"]]
    assert "love" in words
    assert "great" in words
