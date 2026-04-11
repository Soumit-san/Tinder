"""Tests for GET /api/mismatches endpoint."""


def test_mismatches_status_code(client):
    resp = client.get("/api/mismatches")
    assert resp.status_code == 200


def test_mismatches_total(client):
    data = client.get("/api/mismatches").json()
    # Only review_id "9" has is_mismatch=True in synthetic data
    assert data["total"] == 1


def test_mismatches_item_structure(client):
    data = client.get("/api/mismatches").json()
    item = data["mismatches"][0]
    assert "review_id" in item
    assert "review_text" in item
    assert "star_rating" in item
    assert "sentiment_label" in item
    assert "sentiment_score" in item


def test_mismatches_contains_flagged_review(client):
    data = client.get("/api/mismatches").json()
    ids = [m["review_id"] for m in data["mismatches"]]
    assert "9" in ids
