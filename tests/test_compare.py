"""Tests for GET /api/compare endpoint."""


def test_compare_status_code(client):
    resp = client.get("/api/compare")
    assert resp.status_code == 200


def test_compare_returns_apps(client):
    data = client.get("/api/compare").json()
    assert isinstance(data["apps"], list)
    assert len(data["apps"]) == 2  # Tinder + Bumble in synthetic data


def test_compare_item_structure(client):
    data = client.get("/api/compare").json()
    app = data["apps"][0]
    assert "app_name" in app
    assert "total_reviews" in app
    assert "avg_rating" in app
    assert "positive_pct" in app
    assert "negative_pct" in app


def test_compare_app_names(client):
    data = client.get("/api/compare").json()
    names = {a["app_name"] for a in data["apps"]}
    assert "Tinder" in names
    assert "Bumble" in names
