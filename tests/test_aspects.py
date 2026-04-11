"""Tests for GET /api/aspects endpoint."""


def test_aspects_status_code(client):
    resp = client.get("/api/aspects")
    assert resp.status_code == 200


def test_aspects_returns_list(client):
    data = client.get("/api/aspects").json()
    assert isinstance(data["aspects"], list)
    assert len(data["aspects"]) > 0


def test_aspects_item_structure(client):
    data = client.get("/api/aspects").json()
    aspect = data["aspects"][0]
    assert "name" in aspect
    assert "positive" in aspect
    assert "negative" in aspect
    assert "total" in aspect


def test_aspects_total_equals_sum(client):
    """Each aspect's total should equal positive + negative."""
    data = client.get("/api/aspects").json()
    for aspect in data["aspects"]:
        assert aspect["total"] == aspect["positive"] + aspect["negative"]


def test_aspects_contains_expected_names(client):
    data = client.get("/api/aspects").json()
    names = {a["name"] for a in data["aspects"]}
    # Synthetic data has: Matches, Bugs, General, UI/UX, Pricing, Safety
    assert "Bugs" in names
    assert "Matches" in names
