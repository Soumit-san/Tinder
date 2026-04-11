"""Tests for GET /api/reviews endpoint (pagination + filters)."""


class TestReviewsPagination:
    def test_default_page(self, client):
        resp = client.get("/api/reviews")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 1
        assert data["total"] == 10

    def test_page_size(self, client):
        resp = client.get("/api/reviews?page_size=3")
        data = resp.json()
        assert len(data["reviews"]) == 3
        assert data["page_size"] == 3

    def test_second_page(self, client):
        resp = client.get("/api/reviews?page=2&page_size=5")
        data = resp.json()
        assert data["page"] == 2
        assert len(data["reviews"]) == 5

    def test_review_item_structure(self, client):
        data = client.get("/api/reviews?page_size=1").json()
        review = data["reviews"][0]
        assert "review_id" in review
        assert "app_name" in review
        assert "review_text" in review
        assert "star_rating" in review


class TestReviewsFilters:
    def test_filter_positive_sentiment(self, client):
        resp = client.get("/api/reviews?sentiment=Positive")
        data = resp.json()
        for review in data["reviews"]:
            assert review["sentiment_label"] == "Positive"

    def test_filter_negative_sentiment(self, client):
        resp = client.get("/api/reviews?sentiment=Negative")
        data = resp.json()
        for review in data["reviews"]:
            assert review["sentiment_label"] == "Negative"

    def test_filter_min_stars(self, client):
        resp = client.get("/api/reviews?min_stars=4")
        data = resp.json()
        for review in data["reviews"]:
            assert review["star_rating"] >= 4

    def test_filter_max_stars(self, client):
        resp = client.get("/api/reviews?max_stars=2")
        data = resp.json()
        for review in data["reviews"]:
            assert review["star_rating"] <= 2

    def test_filter_mismatch(self, client):
        resp = client.get("/api/reviews?is_mismatch=true")
        data = resp.json()
        assert data["total"] >= 1
        for review in data["reviews"]:
            assert review["is_mismatch"] is True

    def test_filter_spam(self, client):
        resp = client.get("/api/reviews?is_spam=true")
        data = resp.json()
        assert data["total"] >= 1
        for review in data["reviews"]:
            assert review["is_spam"] is True
