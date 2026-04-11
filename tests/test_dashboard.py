"""Tests for dashboard summary and trends endpoints."""


class TestDashboardSummary:
    def test_summary_status_code(self, client):
        resp = client.get("/api/dashboard/summary")
        assert resp.status_code == 200

    def test_summary_total_reviews(self, client):
        data = client.get("/api/dashboard/summary").json()
        assert data["total_reviews"] == 10

    def test_summary_avg_rating(self, client):
        data = client.get("/api/dashboard/summary").json()
        # ratings: [5,1,3,5,1,5,1,2,5,1] → mean = 2.9
        assert data["avg_rating"] == 2.9

    def test_summary_sentiment_percentages(self, client):
        data = client.get("/api/dashboard/summary").json()
        # 5 positive, 5 negative in aspect_results → 50/50
        assert data["positive_pct"] == 50.0
        assert data["negative_pct"] == 50.0

    def test_summary_spam_count(self, client):
        data = client.get("/api/dashboard/summary").json()
        assert data["spam_count"] == 1

    def test_summary_mismatch_count(self, client):
        data = client.get("/api/dashboard/summary").json()
        assert data["mismatch_count"] == 1


class TestDashboardTrends:
    def test_trends_status_code(self, client):
        resp = client.get("/api/dashboard/trends")
        assert resp.status_code == 200

    def test_trends_returns_list(self, client):
        data = client.get("/api/dashboard/trends").json()
        assert isinstance(data["trends"], list)
        assert len(data["trends"]) > 0

    def test_trends_point_structure(self, client):
        data = client.get("/api/dashboard/trends").json()
        point = data["trends"][0]
        assert "date" in point
        assert "positive" in point
        assert "negative" in point
