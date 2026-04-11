"""Tests for predict endpoints (single + batch)."""

import io
from unittest.mock import MagicMock, patch


class TestPredictSingle:
    def test_predict_success(self, client):
        with patch("backend.main.inference.is_loaded", return_value=True), patch(
            "backend.main.inference.predict_one",
            return_value=("Positive", 0.95, {"Positive": 0.95, "Negative": 0.05}),
        ):
            resp = client.post("/api/predict", json={"text": "I love this app"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["sentiment"] == "Positive"
            assert data["confidence"] == 0.95
            assert "Positive" in data["scores"]
            assert "Negative" in data["scores"]

    def test_predict_empty_text_422(self, client):
        resp = client.post("/api/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_missing_body_422(self, client):
        resp = client.post("/api/predict")
        assert resp.status_code == 422

    def test_predict_model_not_loaded_503(self, client):
        with patch("backend.main.inference.is_loaded", return_value=False):
            resp = client.post("/api/predict", json={"text": "test"})
            assert resp.status_code == 503


class TestPredictBatch:
    def _make_csv(self, rows):
        """Helper: builds a CSV string from list of dicts."""
        import pandas as pd

        df = pd.DataFrame(rows)
        return df.to_csv(index=False).encode("utf-8")

    def test_batch_upload_csv(self, client):
        csv = self._make_csv(
            [
                {"review_text": "Great app", "star_rating": 5},
                {"review_text": "Terrible", "star_rating": 1},
            ]
        )
        mock_task = MagicMock()
        mock_task.id = "test-job-123"
        with patch("backend.main.predict_batch_task.delay", return_value=mock_task):
            resp = client.post(
                "/api/predict/batch",
                files={"file": ("test.csv", io.BytesIO(csv), "text/csv")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["job_id"] == "test-job-123"
            assert data["status"] == "processing"

    def test_batch_upload_non_csv_400(self, client):
        resp = client.post(
            "/api/predict/batch",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_batch_upload_missing_text_col_400(self, client):
        csv = self._make_csv([{"name": "Alice", "rating": 5}])
        resp = client.post(
            "/api/predict/batch",
            files={"file": ("test.csv", io.BytesIO(csv), "text/csv")},
        )
        assert resp.status_code == 400


class TestBatchStatus:
    def test_status_completed(self, client):
        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.result = {"results": [{"text": "Great", "sentiment": "Positive"}]}

        with patch("backend.main.AsyncResult", return_value=mock_result):
            resp = client.get("/api/predict/batch/test-job-123")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "completed"
            assert "result" in data

    def test_status_processing(self, client):
        mock_result = MagicMock()
        mock_result.ready.return_value = False

        with patch("backend.main.AsyncResult", return_value=mock_result):
            resp = client.get("/api/predict/batch/test-job-123")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "processing"

    def test_status_failed(self, client):
        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_result.successful.return_value = False
        mock_result.result = Exception("Something went wrong")

        with patch("backend.main.AsyncResult", return_value=mock_result):
            resp = client.get("/api/predict/batch/test-job-123")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "failed"
