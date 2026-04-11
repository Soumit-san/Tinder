# Sentix Tinder — App Store Review Analytics

Sentix Tinder is an advanced sentiment analysis and analytics dashboard for dating app reviews (Tinder, Bumble, Hinge). It leverages a fine-tuned BERT model (exported to ONNX) to classify review sentiment, perform Aspect-Based Sentiment Analysis (ABSA), detect rating mismatches, and flag fake/spam reviews.

![Sentix Tinder](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?logo=tailwindcss)

## 🌟 Key Features

- **Fine-Tuned BERT Sentiment:** High-performance ONNX inference for binary sentiment classification (Positive/Negative).
- **Aspect-Based Sentiment Analysis (ABSA):** Breakdowns by UI/UX, Pricing, Matches, Bugs, and Safety.
- **Mismatch Detection:** Automatically flags reviews where the star rating contradicts the textual sentiment (e.g., 5-star rating with a highly negative review).
- **Spam & Fake Review Detection:** Uses TF-IDF cosine similarity (>0.90) and repetition heuristics to flag bot-generated or low-effort reviews.
- **Interactive Dashboard:** Beautiful glassmorphism UI offering top keywords, trend charts, and aspect breakdowns.
- **Async Batch Upload:** Upload a CSV of your own reviews and process them asynchronously via Celery and Redis.

## 🏗 System Architecture

The project is structured into three main directories:
- **`ml/`**: Machine learning pipelines. Scripts for scraping Google Play store reviews, preprocessing, training baseline models, fine-tuning BERT, and exporting to ONNX. Includes standalone analytic scripts for spam and mismatch detection.
- **`backend/`**: A centralized FastAPI server powering the web dashboard. It utilizes `onnxruntime` for fast CPU-bound model inference and integrates with Celery/Redis for handling large CSV uploads via the `/api/predict/batch` endpoint.
- **`frontend/`**: A React 19 + Vite application stylized with Tailwind CSS. Follows a mobile-first dark-mode aesthetic with interactive charts (`recharts` & `chart.js`) and word clouds.

## 🚀 Getting Started

### Prerequisites

You will need Python 3.10+, Node.js 18+, and Redis running locally (for Celery background tasks).
Alternatively, you can run the backend via Docker Compose.

### 1. Setup Backend & ML 

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or `.\.venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```
2. The pre-trained ONNX model should be placed in `ml/models/model.onnx`.
3. Start the FastAPI server:
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
4. Start Redis and the Celery worker (in a separate terminal) for CSV uploads:
   ```bash
   cd backend
   # Windows:
   celery -A worker.celery_app worker -l info -P threads
   # Unix:
   celery -A worker.celery_app worker -l info
   ```

### 2. Setup Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:5173` to view the dashboard in your browser.

## 🧪 Running Tests

The backend application contains a comprehensive 84-test Pytest suite heavily utilizing mocked synthetic data, meaning it can run instantly without requiring the ONNX model, Redis, or Celery.

```bash
# Run tests from the project root
python -m pytest tests/ -v --tb=short
```

## 🐳 Docker Deployment

To spin up the Backend API, Redis, and Celery Worker simultaneously, use:

```bash
docker-compose up --build
```
This will mount the local `ml/` data and start the FastAPI service on port 8000.

---

*Phase 8 - Final Polish & Testing Completed. System holds a perfect 100% Lighthouse Accessibility score.*