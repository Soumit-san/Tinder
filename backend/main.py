"""
Sentix Tinder — FastAPI Backend
Serves Phase 5 analytics data and ONNX model predictions.
"""

import io
import json
import os
import re
from collections import Counter
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from celery.result import AsyncResult
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from kombu.exceptions import OperationalError

from backend import inference
from backend.schemas import (
    AppCompare,
    AspectsResponse,
    AspectStat,
    BatchPredictItem,
    BatchPredictResponse,
    CompareResponse,
    HealthResponse,
    KeywordItem,
    KeywordsResponse,
    MismatchesResponse,
    MismatchItem,
    PredictRequest,
    PredictResponse,
    ReviewItem,
    ReviewsResponse,
    SummaryResponse,
    TrendPoint,
    TrendsResponse,
)
from backend.worker import celery_app, predict_batch_task

# ── Resolve data paths relative to project root ──────────────────────────────
# API reads from the same directory producers write to
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "ml", "data")


# ── In-memory data store ─────────────────────────────────────────────────────
class DataStore:
    """Holds all pre-loaded DataFrames and keyword data in memory."""

    raw_reviews: pd.DataFrame = None
    aspect_results: pd.DataFrame = None
    mismatches: pd.DataFrame = None
    flagged_reviews: pd.DataFrame = None
    keywords: dict = None
    # Merged reviews with sentiment + mismatch + spam flags
    reviews_merged: pd.DataFrame = None


store = DataStore()


def _load_data():
    """Load all Phase 5 CSV/JSON outputs into memory."""
    print("[startup] Loading Phase 5 data files...")

    store.raw_reviews = pd.read_csv(os.path.join(DATA_DIR, "raw_reviews.csv"))
    store.raw_reviews["review_id"] = store.raw_reviews["review_id"].astype(str)

    store.aspect_results = pd.read_csv(os.path.join(DATA_DIR, "aspect_results.csv"))
    store.aspect_results["review_id"] = store.aspect_results["review_id"].astype(str)

    store.mismatches = pd.read_csv(os.path.join(DATA_DIR, "mismatches.csv"))
    store.mismatches["review_id"] = store.mismatches["review_id"].astype(str)

    store.flagged_reviews = pd.read_csv(os.path.join(DATA_DIR, "flagged_reviews.csv"))
    store.flagged_reviews["review_id"] = store.flagged_reviews["review_id"].astype(str)

    with open(os.path.join(DATA_DIR, "keywords.json"), encoding="utf-8") as f:
        store.keywords = json.load(f)

    # Build a merged reviews view for the /api/reviews endpoint
    merged = store.raw_reviews.copy()
    # Join sentiment from mismatches
    if "sentiment_label" in store.mismatches.columns:
        sent_df = store.mismatches[
            ["review_id", "sentiment_label", "sentiment_score", "is_mismatch"]
        ].drop_duplicates("review_id")
        merged = merged.merge(sent_df, on="review_id", how="left")
    # Join spam flags
    if "is_spam" in store.flagged_reviews.columns:
        spam_df = store.flagged_reviews[["review_id", "is_spam"]].drop_duplicates(
            "review_id"
        )
        merged = merged.merge(spam_df, on="review_id", how="left")

    merged["is_mismatch"] = merged.get(
        "is_mismatch", pd.Series(False, index=merged.index)
    ).fillna(False)
    merged["is_spam"] = merged.get(
        "is_spam", pd.Series(False, index=merged.index)
    ).fillna(False)

    # Fix false-positive spam for short positive/negative reviews (e.g. "Good", "worst")
    if "star_rating" in merged.columns and "review_text" in merged.columns:
        short_mask = (merged["review_text"].astype(str).str.len() < 30) & (
            (merged["star_rating"] >= 4) | (merged["star_rating"] <= 2)
        )
        merged.loc[short_mask, "is_spam"] = False

    store.reviews_merged = merged

    print(
        f"[startup] Loaded {len(store.raw_reviews)} raw reviews, "
        f"{len(store.aspect_results)} aspect rows, "
        f"{len(store.mismatches)} mismatch rows, "
        f"{len(store.flagged_reviews)} flagged rows"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load data + model. Shutdown: cleanup."""
    _load_data()
    model_loaded = inference.load_model()
    app.state.model_loaded = model_loaded
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentix Tinder API",
    description="Sentiment analysis API for Tinder app store reviews",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(model_loaded=getattr(app.state, "model_loaded", False))


# ── Dashboard Summary ────────────────────────────────────────────────────────
@app.get("/api/dashboard/summary", response_model=SummaryResponse)
def dashboard_summary():
    raw = store.raw_reviews
    total = len(raw)
    avg_rating = round(float(raw["star_rating"].mean()), 2)

    # Positive/Negative percentages from aspect_results (deduplicated by review_id)
    aspect_dedup = store.aspect_results.drop_duplicates("review_id")
    pos_count = (aspect_dedup["sentiment"] == "Positive").sum()
    neg_count = (aspect_dedup["sentiment"] == "Negative").sum()
    sent_total = pos_count + neg_count
    positive_pct = round(pos_count / sent_total * 100, 1) if sent_total else 0
    negative_pct = round(neg_count / sent_total * 100, 1) if sent_total else 0

    spam_count = int(store.flagged_reviews["is_spam"].sum())
    mismatch_count = int(
        store.mismatches.get("is_mismatch", pd.Series(dtype=bool)).sum()
    )

    return SummaryResponse(
        total_reviews=total,
        avg_rating=avg_rating,
        positive_pct=positive_pct,
        negative_pct=negative_pct,
        spam_count=spam_count,
        mismatch_count=mismatch_count,
    )


# ── Trends ────────────────────────────────────────────────────────────────────
@app.get("/api/dashboard/trends", response_model=TrendsResponse)
def dashboard_trends():
    # Build trends from raw_reviews + sentiment labels from mismatches
    merged = store.reviews_merged.copy()
    if "review_date" not in merged.columns or "sentiment_label" not in merged.columns:
        return TrendsResponse(trends=[])

    merged = merged.dropna(subset=["review_date", "sentiment_label"])
    merged["date"] = pd.to_datetime(merged["review_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    merged = merged.dropna(subset=["date"])

    grouped = (
        merged.groupby("date")["sentiment_label"].value_counts().unstack(fill_value=0)
    )
    trends = []
    for date in sorted(grouped.index):
        trends.append(
            TrendPoint(
                date=date,
                positive=int(grouped.loc[date].get("Positive", 0)),
                negative=int(grouped.loc[date].get("Negative", 0)),
            )
        )
    return TrendsResponse(trends=trends)


# ── Mismatches ────────────────────────────────────────────────────────────────
@app.get("/api/mismatches", response_model=MismatchesResponse)
def mismatches():
    df = store.mismatches
    flagged = df[df["is_mismatch"] == True].copy()
    items = []
    for _, row in flagged.iterrows():
        items.append(
            MismatchItem(
                review_id=str(row["review_id"]),
                review_text=str(row.get("review_text", "")),
                star_rating=float(row.get("star_rating", 0)),
                sentiment_label=str(row.get("sentiment_label", "")),
                sentiment_score=float(row.get("sentiment_score", 0)),
            )
        )
    return MismatchesResponse(total=len(items), mismatches=items)


# ── Reviews (paginated + filterable) ──────────────────────────────────────────
@app.get("/api/reviews", response_model=ReviewsResponse)
def reviews(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sentiment: Optional[str] = Query(None, description="Filter: Positive or Negative"),
    min_stars: Optional[int] = Query(None, ge=1, le=5),
    max_stars: Optional[int] = Query(None, ge=1, le=5),
    is_mismatch: Optional[bool] = Query(None),
    is_spam: Optional[bool] = Query(None),
):
    df = store.reviews_merged.copy()

    if sentiment:
        df = df[df["sentiment_label"] == sentiment]
    if min_stars is not None:
        df = df[df["star_rating"] >= min_stars]
    if max_stars is not None:
        df = df[df["star_rating"] <= max_stars]
    if is_mismatch is not None:
        df = df[df["is_mismatch"] == is_mismatch]
    if is_spam is not None:
        df = df[df["is_spam"] == is_spam]

    total = len(df)
    start = (page - 1) * page_size
    page_df = df.iloc[start : start + page_size]

    items = []
    for _, row in page_df.iterrows():
        items.append(
            ReviewItem(
                review_id=str(row["review_id"]),
                app_name=str(row.get("app_name", "")),
                review_text=str(row.get("review_text", "")),
                star_rating=float(row.get("star_rating", 0)),
                review_date=(
                    str(row["review_date"])
                    if pd.notna(row.get("review_date"))
                    else None
                ),
                sentiment_label=(
                    str(row["sentiment_label"])
                    if pd.notna(row.get("sentiment_label"))
                    else None
                ),
                sentiment_score=(
                    float(row["sentiment_score"])
                    if pd.notna(row.get("sentiment_score"))
                    else None
                ),
                is_mismatch=bool(row.get("is_mismatch", False)),
                is_spam=bool(row.get("is_spam", False)),
            )
        )

    return ReviewsResponse(total=total, page=page, page_size=page_size, reviews=items)


# ── Aspects ───────────────────────────────────────────────────────────────────
@app.get("/api/aspects", response_model=AspectsResponse)
def aspects():
    df = store.aspect_results
    grouped = df.groupby("aspect")["sentiment"].value_counts().unstack(fill_value=0)
    items = []
    for aspect in sorted(grouped.index):
        pos = int(grouped.loc[aspect].get("Positive", 0))
        neg = int(grouped.loc[aspect].get("Negative", 0))
        items.append(
            AspectStat(name=aspect, positive=pos, negative=neg, total=pos + neg)
        )
    return AspectsResponse(aspects=items)


# ── Keywords ──────────────────────────────────────────────────────────────────
@app.get("/api/keywords", response_model=KeywordsResponse)
def keywords():
    data = store.keywords
    return KeywordsResponse(
        positive=[KeywordItem(**kw) for kw in data.get("positive", [])],
        negative=[KeywordItem(**kw) for kw in data.get("negative", [])],
        top=[KeywordItem(**kw) for kw in data.get("top", [])],
    )


# ── Compare (cross-app) ──────────────────────────────────────────────────────
@app.get("/api/compare", response_model=CompareResponse)
def compare():
    # Aggregate by app_name from aspect_results (has sentiment per review)
    df = store.aspect_results.drop_duplicates("review_id")
    raw = store.raw_reviews

    apps = []
    for app_name in sorted(raw["app_name"].dropna().unique()):
        app_raw = raw[raw["app_name"] == app_name]
        app_sent = df[df["app_name"] == app_name]
        total = len(app_raw)
        avg_rating = round(float(app_raw["star_rating"].mean()), 2) if total else 0
        pos = (app_sent["sentiment"] == "Positive").sum()
        neg = (app_sent["sentiment"] == "Negative").sum()
        st = pos + neg
        apps.append(
            AppCompare(
                app_name=app_name,
                total_reviews=total,
                avg_rating=avg_rating,
                positive_pct=round(pos / st * 100, 1) if st else 0,
                negative_pct=round(neg / st * 100, 1) if st else 0,
            )
        )
    return CompareResponse(apps=apps)


# ── Predict (single) ─────────────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not inference.is_loaded():
        raise HTTPException(503, "Model not loaded")
    label, conf, scores = inference.predict_one(req.text)
    return PredictResponse(sentiment=label, confidence=round(conf, 4), scores=scores)


@app.post("/api/predict/batch", response_model=dict)
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    content = await file.read()
    try:
        content_str = content.decode("utf-8")
        df = pd.read_csv(io.StringIO(content_str))
        # Strip descriptive suffixes from columns, e.g. "content [Description...]" -> "content"
        df.columns = [str(c).split("[")[0].strip() for c in df.columns]
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        Exception,
    ) as e:
        raise HTTPException(status_code=400, detail=f"Malformed CSV upload: {str(e)}")

    cols = [c.lower() for c in df.columns]

    # Fuzzy column matching
    text_col = next(
        (
            c
            for c in df.columns
            if c.lower() in ["review_text", "text", "content", "review"]
        ),
        None,
    )
    if not text_col:
        raise HTTPException(
            400, "CSV must have a 'review_text', 'text', 'content', or 'review' column"
        )

    # Dispatch to Celery worker, pass the entire CSV so it can process ratings/dates too
    try:
        job = predict_batch_task.delay(content_str, text_col)
        return {"job_id": job.id, "status": "processing"}
    except OperationalError as e:
        print(f"[predict_batch] Celery broker OperationalError: {e}")
        raise HTTPException(503, "Batch processing service is currently unavailable.")


@app.get("/api/predict/batch/{job_id}", response_model=dict)
def get_batch_status(job_id: str):
    try:
        res = AsyncResult(job_id, app=celery_app)
        if res.ready():
            if res.successful():
                return {"job_id": job_id, "status": "completed", "result": res.result}
            else:
                return {"job_id": job_id, "status": "failed", "error": str(res.result)}

        # Check for numeric progress in meta
        meta = res.info if isinstance(res.info, dict) else {}
        return {
            "job_id": job_id,
            "status": "processing",
            "progress": meta.get("current", 0),
            "total": meta.get("total", 0),
        }
    except OperationalError as e:
        print(
            f"[get_batch_status] Celery broker OperationalError for job {job_id}: {e}"
        )
        raise HTTPException(503, "Batch processing service is currently unavailable.")


@app.post("/api/dataset/apply/{job_id}", response_model=dict)
def apply_uploaded_dataset(job_id: str):
    try:
        res = AsyncResult(job_id, app=celery_app)
        if not res.ready() or not res.successful():
            raise HTTPException(400, "Job is not ready or failed.")

        data = res.result.get("results", [])
        if not data:
            raise HTTPException(400, "No results found in job.")

        new_df = pd.DataFrame(data)

        # Override the global store
        store.raw_reviews = new_df
        store.mismatches = new_df[new_df["is_mismatch"] == True].copy()
        if "is_spam" in new_df.columns:
            store.flagged_reviews = new_df[new_df["is_spam"] == True].copy()

        # Aspects (fake for custom datasets to preserve UI)
        aspect_records = []
        for i, row in new_df.iterrows():
            if pd.notna(row.get("sentiment_label")):
                aspect_records.append(
                    {
                        "review_id": row["review_id"],
                        "app_name": row.get("app_name", "Custom"),
                        "aspect": "Overall",
                        "sentiment": row["sentiment_label"],
                    }
                )
        store.aspect_results = (
            pd.DataFrame(aspect_records)
            if aspect_records
            else pd.DataFrame(columns=["review_id", "app_name", "aspect", "sentiment"])
        )

        # Override merged view
        store.reviews_merged = new_df

        # Generate new keywords
        store.keywords = _generate_keywords_for_df(new_df)

        return {"status": "applied", "message": "Dataset swapped successfully."}
    except Exception as e:
        raise HTTPException(500, f"Failed to apply dataset: {str(e)}")

def _generate_keywords_for_df(df: pd.DataFrame) -> dict:
    stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    }
    
    def get_freq(texts):
        words = []
        for text in texts:
            if not isinstance(text, str):
                continue
            text = text.lower()
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            tokens = text.split()
            words.extend([w for w in tokens if w not in stop_words and len(w) > 2])
        c = Counter(words)
        return [{"word": w, "count": cnt} for w, cnt in c.most_common(50)]

    pos_texts = df[df["sentiment_label"] == "Positive"]["review_text"].tolist() if "sentiment_label" in df.columns and "review_text" in df.columns else []
    neg_texts = df[df["sentiment_label"] == "Negative"]["review_text"].tolist() if "sentiment_label" in df.columns and "review_text" in df.columns else []
    all_texts = df["review_text"].tolist() if "review_text" in df.columns else []

    return {
        "positive": get_freq(pos_texts),
        "negative": get_freq(neg_texts),
        "top": get_freq(all_texts)
    }

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
