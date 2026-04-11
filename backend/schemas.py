"""Pydantic v2 request/response models for all Sentix API endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool


# ── Dashboard Summary ────────────────────────────────────────────────────────
class SummaryResponse(BaseModel):
    total_reviews: int
    avg_rating: float
    positive_pct: float
    negative_pct: float
    spam_count: int
    mismatch_count: int


# ── Trends ────────────────────────────────────────────────────────────────────
class TrendPoint(BaseModel):
    date: str
    positive: int
    negative: int


class TrendsResponse(BaseModel):
    trends: list[TrendPoint]


# ── Mismatches ────────────────────────────────────────────────────────────────
class MismatchItem(BaseModel):
    review_id: str
    review_text: str
    star_rating: float
    sentiment_label: str
    sentiment_score: float


class MismatchesResponse(BaseModel):
    total: int
    mismatches: list[MismatchItem]


# ── Reviews (paginated) ──────────────────────────────────────────────────────
class ReviewItem(BaseModel):
    review_id: str
    app_name: str
    review_text: str
    star_rating: float
    review_date: Optional[str] = None
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    is_mismatch: Optional[bool] = None
    is_spam: Optional[bool] = None


class ReviewsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    reviews: list[ReviewItem]


# ── Aspects ───────────────────────────────────────────────────────────────────
class AspectStat(BaseModel):
    name: str
    positive: int
    negative: int
    total: int


class AspectsResponse(BaseModel):
    aspects: list[AspectStat]


# ── Keywords ──────────────────────────────────────────────────────────────────
class KeywordItem(BaseModel):
    word: str
    tfidf: float = 0.0
    count: int


class KeywordsResponse(BaseModel):
    positive: list[KeywordItem]
    negative: list[KeywordItem]
    top: list[KeywordItem]


# ── Compare ───────────────────────────────────────────────────────────────────
class AppCompare(BaseModel):
    app_name: str
    total_reviews: int
    avg_rating: float
    positive_pct: float
    negative_pct: float


class CompareResponse(BaseModel):
    apps: list[AppCompare]


# ── Predict ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to classify")


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    scores: dict[str, float]


class BatchPredictItem(BaseModel):
    text: str
    sentiment: str
    confidence: float


class BatchPredictResponse(BaseModel):
    results: list[BatchPredictItem]
