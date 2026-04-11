import io
import os

import pandas as pd
from celery import Celery

from backend import inference

broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("worker", broker=broker_url, backend=broker_url)


@celery_app.task(bind=True, name="predict_batch_task")
def predict_batch_task(self, csv_content: str, text_col: str):
    if not inference.is_loaded():
        inference.load_model()

    # Read only first 2000 rows to avoid blowing up memory/GPU lag
    df = pd.read_csv(io.StringIO(csv_content), nrows=2000, low_memory=False)
    # Strip descriptive suffixes from columns
    df.columns = [str(c).split("[")[0].strip() for c in df.columns]

    # Infer other columns
    cols = [c.lower() for c in df.columns]
    rating_col = next(
        (c for c in df.columns if c.lower() in ["star_rating", "rating", "score"]), None
    )
    app_col = next(
        (c for c in df.columns if c.lower() in ["app_name", "app", "source"]), None
    )
    date_col = next(
        (
            c
            for c in df.columns
            if c.lower() in ["review_date", "date", "created_at", "at"]
        ),
        None,
    )

    df["review_text"] = df[text_col].astype(str).fillna("")
    df["review_id"] = [str(i) for i in range(len(df))]
    if rating_col:
        df["star_rating"] = pd.to_numeric(df[rating_col], errors="coerce").fillna(3)
    else:
        df["star_rating"] = 3

    if app_col:
        df["app_name"] = df[app_col].astype(str)
    else:
        df["app_name"] = "Custom Dataset"

    if date_col:
        df["review_date"] = pd.to_datetime(
            df[date_col], errors="coerce", dayfirst=True
        ).dt.strftime("%Y-%m-%d")
    else:
        df["review_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    def progress(current, total):
        self.update_state(state="PROGRESS", meta={"current": current, "total": total})

    texts = df["review_text"].tolist()
    predictions = inference.predict_batch(texts, progress_callback=progress)

    # Apply predictions and heuristics
    df["sentiment_label"] = [p[0] for p in predictions]
    df["sentiment_score"] = [p[1] for p in predictions]
    df["is_mismatch"] = (
        (df["star_rating"] >= 4) & (df["sentiment_label"] == "Negative")
    ) | ((df["star_rating"] <= 2) & (df["sentiment_label"] == "Positive"))

    # Basic Spam: repeated characters or too short but not a strong sentiment rating
    df["is_spam"] = (df["review_text"].str.len() < 4) & (df["star_rating"] == 3)

    return {"results": df.to_dict(orient="records")}
