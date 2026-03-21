import os
import io
import pandas as pd
from celery import Celery

from backend import inference

broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("worker", broker=broker_url, backend=broker_url)

@celery_app.task(name="predict_batch_task")
def predict_batch_task(csv_content: str, text_col: str):
    if not inference.is_loaded():
        inference.load_model()

    df = pd.read_csv(io.StringIO(csv_content))
    texts = df[text_col].dropna().astype(str).tolist()
    
    predictions = inference.predict_batch(texts)
    
    results = [
        {"text": str(t), "sentiment": str(label), "confidence": float(conf)}
        for t, (label, conf) in zip(texts, predictions)
    ]
    return {"results": results}
