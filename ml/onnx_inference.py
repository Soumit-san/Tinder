"""
Shared ONNX BERT inference utilities for Phase 5 analytics modules.
Handles single-sample inference since the exported ONNX model has fixed batch=1 shapes.
"""
import os
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_MAP = {0: 'Negative', 1: 'Positive'}

_session = None
_tokenizer = None


def minimal_clean(text):
    """Minimal cleaning matching the BERT training pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    try:
        import emoji
        text = emoji.demojize(text, delimiters=(" ", " "))
    except ImportError:
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^\w\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_model():
    """Lazily load the ONNX session and tokenizer (singleton)."""
    global _session, _tokenizer
    if _session is None:
        model_path = os.path.join(SCRIPT_DIR, 'models', 'model.onnx')
        _session = ort.InferenceSession(model_path)
        _tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return _session, _tokenizer


def predict_sentiment(texts):
    """Predict sentiment for a list of texts using the ONNX BERT model.
    Returns (labels, confidences) as parallel lists.
    Uses single-sample inference since the ONNX model has fixed batch=1 shape."""
    session, tokenizer = _get_model()
    input_names = [i.name for i in session.get_inputs()]

    all_labels = []
    all_confidences = []

    for text in texts:
        encoded = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=128, return_tensors="np"
        )
        feeds = {k: encoded[k].astype(np.int64) for k in encoded if k in input_names}
        logits = session.run(None, feeds)[0]
        logits_max = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - logits_max)
        probs = exp / exp.sum(axis=1, keepdims=True)
        label = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        all_labels.append(label)
        all_confidences.append(conf)

    return all_labels, all_confidences
