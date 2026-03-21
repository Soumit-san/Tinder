"""
Slim ONNX BERT inference module for the FastAPI backend.
Loads the model once at startup and provides predict_one / predict_batch.
"""
import os
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

LABEL_MAP = {0: "Negative", 1: "Positive"}

_session = None
_tokenizer = None


def minimal_clean(text: str) -> str:
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


def load_model(model_dir: str | None = None) -> bool:
    """Load ONNX session + tokenizer. Returns True on success."""
    global _session, _tokenizer
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
    
    vocab_path = os.path.join(model_dir, "vocab.txt")
    tokenizer_json_path = os.path.join(model_dir, "tokenizer.json")
    if not (os.path.exists(vocab_path) or os.path.exists(tokenizer_json_path)):
        print(f"[inference] Tokenizer files not found in {model_dir}")
        return False

    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        print(f"[inference] ONNX model not found at {model_path}")
        return False
    try:
        _session = ort.InferenceSession(model_path)
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"[inference] Error loading model or tokenizer: {e}")
        _session = None
        _tokenizer = None
        return False

    print(f"[inference] ONNX model and tokenizer loaded successfully from {model_dir}")
    return True


def is_loaded() -> bool:
    return _session is not None and _tokenizer is not None


def predict_one(text: str) -> tuple[str, float, dict[str, float]]:
    """Predict sentiment for a single text.
    Returns (label, confidence, scores_dict)."""
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")
    clean = minimal_clean(text)
    input_names = [i.name for i in _session.get_inputs()]
    encoded = _tokenizer(
        clean, padding="max_length", truncation=True,
        max_length=128, return_tensors="np",
    )
    feeds = {k: encoded[k].astype(np.int64) for k in encoded if k in input_names}
    logits = _session.run(None, feeds)[0]
    # Numerically stable softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - logits_max)
    probs = exp / exp.sum(axis=1, keepdims=True)
    label_idx = int(np.argmax(probs, axis=1)[0])
    conf = float(np.max(probs, axis=1)[0])
    scores = {LABEL_MAP[i]: float(probs[0][i]) for i in range(probs.shape[1])}
    return LABEL_MAP[label_idx], conf, scores


def predict_batch(texts: list[str]) -> list[tuple[str, float]]:
    """Predict sentiment for a list of texts.
    Returns list of (label, confidence) tuples."""
    results = []
    for t in texts:
        label, conf, _ = predict_one(t)
        results.append((label, conf))
    return results
