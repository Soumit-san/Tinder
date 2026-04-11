import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

model_dir = "ml/models"
session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"))
tokenizer = AutoTokenizer.from_pretrained(model_dir)

texts = ["This is great!"] * 100
encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="np")
input_names = [i.name for i in session.get_inputs()]
feeds = {k: encoded[k].astype(np.int64) for k in encoded if k in input_names}

t0 = time.time()
logits = session.run(None, feeds)[0]
t1 = time.time()

print(f"Batch of 100 took {t1-t0:.4f} seconds")
print(f"Logits shape: {logits.shape}")
