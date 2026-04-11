import time
import numpy as np
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Add backend to path
sys.path.append(os.getcwd())
from backend import inference

def bench_threaded(num_threads=4):
    print(f"Loading model (Threads={num_threads})...")
    inference.load_model()
    
    texts = ["I love this app! it is great but sometimes it crashes."] * 100
    
    # We need to hack inference.py logic or use ThreadPoolExecutor here
    input_names = [i.name for i in inference._session.get_inputs()]
    
    def predict_one_task(text):
        clean = inference.minimal_clean(text)
        encoded = inference._tokenizer(clean, padding="max_length", truncation=True, max_length=128, return_tensors="np")
        feeds = {k: encoded[k].astype(np.int64) for k in encoded if k in input_names}
        return inference._session.run(None, feeds)[0]

    print(f"Running threaded bench for {len(texts)} samples with {num_threads} threads...")
    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(executor.map(predict_one_task, texts))
    end = time.time()
    print(f"Threaded time: {end - start:.4f}s ({ (end-start)/len(texts):.4f}s per sample)")

if __name__ == "__main__":
    bench_threaded(4)
