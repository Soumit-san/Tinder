import time
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.getcwd())
from backend import inference

def bench():
    print("Loading model...")
    inference.load_model()
    
    texts = ["I love this app! it is great but sometimes it crashes."] * 100
    
    print(f"Running sequential bench for {len(texts)} samples...")
    start = time.time()
    inference.predict_batch(texts)
    end = time.time()
    print(f"Sequential time: {end - start:.4f}s ({ (end-start)/len(texts):.4f}s per sample)")

if __name__ == "__main__":
    bench()
