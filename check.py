#!/usr/bin/env python3
import pickle
import numpy as np
import sys

def print_summary(data, prefix=""):
    t = type(data)
    # Print basic info
    if isinstance(data, dict):
        print(f"{prefix}dict with keys: {list(data.keys())}")
        for k, v in data.items():
            print_summary(v, prefix + f"  [{k!r}] ")
    elif isinstance(data, (list, tuple, set)):
        print(f"{prefix}{t.__name__} of length {len(data)}")
        for i, v in enumerate(data):
            print_summary(v, prefix + f"  [{i}] ")
    elif isinstance(data, np.ndarray):
        print(f"{prefix}ndarray, shape={data.shape}, dtype={data.dtype}")
    else:
        # fallback for scalars or unknown types
        summary = repr(data)
        if len(summary) > 200:
            summary = summary[:200] + "…"
        print(f"{prefix}{t.__name__}: {summary}")

def main(path):
    print(f"Loading: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("\n--- Top-level summary ---")
    print_summary(data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_features.py path/to/dump/features.pkl")
        sys.exit(1)
    main(sys.argv[1])
