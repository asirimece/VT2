#!/usr/bin/env python3
"""
inspect_pkl.py

Robustly inspect the contents of a pickle file containing preprocessed EEG epochs.
Prints top-level type, keys (if dict), and shapes/types of nested objects.
Usage:
    python inspect_pkl.py /path/to/preprocessed_data_custom.pkl
"""

import sys
import pickle
import numpy as np

def describe_obj(name, obj, indent=0):
    prefix = " " * indent
    if isinstance(obj, np.ndarray):
        print(f"{prefix}{name}: numpy.ndarray, shape={obj.shape}, dtype={obj.dtype}")
    elif hasattr(obj, "get_data"):
        # MNE Epochs or similar
        try:
            data = obj.get_data()
            print(f"{prefix}{name}: {type(obj).__name__}, get_data() -> shape={data.shape}")
        except Exception:
            print(f"{prefix}{name}: {type(obj).__name__}, with get_data() method (could not retrieve shape)")
    elif isinstance(obj, dict):
        print(f"{prefix}{name}: dict with {len(obj)} keys")
        for i, key in enumerate(obj):
            if i >= 5:
                print(f"{prefix}  ... ({len(obj)-5} more keys)")
                break
            describe_obj(f"{name}[{repr(key)}]", obj[key], indent + 4)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: {type(obj).__name__} of length {len(obj)}")
        for i, item in enumerate(obj[:5]):
            describe_obj(f"{name}[{i}]", item, indent + 4)
        if len(obj) > 5:
            print(f"{prefix}  ... ({len(obj)-5} more items)")
    else:
        print(f"{prefix}{name}: {type(obj).__name__} (no further introspection)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_pkl.py /path/to/preprocessed_data_custom.pkl")
        sys.exit(1)

    pkl_path = sys.argv[1]
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        sys.exit(1)

    print(f"Top-level object type: {type(data).__name__}")
    describe_obj("data", data)

if __name__ == "__main__":
    main()
