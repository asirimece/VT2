# debug_mtl_results.py

import pickle

# debug_mtl_results.py

import pickle
import numpy as np

def debug_mtl_results(filename="mtl_training_results.pkl"):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    print("Type of loaded object:", type(obj))
    
    # If the object is an instance of MTLWrapper (or a dict with key 'results_by_subject')
    if isinstance(obj, dict):
        if "results_by_subject" in obj:
            print("Keys in MTLWrapper dict:", list(obj.keys()))
            rbs = obj["results_by_subject"]
            print("Type of results_by_subject:", type(rbs))
            if isinstance(rbs, dict):
                for subj, res in rbs.items():
                    print(f"\nSubject: {subj} (type: {type(res)})")
                    if isinstance(res, dict):
                        keys = list(res.keys())
                        print("  Keys:", keys)
                        for k in keys:
                            val = res[k]
                            # Print only first 10 elements if it's an array or list
                            if isinstance(val, (list, np.ndarray)):
                                preview = np.array(val)[:10]
                                print(f"    {k}: type {type(val)}, first 10 elements: {preview}")
                            else:
                                print(f"    {k}: {val}")
                    elif isinstance(res, list):
                        print("  Result is a list of length:", len(res))
                        if len(res) > 0 and isinstance(res[0], dict):
                            print("  Keys in first element:", list(res[0].keys()))
                    else:
                        print("  Result:", res)
            else:
                print("results_by_subject is not a dict:", rbs)
        else:
            # If the dict doesn't have 'results_by_subject'
            print("Loaded dict keys:", list(obj.keys()))
    elif isinstance(obj, list):
        print("Loaded object is a list of length:", len(obj))
        if len(obj) > 0:
            print("Type of first element:", type(obj[0]))
            if isinstance(obj[0], dict):
                print("Keys in first element:", list(obj[0].keys()))
            else:
                print("First 10 elements:", obj[:10])
    else:
        print("Loaded object:", obj)

if __name__ == "__main__":
    debug_mtl_results()
