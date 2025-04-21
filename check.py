#!/usr/bin/env python
import os
import pickle
from sklearn.metrics import accuracy_score

# Adjust as needed
SUBJECTS = list(range(1, 10))
N_RUNS    = 3
BASE_DIR  = "models/tl/cluster0"

missing = []
accs = {run: {} for run in range(N_RUNS)}

for run in range(N_RUNS):
    run_dir = os.path.join(BASE_DIR, f"run_{run}")
    if not os.path.isdir(run_dir):
        print(f"⚠️  Missing run directory: {run_dir}")
        continue

    for subj in SUBJECTS:
        pkl_path   = os.path.join(run_dir, f"tl_{subj}_results.pkl")
        model_path = os.path.join(run_dir, f"tl_{subj}_model.pth")

        # Check existence
        if not os.path.isfile(pkl_path):
            missing.append(pkl_path)
            continue
        if not os.path.isfile(model_path):
            missing.append(model_path)
            continue

        # Load wrapper and compute accuracy
        wrapper = pickle.load(open(pkl_path, "rb"))
        gt = wrapper.ground_truth
        pr = wrapper.predictions
        acc = accuracy_score(gt, pr)
        accs[run][subj] = acc

print("\n=== Missing files ===")
if missing:
    for path in missing:
        print(" ", path)
else:
    print("  (none)")

print("\n=== Scratch TL accuracies ===")
for run in range(N_RUNS):
    print(f"\n-- Run {run} --")
    for subj in SUBJECTS:
        if subj in accs[run]:
            print(f"Subject {subj}: acc = {accs[run][subj]:.3f}")
        else:
            print(f"Subject {subj}: MISSING")

# Optionally print run‐means
print("\n=== Mean accuracy per run ===")
for run in range(N_RUNS):
    vals = list(accs[run].values())
    if vals:
        print(f"Run {run}: mean acc = {sum(vals)/len(vals):.3f}")
    else:
        print(f"Run {run}: no data")
