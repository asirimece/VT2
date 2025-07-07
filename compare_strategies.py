import os
import pickle
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lib.utils.recorder import Deep4NetTLWrapper

PREPROC_PATH       = "dump/preprocessed_data_custom.pkl"
BEST_JOBLIB        = "dump/trained_models/tl/best_run.joblib"
SWA_JOBLIB         = "dump/trained_models/tl/swa.joblib"
UNIVERSAL_JOBLIB   = "dump/trained_models/tl/universal_autostop.joblib"
OUTPUT_LOG         = "comparison_results.txt"

def load_pipelines():
    best       = joblib.load(BEST_JOBLIB)
    swa        = joblib.load(SWA_JOBLIB)
    universal  = joblib.load(UNIVERSAL_JOBLIB)
    # no-op fit to silence warnings
    best.fit(None); swa.fit(None); universal.fit(None)
    return {"Best-Run": best, "SWA": swa, "Universal": universal}

def main(out_stream):
    # load your TLSubjectDataset splits
    with open(PREPROC_PATH, "rb") as f:
        data = pickle.load(f)

    pipes = load_pipelines()
    # dict to accumulate accuracies
    accs = {name: [] for name in pipes}

    for sub in sorted(data.keys()):
        test_ds = data[sub]["test"]
        X_te     = test_ds.get_data()
        y_te     = test_ds.events[:, -1]

        print(f"\n\n=== Subject {sub} ===", file=out_stream)
        for name, pipe in pipes.items():
            y_pred = pipe.predict(X_te)
            acc    = accuracy_score(y_te, y_pred)
            accs[name].append(acc)

            print(f"\n--- {name} Strategy ---", file=out_stream)
            print(f"Accuracy: {acc:.3f}", file=out_stream)
            print("Classification Report:", file=out_stream)
            print(classification_report(y_te, y_pred, zero_division=0), file=out_stream)
            print("Confusion Matrix:", file=out_stream)
            print(confusion_matrix(y_te, y_pred), file=out_stream)

    # After per‐subject loop, print averages
    print("\n\n=== Average Accuracy Across All Subjects ===", file=out_stream)
    for name, scores in accs.items():
        mean_acc = np.mean(scores) if scores else float("nan")
        print(f"{name}: {mean_acc:.3f}", file=out_stream)

if __name__ == "__main__":
    # open the log file and redirect output there
    with open(OUTPUT_LOG, "w") as f:
        main(f)
    print(f"Comparison results (including averages) written to {OUTPUT_LOG}")
