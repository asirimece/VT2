# eval_pipeline.py

import os
import json
import numpy as np
import joblib
import mne
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from lib.tl.recorder import Deep4NetTLWrapper

def load_epochs(fif_path):
    """Load raw .fif, extract epochs for left/right (0/1) and rest=2 if desired."""
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    events, event_id = mne.events_from_annotations(raw)
    # map your markers: here assuming annotation → integer already
    picks = ['C3','Cz','C4','Fz','Pz','Oz','PO7','PO8']
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=-1.0, tmax=2.0,
        picks=picks, baseline=None,
        preload=True
    )
    return epochs

def main():
    # 1) Paths
    model_path = "dump/trained_models/tl/tl_pooled_model.joblib"
    test_fif   = "data/04_sampleFreq200_80_events/recording_subject_300_session_1_raw.fif"   # adjust to your actual file

    # 2) Load pipeline
    pipe = joblib.load(model_path)

    # 3) Load real epochs
    if not os.path.exists(test_fif):
        raise FileNotFoundError(test_fif)
    epochs = load_epochs(test_fif)

    X_test = epochs.get_data()              # shape (n_epochs, 8, 600)
    y_true = epochs.events[:, -1]           # your ground-truth labels

    # 4) Predict and evaluate
    y_pred = pipe.predict(X_test)
    acc    = accuracy_score(y_true, y_pred)
    print(f"Accuracy on {len(y_true)} epochs: {acc:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # 5) Decision-confidence histogram
    proba = pipe.predict_proba(X_test).max(axis=1)
    plt.hist(proba, bins=20, edgecolor='k')
    plt.title("Distribution of max-class probabilities")
    plt.xlabel("Max class probability")
    plt.ylabel("Count of epochs")
    plt.show()

if __name__ == "__main__":
    main()
