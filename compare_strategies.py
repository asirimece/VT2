import mne
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Load your held-out Raw and epoch it exactly as in your preprocessing:
raw = mne.io.read_raw_fif(
    "data/04_sampleFreq200_80_events/recording_subject_85_session_1_raw.fif",
    preload=True
)
events, _ = mne.events_from_annotations(raw)
epochs = mne.Epochs(
    raw, events,
    event_id={'0':0,'1':1,'2':2},
    tmin=-1.0, tmax=2.0,
    baseline=None, preload=True
)
X = epochs.get_data()                  # shape (n_epochs, 8, 600)
y_true = epochs.events[:, -1]          # labels in {0,1,2}

# 2) Load both pipelines:
best_pipe = joblib.load("dump/trained_models/tl/best_run.joblib")
swa_pipe  = joblib.load("dump/trained_models/tl/swa.joblib")

# 3) Predict & evaluate:
for name, pipe in [("Best-Run", best_pipe), ("SWA", swa_pipe)]:
    y_pred    = pipe.predict(X)
    y_proba   = pipe.predict_proba(X).max(axis=1)
    acc       = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} Strategy ===")
    print(f"Accuracy: {acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    # Optionally inspect confidence distribution:
    print(f"Mean confidence: {y_proba.mean():.3f}\n")
