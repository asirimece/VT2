import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from lib.train.train import TrainingResults

# Load the training results
with open("trained_models/training_results.pkl", "rb") as f:
    results = pickle.load(f)

all_gt = []
all_preds = []
n_runs = 5  # number of runs per subject

for subj, subj_dict in results.results_by_subject.items():
    gt = subj_dict.get("ground_truth")
    preds = subj_dict.get("predictions")
    
    # If predictions are stored as a list (one per run), compute per-run accuracies.
    if isinstance(preds, list):
        run_accuracies = [accuracy_score(gt, run_preds) for run_preds in preds]
        avg_acc = np.mean(run_accuracies)
        print(f"Subject {subj}:")
        print(f"  Run Accuracies: {run_accuracies}")
        print(f"  Average Accuracy: {avg_acc:.4f}\n")
        subject_preds = np.concatenate(preds)
    # Otherwise, assume predictions are already aggregated for the subject.
    else:
        overall_acc = accuracy_score(gt, preds)
        print(f"Subject {subj}: Overall Accuracy: {overall_acc:.4f}\n")
        subject_preds = preds

    all_gt.extend(gt)
    all_preds.extend(subject_preds)

all_gt = np.array(all_gt)
all_preds = np.array(all_preds)

# Compute confusion matrix and classification report
cm = confusion_matrix(all_gt, all_preds)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(all_gt, all_preds))

#  Plot and save the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
