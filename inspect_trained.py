import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from lib.base.trainer import BaseWrapper
import pprint

# Load the training results
with open("trained_models/training_results.pkl", "rb") as f:
    results = pickle.load(f)

# Basic inspection of the results object
print("Type of results object:", type(results))
print("\nFull results object dictionary:")
pprint.pprint(results.__dict__)

# Check if results_by_subject exists
if hasattr(results, 'results_by_subject'):
    subjects = list(results.results_by_subject.keys())
    print("\nSubjects available in results.results_by_subject:")
    print(subjects)
    
    all_gt = []
    all_preds = []
    
    # Iterate over each subject's results
    for subj, subj_dict in results.results_by_subject.items():
        print(f"\nInspecting subject '{subj}':")
        pprint.pprint(subj_dict)
        
        # Check for ground_truth and predictions keys
        if "ground_truth" not in subj_dict or "predictions" not in subj_dict:
            print(f"  Warning: 'ground_truth' or 'predictions' key missing for subject {subj}")
            continue

        gt = subj_dict["ground_truth"]
        preds = subj_dict["predictions"]
        
        print("  Ground truth type:", type(gt))
        print("  Predictions type:", type(preds))
        print("  Unique values in ground truth:", np.unique(gt))
        
        if isinstance(preds, list):
            print("  Predictions is a list with", len(preds), "runs.")
            run_accuracies = []
            for i, run_preds in enumerate(preds):
                print(f"    Run {i} unique predictions:", np.unique(run_preds))
                acc = accuracy_score(gt, run_preds)
                run_accuracies.append(acc)
                print(f"    Run {i} Accuracy: {acc:.4f}")
            avg_acc = np.mean(run_accuracies)
            print(f"  Average Accuracy for subject '{subj}': {avg_acc:.4f}")
            subject_preds = np.concatenate(preds)
        else:
            print("  Unique values in predictions:", np.unique(preds))
            overall_acc = accuracy_score(gt, preds)
            print(f"  Overall Accuracy for subject '{subj}': {overall_acc:.4f}")
            subject_preds = preds

        all_gt.extend(gt)
        all_preds.extend(subject_preds)
else:
    print("No 'results_by_subject' attribute found. Assuming pooled training.")
    gt = getattr(results, "ground_truth", None)
    preds = getattr(results, "predictions", None)
    if gt is None or preds is None:
        raise ValueError("For pooled training, the results must contain 'ground_truth' and 'predictions'.")
    print("Ground truth type:", type(gt))
    print("Predictions type:", type(preds))
    print("Unique values in ground truth:", np.unique(gt))
    print("Unique values in predictions:", np.unique(preds))
    overall_acc = accuracy_score(gt, preds)
    print(f"Pooled Training Overall Accuracy: {overall_acc:.4f}")
    all_gt = gt
    all_preds = preds

all_gt = np.array(all_gt)
all_preds = np.array(all_preds)

# Compute confusion matrix and classification report
cm = confusion_matrix(all_gt, all_preds)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(all_gt, all_preds))

# Plot and save the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
