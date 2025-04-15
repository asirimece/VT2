#!/usr/bin/env python
import os
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize

# Import TLSubjectDataset and TLModel from your library
from lib.tl.dataset import TLSubjectDataset
from lib.tl.model import TLModel
from lib.tl.evaluator import TLEvaluator

###############################
# Plotting helper functions
###############################
def plot_multiclass_roc(y_true, y_prob, n_classes, class_names=None):
    """
    Plot ROC curve(s) for each class in a one-vs-rest manner.
    """
    # Binarize the true labels for multi-class ROC computation
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    
    plt.figure()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        label_str = f"Class {class_names[i] if class_names else i} (AUC = {roc_auc[i]:0.2f})"
        plt.plot(fpr[i], tpr[i], lw=2, label=label_str)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curves")
    plt.legend(loc="lower right")
    plt.show()

def plot_multiclass_pr(y_true, y_prob, n_classes, class_names=None):
    """
    Plot Precision-Recall curves for each class.
    """
    # Binarize the true labels for multi-class PR computation
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    precision, recall, pr_auc = dict(), dict(), dict()
    
    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
        label_str = f"Class {class_names[i] if class_names else i} (AUC = {pr_auc[i]:0.2f})"
        plt.plot(recall[i], precision[i], lw=2, label=label_str)
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.show()

###############################
# Main evaluation function
###############################
def main():
    # --- Configuration (adjust as needed) ---
    device = "cpu"
    # Update the following paths to the correct locations on your system:
    pretrained_model_path = "tl_outputs/tl_1_model.pth"         # Path to the TL checkpoint
    preprocessed_data_path = "./outputs/preprocessed_data.pkl"  # Path to your preprocessed_data.pkl file
    subject_id = 1   # Identifier of the new subject; note that the model uses an integer cluster id.
    
    # Specify model hyperparameters (should be consistent with your training config)
    n_outputs = 4                  # e.g., number of classes (4 for BCI Competition 2a)
    n_clusters_pretrained = 8      # from your configuration
    # We assume the test data has shape (n_trials, n_channels, n_times)
    
    # --- Load preprocessed data ---
    with open(preprocessed_data_path, "rb") as f:
        preprocessed_data = pickle.load(f)
    print(f"[DEBUG] Preprocessed data loaded from {preprocessed_data_path}")
    
    if subject_id not in preprocessed_data:
        raise ValueError(f"Subject '{subject_id}' not found in the preprocessed data!")
    
    subject_data = preprocessed_data[subject_id]
    test_ep = subject_data["1test"]
    X_test = test_ep.get_data()
    y_test = test_ep.events[:, -1]
    
    # Determine shape parameters from test data.
    n_chans = X_test.shape[1]
    window_samples = X_test.shape[2]
    
    print(f"[DEBUG] X_test shape: {X_test.shape}")
    print(f"[DEBUG] y_test shape: {y_test.shape}")
    print(f"[DEBUG] Number of channels: {n_chans}")
    print(f"[DEBUG] Window samples: {window_samples}")
    
    # --- Build and load TLModel ---
    tl_model = TLModel(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_clusters_pretrained=n_clusters_pretrained,
        window_samples=window_samples
    )
    print("[DEBUG] TLModel instance created.")
    
    # Load the pretrained model's state dict.
    state_dict = torch.load(pretrained_model_path, map_location=device)
    tl_model.load_state_dict(state_dict)
    print(f"[DEBUG] Pretrained model loaded from {pretrained_model_path}")
    
    # --- Add a new head for the subject ---
    new_cluster_id = int(subject_id)
    feature_dim = 4  # From previous backbone computation or manually set
    print(f"[DEBUG] Adding new head for cluster id: {new_cluster_id} with feature dimension: {feature_dim}")
    tl_model.add_new_head(new_cluster_id, feature_dim=feature_dim)
    
    # Set the model to evaluation mode
    tl_model.to(device)
    tl_model.eval()
    print("Model loaded and set to evaluation mode.")
    
    # --- Prepare the DataLoader for test data ---
    test_dataset = TLSubjectDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("[DEBUG] Test DataLoader created.")
    
    # --- Run Inference to Obtain Predictions and Probabilities ---
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = tl_model(X_batch, [new_cluster_id] * len(X_batch))
            # Compute softmax probabilities
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # --- Compute and Print Metrics ---
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    
    # Use the evaluator to compute other metrics
    evaluator = TLEvaluator()
    # Wrap the results in a simple object for evaluator.evaluate (it expects attributes ground_truth and predictions)
    class TLWrapperCustom:
        def __init__(self, ground_truth, predictions, y_prob):
            self.ground_truth = ground_truth
            self.predictions = predictions
            self.y_prob = y_prob  # Added probability attribute
    
    tl_results = TLWrapperCustom(all_labels, all_preds, all_probs)
    metrics = evaluator.evaluate(tl_results, class_names=[str(i) for i in range(n_outputs)])
    print("[TL Evaluation] =>", metrics)
    
    # --- Plot Additional Evaluation Figures ---
    # Plot ROC curves (one-vs-all) for multiclass classification:
    plot_multiclass_roc(all_labels, all_probs, n_classes=n_outputs, class_names=[str(i) for i in range(n_outputs)])
    
    # Plot Precision-Recall curves:
    plot_multiclass_pr(all_labels, all_probs, n_classes=n_outputs, class_names=[str(i) for i in range(n_outputs)])
    
if __name__ == "__main__":
    main()
