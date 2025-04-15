#!/usr/bin/env python
"""
analyze_mtl_results.py

This script loads the MTL results from a pickle file (expected to have keys
'ground_truth' and 'predictions') and computes additional metrics and plots:
  1. Overall accuracy, confusion matrix, and classification report.
  2. Bar plot of per-class F1 scores.
  3. (Optional) Learning curves: Training loss and accuracy vs. epochs, if available.
  4. Distribution plots of ground truth and predicted labels.

These figures are saved to PNG files.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_mtl_results(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def compute_overall_metrics(ground_truth, predictions):
    """
    Compute accuracy, confusion matrix, and classification report.
    Returns:
      - accuracy (float)
      - confusion matrix (np.array)
      - report string
      - report dictionary (from classification_report with output_dict=True)
    """
    acc = accuracy_score(ground_truth, predictions)
    cm = confusion_matrix(ground_truth, predictions)
    report_str = classification_report(ground_truth, predictions, zero_division=0)
    report_dict = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)
    return acc, cm, report_str, report_dict

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, save_path="mtl_confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {title} plot to {save_path}")

def plot_per_class_f1(report_dict, save_path="mtl_per_class_f1.png"):
    """
    Extract F1-scores for each numeric class (ignoring summary keys)
    and create a bar plot.
    """
    # Filter keys that are digits (i.e., the class labels)
    classes = [k for k in report_dict.keys() if k.isdigit()]
    f1_scores = [report_dict[k]['f1-score'] for k in classes]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=classes, y=f1_scores, palette="viridis")
    plt.title("Per-Class F1 Scores")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Per-Class F1 Scores bar plot to {save_path}")

def plot_learning_curves(training_logs, loss_path="learning_curve_loss.png", acc_path="learning_curve_accuracy.png"):
    """
    If training logs (with keys 'loss' and 'accuracy') are provided,
    plot learning curves over epochs.
    """
    if training_logs is None or not training_logs:
        print("[INFO] No training logs available to plot learning curves.")
        return

    epochs = range(1, len(training_logs.get("loss", [])) + 1)
    loss = training_logs.get("loss", [])
    accuracy = training_logs.get("accuracy", [])
    
    if loss:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss, marker='o', label='Loss')
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()
        print(f"Saved Training Loss curve to {loss_path}")
        
    if accuracy:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, accuracy, marker='o', color='green', label='Accuracy')
        plt.title("Training Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_path)
        plt.close()
        print(f"Saved Training Accuracy curve to {acc_path}")

def plot_label_distribution(ground_truth, predictions, save_path="mtl_distribution.png"):
    """
    Create a histogram/bar plot comparing the distribution of ground truth and
    predicted labels.
    """
    plt.figure(figsize=(8, 6))
    # Create DataFrames for easier plotting
    df_gt = pd.DataFrame({"Label": ground_truth, "Type": "Ground Truth"})
    df_pred = pd.DataFrame({"Label": predictions, "Type": "Prediction"})
    df = pd.concat([df_gt, df_pred])
    
    sns.countplot(x="Label", hue="Type", data=df, palette="Set2")
    plt.title("Distribution of Ground Truth and Predictions")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Label Distribution plot to {save_path}")

def main():
    mtl_results_file = "mtl_training_results.pkl"  # Adjust path if needed.
    results = load_mtl_results(mtl_results_file)
    
    # Check for training logs (if saved)
    if "training_logs" in results:
        training_logs = results["training_logs"]
    else:
        training_logs = None
        print("[INFO] No training_logs attribute found in MTL results.")
    
    # In your file, the results contain flat keys 'ground_truth' and 'predictions'
    ground_truth = results.get("ground_truth", [])
    predictions = results.get("predictions", [])
    
    if not ground_truth or not predictions:
        print("Error: No ground truth or predictions found in the results.")
        return
    
    # Compute overall metrics.
    acc, cm, report_str, report_dict = compute_overall_metrics(ground_truth, predictions)
    print("Overall Accuracy: {:.4f}".format(acc))
    print("Classification Report:\n", report_str)
    
    # Determine the class labels from ground_truth (assumed to be numeric or string)
    classes = sorted(list(set(ground_truth)))
    classes = [str(c) for c in classes]
    
    # Plot and save confusion matrix.
    plot_confusion_matrix(cm, classes, title="MTL Confusion Matrix", save_path="mtl_confusion_matrix.png")
    
    # Plot and save per-class F1 scores.
    plot_per_class_f1(report_dict, save_path="mtl_per_class_f1.png")
    
    # Plot and save learning curves if available.
    plot_learning_curves(training_logs, loss_path="learning_curve_loss.png", acc_path="learning_curve_accuracy.png")
    
    # Plot and save label distribution.
    plot_label_distribution(ground_truth, predictions, save_path="mtl_distribution.png")
    
if __name__ == "__main__":
    main()
