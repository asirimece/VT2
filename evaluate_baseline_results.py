#!/usr/bin/env python
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from lib.base.trainer import BaseWrapper

def load_baseline_results(filename):
    """
    Loads the baseline results from a pickle file. The file might be an instance of a wrapper
    class (having an attribute 'results_by_subject') or simply a dictionary with keys
    'ground_truth' and 'predictions'. This function aggregates the results into two arrays.
    """
    with open(filename, "rb") as f:
        results = pickle.load(f)
        
    all_gt = []
    all_pred = []
    
    # If the loaded object has attribute 'results_by_subject'
    if hasattr(results, "results_by_subject"):
        for subj, res in results.results_by_subject.items():
            # If result is a list, assume the last element is the final result.
            if isinstance(res, list):
                res = res[-1]
            if isinstance(res, dict) and "ground_truth" in res and "predictions" in res:
                all_gt.extend(res["ground_truth"])
                all_pred.extend(res["predictions"])
    elif isinstance(results, dict):
        # If the dict directly has the keys "ground_truth", "predictions"
        if "ground_truth" in results and "predictions" in results:
            all_gt = results["ground_truth"]
            all_pred = results["predictions"]
        else:
            # Otherwise, assume it is a dict mapping subject IDs to results.
            for subj, res in results.items():
                if isinstance(res, list):
                    res = res[-1]
                if isinstance(res, dict) and "ground_truth" in res and "predictions" in res:
                    all_gt.extend(res["ground_truth"])
                    all_pred.extend(res["predictions"])
    else:
        raise ValueError("Unknown result type in file: " + filename)
        
    return np.array(all_gt), np.array(all_pred)

def compute_metrics(gt, pred):
    """
    Compute overall accuracy, confusion matrix, and classification report.
    """
    accuracy = accuracy_score(gt, pred)
    cm = confusion_matrix(gt, pred)
    report = classification_report(gt, pred, zero_division=0)
    return accuracy, cm, report

def plot_confusion_matrix(cm, title, output_file):
    """
    Plots and saves a confusion matrix using seaborn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {title} plot to {output_file}")

def plot_classification_report(report_dict, title, output_file):
    """
    Given the classification report in dict format, plots a bar plot of the F1-scores per class.
    Only classes that can be cast as a digit are plotted (assuming class labels are digits).
    """
    classes = [cls for cls in report_dict.keys() if cls.isdigit()]
    f1_scores = [report_dict[cls]["f1-score"] for cls in classes]
    
    plt.figure(figsize=(8, 6))
    plt.bar(classes, f1_scores, color='green')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("F1-score")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {title} bar plot to {output_file}")

def main():
    # Define file paths for baseline results.
    pooled_baseline_path = "./trained_models/pooled_baseline_results.pkl"
    single_baseline_path = "./trained_models/single_baseline_results.pkl"
    
    # Create a folder to save plots.
    os.makedirs("baseline_plots", exist_ok=True)
    
    # Process the pooled baseline results.
    gt_pooled, pred_pooled = load_baseline_results(pooled_baseline_path)
    acc_pooled, cm_pooled, report_pooled_str = compute_metrics(gt_pooled, pred_pooled)
    print("Pooled Baseline Accuracy:", acc_pooled)
    print("Pooled Baseline Classification Report:\n", report_pooled_str)
    
    # Save pooled baseline plots.
    plot_confusion_matrix(cm_pooled, "Pooled Baseline Confusion Matrix", "baseline_plots/pooled_confusion_matrix.png")
    # Convert report to dict for bar plot.
    report_pooled_dict = classification_report(gt_pooled, pred_pooled, zero_division=0, output_dict=True)
    plot_classification_report(report_pooled_dict, "Pooled Baseline F1-scores", "baseline_plots/pooled_f1_scores.png")
    
    # Process the single-subject baseline results.
    gt_single, pred_single = load_baseline_results(single_baseline_path)
    acc_single, cm_single, report_single_str = compute_metrics(gt_single, pred_single)
    print("Single-Subject Baseline Accuracy:", acc_single)
    print("Single-Subject Baseline Classification Report:\n", report_single_str)
    
    # Save single-subject baseline plots.
    plot_confusion_matrix(cm_single, "Single-Subject Baseline Confusion Matrix", "baseline_plots/single_confusion_matrix.png")
    report_single_dict = classification_report(gt_single, pred_single, zero_division=0, output_dict=True)
    plot_classification_report(report_single_dict, "Single-Subject Baseline F1-scores", "baseline_plots/single_f1_scores.png")

if __name__ == "__main__":
    # Importing classification_report with output_dict=True.
    from sklearn.metrics import classification_report
    main()
