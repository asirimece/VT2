# lib/evaluate/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)

class MetricsEvaluator:
    """
    Compute a configurable set of quantitative metrics for a single run.
    Config dict should contain:
      - "metrics": list of strings from
         ["accuracy","kappa","precision","recall","f1_score",
          "confusion_matrix","roc_curve"]
    """
    def __init__(self, config: dict):
        self.metrics = config.get("metrics", [])
        print(f"[DEBUG] MetricsEvaluator: using metrics {self.metrics}")

    def evaluate(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray = None
    ) -> dict:
        results = {}
        gt = ground_truth
        pr = predictions

        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(gt, pr)
        if "kappa" in self.metrics:
            results["kappa"] = cohen_kappa_score(gt, pr)
        if "precision" in self.metrics:
            # macro-average across classes
            results["precision"] = precision_score(gt, pr, average="macro", zero_division=0)
        if "recall" in self.metrics:
            results["recall"] = recall_score(gt, pr, average="macro", zero_division=0)
        if "f1_score" in self.metrics:
            results["f1_score"] = f1_score(gt, pr, average="macro", zero_division=0)
        if "confusion_matrix" in self.metrics:
            results["confusion_matrix"] = confusion_matrix(gt, pr)
        if "roc_curve" in self.metrics and probabilities is not None:
            roc_results = {}
            classes = np.unique(gt)
            for cls in classes:
                binary_truth = (gt == cls).astype(int)
                fpr, tpr, thresh = roc_curve(binary_truth, probabilities[:, cls])
                roc_results[int(cls)] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresh,
                    "auc": auc(fpr, tpr),
                }
            results["roc_curve"] = roc_results

        return results
