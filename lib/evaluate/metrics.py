# lib/evaluate/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    auc,
)

class MetricsEvaluator:
    """
    Compute a configurable set of quantitative metrics for a single run.
    Config dict should contain:
      - "metrics": list of strings from ["accuracy","kappa","confusion_matrix","roc_curve"]
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
        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(ground_truth, predictions)
        if "kappa" in self.metrics:
            results["kappa"] = cohen_kappa_score(ground_truth, predictions)
        if "confusion_matrix" in self.metrics:
            results["confusion_matrix"] = confusion_matrix(ground_truth, predictions)
        if "roc_curve" in self.metrics and probabilities is not None:
            roc_results = {}
            classes = np.unique(ground_truth)
            for cls in classes:
                binary_truth = (ground_truth == cls).astype(int)
                fpr, tpr, thresh = roc_curve(binary_truth, probabilities[:, cls])
                roc_results[int(cls)] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresh,
                    "auc": auc(fpr, tpr),
                }
            results["roc_curve"] = roc_results
        return results
