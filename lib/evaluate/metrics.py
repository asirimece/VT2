import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_curve, auc

class MetricsEvaluator:
    def __init__(self, config):
        """
        config: dict containing keys like "metrics" (a list)
        """
        self.config = config
        self.metrics = config.get("metrics", [])
    
    def evaluate(self, ground_truth, predictions, probabilities=None):
        """
        Computes quantitative metrics.
        :param ground_truth: true labels (array-like)
        :param predictions: predicted labels (array-like)
        :param probabilities: predicted probabilities (2D array) for ROC curve (optional)
        :return: dict of metric results
        """
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
                fpr, tpr, thresholds = roc_curve(binary_truth, probabilities[:, cls])
                roc_results[cls] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc(fpr, tpr)}
            results["roc_curve"] = roc_results
        return results
