# mtlevaluator.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lib.mtl.trainer import MTLWrapper
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class MTLEvaluator:
    def __init__(self, mtl_wrapper, baseline_wrapper, config):
        """
        Initializes the evaluator.
        
        Parameters:
          mtl_wrapper (MTLWrapper): Wrapped MTL training results.
              Expected to have:
                - results_by_subject: dict mapping subject IDs (or "pooled") to a dict with keys "ground_truth" and "predictions"
                - cluster_assignments: (optional) dict mapping subject IDs to cluster IDs.
          baseline_wrapper (MTLWrapper or dict): Wrapped baseline results with a similar interface.
          config (dict): Evaluation configuration (loaded from YAML via OmegaConf) for additional parameters.
        """
        self.mtl_wrapper = mtl_wrapper
        # If baseline_wrapper is a plain dict, wrap it.
        if isinstance(baseline_wrapper, dict):
            self.baseline_wrapper = type(mtl_wrapper)(results_by_subject=baseline_wrapper)
        else:
            self.baseline_wrapper = baseline_wrapper
        self.config = config

    @staticmethod
    def load_results(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        # If obj is already an instance of MTLWrapper, return it.
        if hasattr(obj, "results_by_subject"):
            return obj
        # If it's a dict with "ground_truth" and "predictions", wrap under key "pooled".
        if isinstance(obj, dict) and set(obj.keys()) == {"ground_truth", "predictions"}:
            wrapped = {"pooled": obj}
            print("[DEBUG] Loaded results as dict with keys ['ground_truth', 'predictions']. Wrapping under key 'pooled'.")
            from mtl_wrapper import MTLWrapper
            return MTLWrapper(results_by_subject=wrapped)
        # If it's a dict mapping subject IDs, wrap it.
        if isinstance(obj, dict):
            from mtl_wrapper import MTLWrapper
            return MTLWrapper(results_by_subject=obj)
        # If it's a list, wrap it as pooled.
        if isinstance(obj, list):
            from mtl_wrapper import MTLWrapper
            return MTLWrapper(results_by_subject={"pooled": obj})
        return obj

    def compute_overall_metrics(self, wrapper):
        """Compute overall accuracy, confusion matrix, and classification report from a wrapper."""
        all_gt = []
        all_pred = []
        for subj, res in wrapper.results_by_subject.items():
            if isinstance(res, list):
                res = res[-1]
            if not (isinstance(res, dict) and "ground_truth" in res and "predictions" in res):
                print(f"Warning: Results for subject {subj} are not in the expected format; skipping.")
                continue
            all_gt.extend(res["ground_truth"])
            all_pred.extend(res["predictions"])
        all_gt = np.array(all_gt)
        all_pred = np.array(all_pred)
        overall_acc = accuracy_score(all_gt, all_pred)
        overall_cm = confusion_matrix(all_gt, all_pred)
        overall_report = classification_report(all_gt, all_pred, zero_division=0)
        return {"accuracy": overall_acc, "confusion_matrix": overall_cm, "report": overall_report}

    def compute_cluster_metrics(self, wrapper):
        """
        Computes cluster-level metrics by grouping subjects based on cluster assignments.
        Returns a dict mapping each cluster to its metrics.
        """
        cluster_data = {}
        if not hasattr(wrapper, "cluster_assignments") or not wrapper.cluster_assignments:
            return {}
        for subj, res in wrapper.results_by_subject.items():
            if isinstance(res, list):
                res = res[-1]
            if not (isinstance(res, dict) and "ground_truth" in res and "predictions" in res):
                continue
            cl = wrapper.cluster_assignments.get(subj, "None")
            cluster_data.setdefault(cl, {"ground_truth": [], "predictions": []})
            cluster_data[cl]["ground_truth"].extend(res["ground_truth"])
            cluster_data[cl]["predictions"].extend(res["predictions"])
        cluster_metrics = {}
        for cl, data in cluster_data.items():
            gt = np.array(data["ground_truth"])
            preds = np.array(data["predictions"])
            acc = accuracy_score(gt, preds)
            cm = confusion_matrix(gt, preds)
            report = classification_report(gt, preds, zero_division=0)
            cluster_metrics[cl] = {"accuracy": acc, "confusion_matrix": cm, "report": report}
        return cluster_metrics

    def compute_subject_metrics(self):
        """
        Computes subject-level metrics and returns a DataFrame.
        Each record includes: subject, cluster, baseline_accuracy, mtl_accuracy, difference.
        """
        records = []
        baseline_dict = (self.baseline_wrapper.results_by_subject 
                         if hasattr(self.baseline_wrapper, "results_by_subject") 
                         else self.baseline_wrapper)
        for subj, res in self.mtl_wrapper.results_by_subject.items():
            if isinstance(res, list):
                res = res[-1]
            if not (isinstance(res, dict) and "ground_truth" in res and "predictions" in res):
                continue
            mtl_acc = accuracy_score(np.array(res["ground_truth"]), np.array(res["predictions"]))
            baseline_res = baseline_dict.get(subj, None)
            if baseline_res is None:
                continue
            if isinstance(baseline_res, list):
                baseline_res = baseline_res[-1]
            if not (isinstance(baseline_res, dict) and "ground_truth" in baseline_res and "predictions" in baseline_res):
                continue
            baseline_acc = accuracy_score(np.array(baseline_res["ground_truth"]), np.array(baseline_res["predictions"]))
            cluster = (self.mtl_wrapper.cluster_assignments.get(subj, "Unknown")
                       if hasattr(self.mtl_wrapper, "cluster_assignments") else "Unknown")
            diff = mtl_acc - baseline_acc
            records.append({
                "subject": subj,
                "cluster": cluster,
                "baseline_accuracy": baseline_acc,
                "mtl_accuracy": mtl_acc,
                "difference": diff
            })
        return pd.DataFrame(records)

    def plot_confusion_matrix(self, cm, title, output_file):
        plt.figure(figsize=(8,6))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Saved {title} to {output_file}")

    def plot_cluster_comparison(self, mtl_cluster, baseline_cluster, output_file="cluster_comparison.png"):
        clusters = sorted(list(mtl_cluster.keys()))
        mtl_acc = [mtl_cluster[cl]["accuracy"] for cl in clusters]
        baseline_acc = [baseline_cluster.get(cl, {"accuracy": 0})["accuracy"] for cl in clusters]
        x = np.arange(len(clusters))
        width = 0.35
        plt.figure(figsize=(10,6))
        plt.bar(x - width/2, baseline_acc, width, label="Baseline")
        plt.bar(x + width/2, mtl_acc, width, label="MTL")
        plt.xlabel("Cluster")
        plt.ylabel("Accuracy")
        plt.title("Cluster-Level Performance Comparison")
        plt.xticks(x, clusters)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Saved cluster comparison plot to {output_file}")

    def evaluate(self, verbose=True):
        # Compute overall metrics.
        mtl_overall = self.compute_overall_metrics(self.mtl_wrapper)
        baseline_overall = self.compute_overall_metrics(self.baseline_wrapper)
        if verbose:
            print("=== Overall Metrics ===")
            print("MTL Accuracy: {:.4f}".format(mtl_overall["accuracy"]))
            print("Baseline Accuracy: {:.4f}".format(baseline_overall["accuracy"]))
            print("\nMTL Classification Report:\n", mtl_overall["report"])
            print("\nBaseline Classification Report:\n", baseline_overall["report"])
            self.plot_confusion_matrix(mtl_overall["confusion_matrix"], "MTL Overall Confusion Matrix", "mtl_overall_cm.png")
            self.plot_confusion_matrix(baseline_overall["confusion_matrix"], "Baseline Overall Confusion Matrix", "baseline_overall_cm.png")
        
        # Compute cluster-level metrics.
        mtl_cluster = self.compute_cluster_metrics(self.mtl_wrapper)
        baseline_cluster = self.compute_cluster_metrics(self.baseline_wrapper)
        if verbose:
            print("\n=== Cluster-Level Metrics ===")
            for cl in mtl_cluster.keys():
                diff = mtl_cluster[cl]["accuracy"] - baseline_cluster.get(cl, {"accuracy": 0})["accuracy"]
                print(f"Cluster {cl}: Baseline = {baseline_cluster.get(cl, {'accuracy': 0})['accuracy']:.4f}, MTL = {mtl_cluster[cl]['accuracy']:.4f}, Diff = {diff:.4f}")
                self.plot_confusion_matrix(mtl_cluster[cl]["confusion_matrix"], f"MTL Confusion Matrix - Cluster {cl}", f"mtl_cluster_{cl}_cm.png")
                self.plot_confusion_matrix(baseline_cluster.get(cl, {"confusion_matrix": np.zeros((1,1), dtype=int)})["confusion_matrix"],
                                             f"Baseline Confusion Matrix - Cluster {cl}", f"baseline_cluster_{cl}_cm.png")
            self.plot_cluster_comparison(mtl_cluster, baseline_cluster)
        
        # Compute subject-level metrics.
        try:
            subject_df = self.compute_subject_metrics()
            if verbose:
                print("\n=== Subject-Level Metrics ===")
                print(subject_df)
        except Exception as e:
            print("Error computing subject-level metrics:", e)
            subject_df = None
        
        summary = {
            "overall": {"mtl": mtl_overall, "baseline": baseline_overall},
            "clusters": {"mtl": mtl_cluster, "baseline": baseline_cluster},
            "subjects": subject_df
        }
        return summary

if __name__ == "__main__":
    # Load evaluation configuration from config/mtl.yaml under the key 'evaluators'
    config = OmegaConf.load("config/experiment/mtl.yaml")
    eval_config = config.experiment.evaluators

    # Load the MTL and baseline results. They should be saved as MTLWrapper objects.
    mtl_results = MTLWrapper.load("mtl_training_results.pkl")
    baseline_results = MTLWrapper.load("baseline_training_results.pkl")
    
    evaluator = MTLEvaluator(mtl_results, baseline_results, eval_config)
    summary = evaluator.evaluate(verbose=True)
    
    with open("mtl_evaluation_summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    print("Evaluation summary saved to mtl_evaluation_summary.pkl")
