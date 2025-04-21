import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

from lib.evaluate.metrics import MetricsEvaluator
from lib.evaluate.visuals import VisualEvaluator
from lib.logging import logger

logger = logger.get()


class TLEvaluator:
    def __init__(self, tl_results: dict, config: DictConfig):
        self.tl_results = tl_results

        # Normalize config
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        while "evaluators" not in config and "experiment" in config:
            config = config.experiment
        self.cfg = config

        # Metrics & visuals
        qc = self.cfg.evaluators.quantitative
        self.metrics = MetricsEvaluator({"metrics": qc.metrics})

        qv = self.cfg.evaluators.qualitative
        out_dir = self.cfg.evaluators.tl_output_dir
        self.visuals = VisualEvaluator({
            "visualizations": qv.visualizations,
            "pca_n_components": qv.pca_n_components,
            "tsne": qv.tsne,
            'output_dir':       out_dir,
        })
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # Load baseline results
        try:
            from lib.base.train import BaseWrapper
            base_cfg = OmegaConf.load("config/experiment/base.yaml")
            single_wrapped = pickle.load(open(base_cfg.logging.single_results_path, "rb"))
            pooled_wrapped = pickle.load(open(base_cfg.logging.pooled_results_path, "rb"))
            self.baseline_wrapper = BaseWrapper({
                "single": single_wrapped,
                "pooled": pooled_wrapped
            })
            self.baseline_results = self.baseline_wrapper.get_experiment_results("single")
            logger.info("Loaded baseline results from base.yaml")
        except Exception as e:
            self.baseline_results = None
            self.baseline_wrapper = None
            logger.warning(f"Baseline loading skipped: {e}")

        # Load MTL wrapper (mtl_wrapper.pkl)
        try:
            mtl_path = os.path.join(self.cfg.model_output_dir, "mtl_wrapper.pkl")
            with open(mtl_path, "rb") as f:
                mtl_wrapper = pickle.load(f)
            self.mtl_results = mtl_wrapper.results_by_subject
            logger.info(f"Loaded MTL wrapper from {mtl_path}")
        except Exception as e:
            self.mtl_results = None
            logger.warning(f"MTL loading skipped: {e}")

    def evaluate(self):
        logger.info("Evaluating TL performance...")
        rows = []

        for subj, runs in self.tl_results.items():
            for run_idx, wrapper in enumerate(runs):
                gt, pr = wrapper.ground_truth, wrapper.predictions
                row = {"subject": subj, "run": run_idx}
                row.update(self.metrics.evaluate(gt, pr))
                rows.append(row)

                if "confusion_matrix" in self.visuals.visualizations:
                    self.visuals.plot_confusion_matrix(
                        gt, pr, filename=f"cm_tl_subject_{subj}_run{run_idx}.png"
                    )

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.out_dir, "tl_subject_run_metrics.csv"), index=False)

        # subject-level mean ± std
        subj_stats = df.groupby("subject").agg(["mean", "std"])
        subj_stats.columns = [f"{m}_{s}" for m, s in subj_stats.columns]
        subj_stats = subj_stats.reset_index()
        subj_stats.to_csv(os.path.join(self.out_dir, "tl_subject_stats.csv"), index=False)

        # pooled metrics
        pooled = df.drop(columns=["subject", "run"]).agg(["mean", "std"]).T
        pooled.columns = ["mean", "std"]
        pooled = pooled.reset_index().rename(columns={"index": "metric"})
        pooled.to_csv(os.path.join(self.out_dir, "tl_pooled_stats.csv"), index=False)

        outputs = {
            "tl_subject_run": df,
            "tl_subject_stats": subj_stats,
            "tl_pooled_stats": pooled
        }

        # —— TL vs Baseline ——
        if self.baseline_results:
            logger.info("Comparing TL to Baseline...")
            cmp_df = self._compare_wrappers(self.tl_results, self.baseline_results, "tl", "baseline")
            cmp_df.to_csv(os.path.join(self.out_dir, "tl_vs_baseline.csv"), index=False)
            outputs["tl_vs_baseline"] = cmp_df

        # —— MTL vs TL vs Baseline ——
        if self.baseline_results and self.mtl_results:
            logger.info("Comparing TL vs MTL vs Baseline...")

            def flatten(source_dict, label):
                data = []
                for subj, runs in source_dict.items():
                    for run_idx, wrapper in enumerate(runs):
                        if hasattr(wrapper, "ground_truth"):
                            gt, pr = wrapper.ground_truth, wrapper.predictions
                        else:
                            gt, pr = wrapper["ground_truth"], wrapper["predictions"]

                        row = {"model": label, "subject": subj, "run": run_idx}
                        row.update(self.metrics.evaluate(gt, pr))
                        data.append(row)
                return data

            all_rows = (
                flatten(self.baseline_results, "baseline") +
                flatten(self.tl_results, "tl") +
                flatten(self.mtl_results, "mtl")
            )
            df_all = pd.DataFrame(all_rows)
            df_all.to_csv(os.path.join(self.out_dir, "all_model_comparison.csv"), index=False)
            outputs["all_model_comparison"] = df_all

            # Accuracy plot
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 4))
                sns.boxplot(data=df_all, x="model", y="accuracy", palette="pastel")
                plt.title("Accuracy Comparison: Baseline vs TL vs MTL")
                plt.tight_layout()
                plot_path = os.path.join(self.out_dir, "model_accuracy_boxplot.png")
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Saved model comparison plot → {plot_path}")
            except Exception as e:
                logger.warning(f"Could not generate comparison plot: {e}")

        logger.info("TL evaluation complete.")
        return outputs

    def _compare_wrappers(self, a_dict, b_dict, label_a, label_b):
        """
        Compare per-subject results from two wrappers (e.g. TL vs Baseline)
        Returns: DataFrame with per-run metrics and deltas
        """
        rows = []
        for subj in a_dict.keys():
            runs_a = a_dict.get(subj, [])
            runs_b = b_dict.get(subj, [])

            for i in range(min(len(runs_a), len(runs_b))):
                # Handle TLWrapper object
                if hasattr(runs_a[i], "ground_truth"):
                    gt_a = runs_a[i].ground_truth
                    pr_a = runs_a[i].predictions
                else:
                    gt_a = runs_a[i]["ground_truth"]
                    pr_a = runs_a[i]["predictions"]

                if hasattr(runs_b[i], "ground_truth"):
                    gt_b = runs_b[i].ground_truth
                    pr_b = runs_b[i].predictions
                else:
                    gt_b = runs_b[i]["ground_truth"]
                    pr_b = runs_b[i]["predictions"]

                a_metrics = self.metrics.evaluate(gt_a, pr_a)
                b_metrics = self.metrics.evaluate(gt_b, pr_b)

                row = {"subject": subj, "run": i}
                for k in a_metrics:
                    row[f"{label_a}_{k}"] = a_metrics[k]
                    row[f"{label_b}_{k}"] = b_metrics[k]
                    row[f"delta_{k}"] = a_metrics[k] - b_metrics[k]
                rows.append(row)

        return pd.DataFrame(rows)




"""# tl_evaluator.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import itertools

class TLEvaluator:
    def evaluate(self, tl_wrapper, class_names=None, plot_confusion=False, save_plot_path=None):
        gt = tl_wrapper.ground_truth
        pred = tl_wrapper.predictions
        
        accuracy = accuracy_score(gt, pred)
        kappa = cohen_kappa_score(gt, pred)
        cm = confusion_matrix(gt, pred)
        report = classification_report(gt, pred, target_names=class_names) if class_names is not None else classification_report(gt, pred)
        precision, recall, f1, support = precision_recall_fscore_support(gt, pred, average=None)
        
        metrics = {
            "accuracy": accuracy,
            "kappa": kappa,
            "confusion_matrix": cm,
            "classification_report": report,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist()
        }
        
        if plot_confusion:
            self.plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')
            if save_plot_path:
                plt.savefig(save_plot_path)
                print(f"[INFO] Confusion matrix plot saved to {save_plot_path}")
            plt.show()

        return metrics

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
"""