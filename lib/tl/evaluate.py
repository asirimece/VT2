import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from lib.base.train import BaseWrapper
from lib.evaluate.metrics import MetricsEvaluator
from lib.evaluate.visuals import VisualEvaluator
from lib.logging import logger

logger = logger.get()


class TLEvaluator:
    def __init__(self, tl_results: dict, config: DictConfig):
        self.tl_results = tl_results

        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        while "evaluators" not in config and "experiment" in config:
            config = config.experiment
        self.cfg = config

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

        try:
            base_cfg = OmegaConf.load("config/experiment/base.yaml")
            single_wrapped = pickle.load(open(base_cfg.logging.single_results_path, "rb"))
            pooled_wrapped = pickle.load(open(base_cfg.logging.pooled_results_path, "rb"))
            self.baseline_wrapper = BaseWrapper({
                "single": single_wrapped,
                "pooled": pooled_wrapped
            })
            self.baseline_results = self.baseline_wrapper.get_experiment_results("single")
        except Exception as e:
            self.baseline_results = None
            self.baseline_wrapper = None
            logger.warning(f"Skipped loading baseline results: {e}")

        # Load MTL wrapper.
        try:
            mtl_path = os.path.join(self.cfg.mtl.mtl_model_output, "mtl_wrapper.pkl")
            with open(mtl_path, "rb") as f:
                mtl_wrapper = pickle.load(f)
            self.mtl_results = mtl_wrapper.results_by_subject
        except Exception as e:
            self.mtl_results = None
            logger.warning(f"MTL loading skipped: {e}")

    def evaluate(self):
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

        # subject-level
        subj_stats = df.groupby("subject").agg(["mean", "std"])
        subj_stats.columns = [f"{m}_{s}" for m, s in subj_stats.columns]
        subj_stats = subj_stats.reset_index()
        subj_stats.to_csv(os.path.join(self.out_dir, "tl_subject_stats.csv"), index=False)

        # pooled
        pooled = df.drop(columns=["subject", "run"]).agg(["mean", "std"]).T
        pooled.columns = ["mean", "std"]
        pooled = pooled.reset_index().rename(columns={"index": "metric"})
        pooled.to_csv(os.path.join(self.out_dir, "tl_pooled_stats.csv"), index=False)

        outputs = {
            "tl_subject_run": df,
            "tl_subject_stats": subj_stats,
            "tl_pooled_stats": pooled
        }

        # TL vs Baseline
        if self.baseline_results:
            logger.info("Comparing TL to Baseline.")
            cmp_df = self._compare_wrappers(self.tl_results, self.baseline_results, "tl", "baseline")
            cmp_df.to_csv(os.path.join(self.out_dir, "tl_vs_baseline.csv"), index=False)
            outputs["tl_vs_baseline"] = cmp_df

        # MTL vs TL vs Baseline
        if self.baseline_results and self.mtl_results:
            logger.info("Comparing TL vs MTL vs Baseline.")

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
            except Exception as e:
                logger.warning(f"Could not generate comparison plot: {e}")

        logger.info("TL evaluation complete.")
        return outputs

    def _compare_wrappers(self, a_dict, b_dict, label_a, label_b):
        rows = []
        for subj in a_dict.keys():
            runs_a = a_dict.get(subj, [])
            runs_b = b_dict.get(subj, [])

            for i in range(min(len(runs_a), len(runs_b))):
                # Handle TLWrapper 
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
