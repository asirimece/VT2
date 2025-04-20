import os
import numpy as np
import pandas as pd
from lib.evaluate.visuals import VisualEvaluator
from lib.evaluate.metrics import MetricsEvaluator
from lib.logging import logger

logger = logger.get()

class BaselineEvaluator:
    def __init__(self, config: dict):
        # pull out sub‐configs
        quant = config.get("quantitative", {})
        self.metrics     = MetricsEvaluator(quant)

        qual = config.get("qualitative", {})
        self.visualizer  = VisualEvaluator(qual)
        self.output_dir  = qual.get("output_dir", "./evaluation_plots")
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_all(self, wrapper):
        rows = []

        # ----- single-subject -----
        for subj, runs in wrapper.results_by_experiment["single"].items():
            per_run_metrics = []
            for run_res in runs:
                gt  = np.array(run_res["ground_truth"])
                pr  = np.array(run_res["predictions"])
                prb = run_res.get("probabilities", None)
                per_run_metrics.append(self.metrics.evaluate(gt, pr, prb))

            # report each run
            for i, m in enumerate(per_run_metrics, start=1):
                rows.append({
                    "mode":   "single",
                    "subject": subj,
                    "run":    i,
                    **m
                })

            # compute and report mean over runs
            agg = {}
            for k, v0 in per_run_metrics[0].items():
                vals = [m[k] for m in per_run_metrics]
                if isinstance(v0, (int, float, np.floating)):
                    agg[k] = float(np.mean(vals))
                else:
                    agg[k] = v0
            rows.append({
                "mode":   "single",
                "subject": subj,
                "run":    "mean",
                **agg
            })

            # plot using last run
            last = runs[-1]
            gt   = np.array(last["ground_truth"])
            pr   = np.array(last["predictions"])
            self.visualizer.plot_confusion_matrix(
                gt,
                pr,
                filename=f"cm_subj_{subj}.png"
            )

        # ----- pooled -----
        pooled_runs = wrapper.results_by_experiment["pooled"]
        per_run_metrics = []
        for run_res in pooled_runs:
            gt  = np.array(run_res["ground_truth"])
            pr  = np.array(run_res["predictions"])
            prb = run_res.get("probabilities", None)
            per_run_metrics.append(self.metrics.evaluate(gt, pr, prb))

        # report each pooled run
        for i, m in enumerate(per_run_metrics, start=1):
            rows.append({
                "mode":   "pooled",
                "subject":"all",
                "run":    i,
                **m
            })

        # pooled mean
        agg = {}
        for k, v0 in per_run_metrics[0].items():
            vals = [m[k] for m in per_run_metrics]
            if isinstance(v0, (int, float, np.floating)):
                agg[k] = float(np.mean(vals))
            else:
                agg[k] = v0
        rows.append({
            "mode":   "pooled",
            "subject":"all",
            "run":    "mean",
            **agg
        })

        # plot using last pooled run
        last = pooled_runs[-1]
        gt   = np.array(last["ground_truth"])
        pr   = np.array(last["predictions"])
        self.visualizer.plot_confusion_matrix(
            gt,
            pr,
            filename="cm_pooled.png"
        )

        # finally: save a CSV & log it
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "baseline_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Saved baseline metrics → %s", csv_path)
        logger.info("\n%s", df.to_markdown(index=False))




"""from lib.base.metrics import MetricsEvaluator
from lib.base.visuals import VisualEvaluator

class Evaluator:
    def __init__(self, config):
        # Retrieve only the necessary parts from the evaluators config.
        quant_config = {}
        if "quantitative" in config:
            quant = config["quantitative"]
            # Convert the metrics list explicitly to a native Python list.
            quant_config["metrics"] = list(quant.get("metrics", []))
            quant_config["n_runs_aggregation"] = quant.get("n_runs_aggregation", False)
        else:
            quant_config["metrics"] = []
            quant_config["n_runs_aggregation"] = False
        self.quant_config = quant_config

        qual_config = {}
        if "qualitative" in config:
            qual = config["qualitative"]
            qual_config["visualizations"] = list(qual.get("visualizations", []))
            qual_config["pca_n_components"] = qual.get("pca_n_components", 3)
            # Convert tsne config into a plain dictionary.
            qual_config["tsne"] = dict(qual.get("tsne", {}))
            qual_config["output_dir"] = qual.get("output_dir", "./evaluation_plots")
        else:
            qual_config["visualizations"] = []
            qual_config["pca_n_components"] = 3
            qual_config["tsne"] = {}
            qual_config["output_dir"] = "./evaluation_plots"
        self.qual_config = qual_config

        # Initialize the evaluators using the extracted configurations.
        self.metrics_evaluator = MetricsEvaluator(self.quant_config)
        self.visual_evaluator = VisualEvaluator(self.qual_config)

    def evaluate(self, ground_truth, predictions, features=None, probabilities=None):
        # Compute quantitative metrics.
        metrics_results = self.metrics_evaluator.evaluate(ground_truth, predictions, probabilities)
        # If features are supplied, generate the corresponding visualizations.
        if features is not None:
            self.visual_evaluator.visualize(features, ground_truth)
        return metrics_results

    def evaluate_all(self, results):
        overall_eval = {}
        for key, res in results.items():
            if key == "single":
                overall_eval[key] = {}
                for subj, subject_res in res.items():
                    gt = subject_res.get("ground_truth")
                    preds = subject_res.get("predictions")
                    overall_eval[key][subj] = self.evaluate(gt, preds)
            else:
                gt = res.get("ground_truth")
                preds = res.get("predictions")
                overall_eval[key] = self.evaluate(gt, preds)
        return overall_eval
"""