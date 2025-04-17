from lib.base.metrics import MetricsEvaluator
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
        """
        Handles evaluation for both single-subject and pooled experiments.
        :param results: dict with keys like "single" and "pooled". For "single" this is a dict mapping
                        subject IDs to result dicts (each with at least "ground_truth" and "predictions").
        :return: dict of evaluation metrics.
        """
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
