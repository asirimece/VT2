from lib.evaluate.metrics import MetricsEvaluator
from lib.evaluate.visuals import VisualEvaluator

class Evaluator:
    def __init__(self, config):
        """
        config: dict containing two keys: 'quantitative' and 'qualitative'
        """
        self.quant_config = config.get("quantitative", {})
        self.qual_config = config.get("qualitative", {})
        self.metrics_evaluator = MetricsEvaluator(self.quant_config)
        self.visual_evaluator = VisualEvaluator(self.qual_config)
    
    def evaluate(self, ground_truth, predictions, features=None, probabilities=None):
        """
        Evaluates both quantitative and qualitative metrics.
        :param ground_truth: true labels
        :param predictions: predicted labels
        :param features: features used for visualization (optional)
        :param probabilities: predicted probabilities (optional, for ROC)
        :return: dict containing quantitative metric results
        """
        metrics_results = self.metrics_evaluator.evaluate(ground_truth, predictions, probabilities)
        if features is not None:
            self.visual_evaluator.visualize(features, ground_truth)
        return metrics_results
