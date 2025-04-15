# tl_evaluator.py
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
        """
        Evaluate predictions with a comprehensive set of metrics.
        
        Args:
            tl_wrapper: TLResultsWrapper object containing ground_truth and predictions.
            class_names (list, optional): List of class names for a prettier report and confusion matrix.
            plot_confusion (bool, optional): If True, displays a confusion matrix plot.
            save_plot_path (str, optional): If provided, saves the plot to this file path.
        
        Returns:
            metrics (dict): Dictionary containing overall accuracy, kappa, per-class metrics,
                            confusion matrix, and a classification report.
        """
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
        """
        Plots the confusion matrix.
        
        Args:
            cm (np.array): Confusion matrix.
            classes (list): List of class names.
            normalize (bool): If True, normalize the values in the confusion matrix.
            title (str): Title for the plot.
            cmap: Colormap instance.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        #tick_marks = np.arange(len(classes))
        plt.xticks(classes, rotation=45)
        plt.yticks(classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
