import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay

class VisualEvaluator:
    """
    Produce PCA and/or t-SNE scatter plots of feature embeddings,
    confusion matrices, and ROC curves.
    Config dict should contain:
      - "visualizations": list from ["pca","tsne","confusion","roc"]
      - "pca_n_components": int
      - "tsne": { "perplexity":…, "n_iter":… }
      - "output_dir": path
    """
    def __init__(self, config: dict):
        self.visualizations = config.get("visualizations", [])
        self.pca_n_components = config.get("pca_n_components", 3)
        self.tsne_cfg = config.get("tsne", {"perplexity": 30, "n_iter": 1000})
        self.output_dir = config.get("output_dir", "./evaluation_plots")
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_pca(self, features: np.ndarray, labels: np.ndarray):
        pca = PCA(n_components=self.pca_n_components)
        Xp = pca.fit_transform(features)
        plt.figure()
        for cls in np.unique(labels):
            idx = labels == cls
            plt.scatter(Xp[idx, 0], Xp[idx, 1], label=str(cls), s=20)
        plt.legend()
        plt.title("PCA Projection")
        out = os.path.join(self.output_dir, "pca_plot.png")
        plt.savefig(out)
        plt.close()
        print(f"[DEBUG] Saved PCA plot → {out}")

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray):
        tsne = TSNE(
            n_components=2,
            perplexity=self.tsne_cfg.get("perplexity", 30),
            n_iter=self.tsne_cfg.get("n_iter", 1000)
        )
        Xt = tsne.fit_transform(features)
        plt.figure()
        for cls in np.unique(labels):
            idx = labels == cls
            plt.scatter(Xt[idx, 0], Xt[idx, 1], label=str(cls), s=20)
        plt.legend()
        plt.title("t-SNE Projection")
        out = os.path.join(self.output_dir, "tsne_plot.png")
        plt.savefig(out)
        plt.close()
        print(f"[DEBUG] Saved t-SNE plot → {out}")

    def plot_confusion_matrix(self, ground_truth: np.ndarray, predictions: np.ndarray,
                              labels: list = None, filename: str = "confusion_matrix.png"):
        """Plot and save a confusion matrix."""
        cm = confusion_matrix(ground_truth, predictions, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix")
        out = os.path.join(self.output_dir, filename)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[DEBUG] Saved confusion matrix → {out}")

    def plot_roc_curve(self, ground_truth: np.ndarray, probabilities: np.ndarray,
                        filename_prefix: str = "roc"):
        """Plot and save one ROC curve per class."""
        classes = np.unique(ground_truth)
        for cls in classes:
            binary_gt = (ground_truth == cls).astype(int)
            fpr, tpr, _ = roc_curve(binary_gt, probabilities[:, cls])
            roc_auc = auc(fpr, tpr)
            disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            ax.set_title(f"ROC – class {cls}")
            out = os.path.join(self.output_dir, f"{filename_prefix}_class_{cls}.png")
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"[DEBUG] Saved ROC curve → {out}")

    def visualize(
        self,
        features: np.ndarray = None,
        labels: np.ndarray = None,
        predictions: np.ndarray = None,
        probabilities: np.ndarray = None
    ):
        """
        Generate configured visualizations. supply:
          - features, labels for PCA/t-SNE
          - labels, predictions for confusion
          - labels, probabilities for ROC
        """
        if features is not None and labels is not None:
            if "pca" in self.visualizations:
                self.plot_pca(features, labels)
            if "tsne" in self.visualizations:
                self.plot_tsne(features, labels)
        if predictions is not None:
            if "confusion_matrix" in self.visualizations:
                self.plot_confusion_matrix(labels, predictions)
        if probabilities is not None:
            if "roc_curve" in self.visualizations:
                self.plot_roc_curve(labels, probabilities)
