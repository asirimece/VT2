import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

class VisualEvaluator:
    def __init__(self, config):
        """
        config: dict containing keys for visualization methods and parameters.
        """
        self.config = config
        self.visualizations = config.get("visualizations", [])
        self.output_dir = config.get("output_dir", "./evaluation_plots")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_pca(self, features, labels):
        pca_n_components = self.config.get("pca_n_components", 3)
        pca = PCA(n_components=pca_n_components)
        transformed = pca.fit_transform(features)
        plt.figure()
        for cls in np.unique(labels):
            idx = labels == cls
            plt.scatter(transformed[idx, 0], transformed[idx, 1], label=str(cls))
        plt.legend()
        plt.title("PCA Visualization")
        out_file = os.path.join(self.output_dir, "pca_plot.png")
        plt.savefig(out_file)
        plt.close()
        print(f"[DEBUG] PCA plot saved to {out_file}")
    
    def plot_tsne(self, features, labels):
        tsne_config = self.config.get("tsne", {})
        perplexity = tsne_config.get("perplexity", 30)
        n_iter = tsne_config.get("n_iter", 1000)
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
        transformed = tsne.fit_transform(features)
        plt.figure()
        for cls in np.unique(labels):
            idx = labels == cls
            plt.scatter(transformed[idx, 0], transformed[idx, 1], label=str(cls))
        plt.legend()
        plt.title("t-SNE Visualization")
        out_file = os.path.join(self.output_dir, "tsne_plot.png")
        plt.savefig(out_file)
        plt.close()
        print(f"[DEBUG] t-SNE plot saved to {out_file}")
    
    def visualize(self, features, labels):
        if "pca" in self.visualizations:
            self.plot_pca(features, labels)
        if "tsne" in self.visualizations:
            self.plot_tsne(features, labels)
