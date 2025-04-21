# lib/evaluate/visuals.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, \
    ConfusionMatrixDisplay, RocCurveDisplay
import seaborn as sns

class VisualEvaluator:
    """
    Produce PCA and/or t-SNE scatter plots of feature embeddings,
    confusion matrices, and ROC curves.

    Config dict should contain:
      - "visualizations": list from ["pca","tsne","confusion_matrix","roc_curve","cluster_scatter"]
      - "pca_n_components": int
      - "tsne": { "perplexity":…, "n_iter":… }
      - "output_dir": path
    """

    def __init__(self, config: dict):
        self.visualizations     = config.get("visualizations", [])
        self.pca_n_components   = config.get("pca_n_components", 3)
        self.tsne_cfg           = config.get("tsne", {"perplexity":30, "n_iter":1000})
        self.output_dir         = config.get("output_dir", "./evaluation_plots")
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_pca(self, features: np.ndarray, labels: np.ndarray):
        if "pca" not in self.visualizations:
            return
        pca = PCA(n_components=self.pca_n_components)
        Xp  = pca.fit_transform(features)
        plt.figure()
        for cls in np.unique(labels):
            idx = labels == cls
            plt.scatter(Xp[idx,0], Xp[idx,1], label=str(cls), s=20, alpha=0.8)
        plt.legend()
        plt.title("PCA Projection")
        out = os.path.join(self.output_dir, "pca_plot.png")
        plt.savefig(out)
        plt.close()
        print(f"[DEBUG] Saved PCA plot → {out}")

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray):
        if "tsne" not in self.visualizations:
            return
        tsne = TSNE(
            n_components=2,
            perplexity=self.tsne_cfg.get("perplexity",30),
            n_iter=self.tsne_cfg.get("n_iter",1000)
        )
        Xt = tsne.fit_transform(features)
        plt.figure()
        for cls in np.unique(labels):
            idx = labels == cls
            plt.scatter(Xt[idx,0], Xt[idx,1], label=str(cls), s=20, alpha=0.8)
        plt.legend()
        plt.title("t-SNE Projection")
        out = os.path.join(self.output_dir, "tsne_plot.png")
        plt.savefig(out)
        plt.close()
        print(f"[DEBUG] Saved t-SNE plot → {out}")

    def plot_confusion_matrix(self,
                              ground_truth: np.ndarray,
                              predictions:  np.ndarray,
                              labels: list = None,
                              filename: str = "confusion_matrix.png"):
        if "confusion_matrix" not in self.visualizations:
            return
        cm   = confusion_matrix(ground_truth, predictions, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6,6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix")
        out = os.path.join(self.output_dir, filename)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[DEBUG] Saved confusion matrix → {out}")

    def plot_roc_curve(self,
                       ground_truth:  np.ndarray,
                       probabilities: np.ndarray,
                       filename_prefix: str = "roc_curve"):
        if "roc_curve" not in self.visualizations:
            return
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

    def plot_cluster_scatter(self,
                             subject_reprs: dict,
                             cluster_assignments: dict,
                             method: str = "pca",
                             filename: str = "cluster_scatter.png"):
        """
        Projects subject-level feature vectors into 2D and colors points by cluster.
        - method: 'pca' or 'tsne'
        - subject_reprs: {subject_id: vector}
        - cluster_assignments: {subject_id: cluster_id}
        """
        if "cluster_scatter" not in self.visualizations:
            return

        # Prepare data
        ids    = list(subject_reprs.keys())
        X      = np.stack([subject_reprs[sid] for sid in ids], axis=0)
        labels = np.array([cluster_assignments.get(sid, -1) for sid in ids])

        if method.lower() == "pca":
            transformer = PCA(n_components=2)
            title       = "PCA projection of subjects"
        elif method.lower() == "tsne":
            cfg         = self.tsne_cfg
            n_samples   = X.shape[0]
            perplexity  = cfg.get("perplexity", 30)
            if n_samples <= perplexity:
                print(f"[DEBUG] Skipping t-SNE scatter: only {n_samples} samples ≤ perplexity={perplexity}")
                return
            transformer = TSNE(
                n_components=2,
                perplexity=cfg.get("perplexity",30),
                n_iter=cfg.get("n_iter",1000)
            )
            title       = "t-SNE projection of subjects"
        else:
            raise ValueError(f"Unknown method: {method}")

        proj = transformer.fit_transform(X)
        plt.figure(figsize=(6,6))
        scatter = plt.scatter(proj[:,0], proj[:,1], c=labels, cmap="tab10",
                              s=50, alpha=0.8)
        plt.title(f"{title} colored by cluster")
        plt.xlabel(f"{method.upper()} 1")
        plt.ylabel(f"{method.upper()} 2")
        # Legend
        handles, _    = scatter.legend_elements()
        unique_lbls   = sorted(set(labels.tolist()))
        plt.legend(handles, unique_lbls, title="Cluster")
        out = os.path.join(self.output_dir, filename.replace(".png", f"_{method}.png"))
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"[DEBUG] Saved cluster scatter ({method}) → {out}")
    
    def plot_delta_by_cluster(
        self,
        df_cmp_subj: pd.DataFrame,
        cluster_assignments: dict | None = None
    ):
        """
        Boxplot of accuracy_delta grouped by cluster.
        If df_cmp_subj lacks a 'cluster' column, we map via cluster_assignments.
        """
        if "delta_by_cluster" not in self.visualizations:
            return

        df = df_cmp_subj.copy()
        if "cluster" not in df.columns:
            if cluster_assignments is None:
                raise ValueError(
                    "plot_delta_by_cluster needs either a 'cluster' column "
                    "in your DataFrame or a cluster_assignments dict."
                )
            df["cluster"] = df["subject"].map(cluster_assignments)

        plt.figure(figsize=(6,4))
        sns.boxplot(
            x="cluster",
            y="accuracy_delta",
            data=df,
            palette="pastel"
        )
        plt.title("Accuracy Δ by Cluster (MTL − Baseline)")
        plt.xlabel("Cluster")
        plt.ylabel("Δ Accuracy")
        out = os.path.join(self.output_dir, "delta_by_cluster_boxplot.png")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"[DEBUG] Saved Δ by cluster → {out}")

    def plot_subject_delta_sorted(self, df_cmp_subj: "pd.DataFrame"):
        """Bar chart of |accuracy_delta| sorted by absolute Δ per subject."""
        if "subject_delta_sorted" not in self.visualizations:
            return
        # compute mean Δ per subject (absolute)
        df_mean = (
            df_cmp_subj
            .groupby("subject")["accuracy_delta"]
            .mean()
            .abs()
            .sort_values(ascending=False)
            .reset_index()
        )
        plt.figure(figsize=(8,3))
        plt.bar(
            df_mean["subject"].astype(str),
            df_mean["accuracy_delta"],
            color="skyblue"
        )
        plt.xticks(rotation=90)
        plt.title("Subject |Δ Accuracy| (MTL vs Baseline), sorted")
        plt.xlabel("Subject")
        plt.ylabel("|Δ Accuracy|")
        out = os.path.join(self.output_dir, "subject_delta_sorted.png")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"[DEBUG] Saved subject‐sorted Δ → {out}")

    def plot_delta_violin(self, df_cmp_subj: "pd.DataFrame"):
        """Violin plot of the full distribution of Δ‑accuracy."""
        if "delta_violin" not in self.visualizations:
            return
        plt.figure(figsize=(6,4))
        sns.violinplot(
            y=df_cmp_subj["accuracy_delta"],
            inner="quart",
            color="lightcoral"
        )
        plt.title("Distribution of Δ Accuracy (MTL − Baseline)")
        plt.ylabel("Δ Accuracy")
        out = os.path.join(self.output_dir, "delta_violin.png")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"[DEBUG] Saved Δ violin → {out}")

    def visualize(self,
                  features:      np.ndarray = None,
                  labels:        np.ndarray = None,
                  predictions:   np.ndarray = None,
                  probabilities: np.ndarray = None,
                  subject_reprs: dict       = None,
                  cluster_assignments: dict = None):
        """
        Dispatch to whichever plots are configured:
          - features+labels → PCA/t-SNE
          - labels+predictions → confusion
          - labels+probabilities → ROC
          - subject_reprs+cluster_assignments → cluster scatter
        """
        if features is not None and labels is not None:
            self.plot_pca(features, labels)
            self.plot_tsne(features, labels)

        if predictions is not None:
            self.plot_confusion_matrix(labels, predictions)

        if probabilities is not None:
            self.plot_roc_curve(labels, probabilities)

        if subject_reprs is not None and cluster_assignments is not None:
            # produce both PCA‑ and t‑SNE‑based scatter
            self.plot_cluster_scatter(subject_reprs, cluster_assignments, method="pca")
            self.plot_cluster_scatter(subject_reprs, cluster_assignments, method="tsne")
