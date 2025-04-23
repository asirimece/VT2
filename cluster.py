#!/usr/bin/env python
import os
import pickle
import yaml

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from lib.pipeline.cluster.cluster import SubjectClusterer

def vectorize(feat):
    """
    Recursively flatten whatever is in feats_dict[sid]:
      - If it's an ndarray, just flatten.
      - If it's a dict, recurse into each value (sorted by key) and concat.
      - Otherwise, coerce to array and flatten.
    """
    if isinstance(feat, np.ndarray):
        return feat.ravel()
    if isinstance(feat, dict):
        parts = []
        for k in sorted(feat.keys()):
            parts.append(vectorize(feat[k]))
        return np.concatenate(parts)
    return np.array(feat).ravel()

def main():
    # 1) Load raw features
    feats_path = "dump/features.pkl"
    feats_dict = pickle.load(open(feats_path, "rb"))

    # 2) Cluster config + clustering
    clust_cfg = yaml.safe_load(open("config/experiment/transfer.yaml"))["experiment"]["clustering"]
    clusterer = SubjectClusterer(feats_path, clust_cfg)
    cw = clusterer.cluster_subjects(method=clust_cfg["method"])

    # 3) Prepare data
    subject_ids = sorted(cw.labels.keys(), key=int)
    vecs, labels = [], []
    for sid in subject_ids:
        raw = feats_dict.get(sid, feats_dict.get(str(sid)))
        if raw is None:
            raise KeyError(f"Subject {sid!r} not in features.pkl")
        vecs.append(vectorize(raw))
        labels.append(cw.labels[sid])
    X = np.stack(vecs)
    y = np.array(labels, dtype=int)

    X2 = PCA(n_components=2).fit_transform(X)

    os.makedirs("evaluation", exist_ok=True)
    plt.figure(figsize=(8,6))
    palette = ["tab:blue","tab:orange","tab:green","tab:red"]
    for cluster_id in range(4):
        idx = np.where(y == cluster_id)[0]
        plt.scatter(
            X2[idx,0], X2[idx,1],
            c=palette[cluster_id],
            label=f"Cluster {cluster_id}",
            s=200,
            alpha=0.7,
            edgecolor="k"
        )
    # annotate each point by subject id
    for i,sid in enumerate(subject_ids):
        plt.text(
            X2[i,0], X2[i,1], sid,
            ha="center", va="center",
            fontsize=9, color="white", weight="bold"
        )

    plt.legend(title="Clusters", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.title("Subject Clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    out = "evaluation/cluster_scatter.png"
    plt.savefig(out, dpi=150)
    print(f"Saved refined cluster scatter to {out}")

if __name__=="__main__":
    main()
