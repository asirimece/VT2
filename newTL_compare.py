#!/usr/bin/env python3
"""
analyze_clustering_benefit.py

Load per‐subject stats for pooled vs clustered TL, compute Δaccuracy,
and plot clustering benefit vs baseline performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # ─── CONFIG ────────────────────────────────────────────────────────────
    # Point these to your actual CSV outputs from TLEvaluator:
    pooled_csv    = Path("results/tl_n1/tl_subject_stats.csv")
    clustered_csv = Path("results/tl_n4/tl_subject_stats.csv")
    # Output figure path:
    out_fig = Path("results/clustering_benefit.png")
    # ───────────────────────────────────────────────────────────────────────

    # 1) Load stats
    df_pooled    = pd.read_csv(pooled_csv,    usecols=["subject", "accuracy_mean"])
    df_clustered = pd.read_csv(clustered_csv, usecols=["subject", "accuracy_mean"])

    # 2) Join and compute Δ
    df = (
        df_pooled.rename(columns={"accuracy_mean": "acc_pooled"})
        .merge(
            df_clustered.rename(columns={"accuracy_mean": "acc_clustered"}),
            on="subject"
        )
    )
    df["delta_acc"] = df["acc_clustered"] - df["acc_pooled"]

    # 3) Scatter & line‐of‐best‐fit
    plt.figure(figsize=(6,5))
    plt.scatter(df["acc_pooled"], df["delta_acc"], s=50, alpha=0.7)
    # fit a simple line
    m, b = pd.np.polyfit(df["acc_pooled"], df["delta_acc"], 1)
    xs = [df["acc_pooled"].min(), df["acc_pooled"].max()]
    plt.plot(xs, [m*x + b for x in xs], "r--", lw=2)

    plt.xlabel("Baseline accuracy (pooled)")
    plt.ylabel("Δ accuracy (clustered – pooled)")
    plt.title("Clustering benefit vs. baseline performance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.show()

    # 4) Compute Pearson r
    r = df["acc_pooled"].corr(df["delta_acc"])
    print(f"Pearson r between baseline and benefit: {r:.3f}")

if __name__ == "__main__":
    main()
