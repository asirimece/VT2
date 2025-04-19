# summarize_tl.py
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from lib.tl.evaluator import TLEvaluator

PIPELINES = ["scratch", "cluster", "global"]
SUBJECTS  = [1,2,3,4,5,6,7,8,9]  # adjust to your set
OUTDIR = "evaluation"

# 1) load per-subject metrics
records = []
for subj in SUBJECTS:
    for p in PIPELINES:
        res_path = os.path.join("./dump/tl", p, str(subj), f"tl_{subj}_results.pkl")
        wrapper = pickle.load(open(res_path, "rb"))
        # evaluate with TLEvaluator
        metrics = TLEvaluator().evaluate(wrapper, plot_confusion=False)
        records.append({
            "subject": subj,
            "pipeline": p,
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
            "cm": metrics["confusion_matrix"],
            "gt": wrapper.ground_truth,
            "pred": wrapper.predictions
        })
df = pd.DataFrame(records)
os.makedirs(OUTDIR, exist_ok=True)
df.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)

# 2) boxplot of accuracy
plt.figure()
df.boxplot(column="accuracy", by="pipeline")
plt.title("TL Accuracy by Pipeline")
plt.suptitle("")
plt.savefig(os.path.join(OUTDIR,"accuracy_boxplot.png"))

# 3) scatter global vs cluster
pivot = df.pivot(index="subject", columns="pipeline", values="accuracy")
x = pivot["global"]; y = pivot["cluster"]
plt.figure()
plt.scatter(x,y)
m = max(x.max(), y.max())
plt.plot([0,m],[0,m], "--", color="gray")
plt.xlabel("Global TL Accuracy")
plt.ylabel("Cluster TL Accuracy")
plt.title("Global vs Cluster TL")
plt.savefig(os.path.join(OUTDIR,"global_vs_cluster_scatter.png"))

# 4) ROC macro‐averaged
plt.figure()
for p in PIPELINES:
    # concatenate all subjects
    gt = np.concatenate(df[df.pipeline==p]["gt"].values)
    pred = np.concatenate(df[df.pipeline==p]["pred"].values)
    # one‐hot
    from sklearn.preprocessing import label_binarize
    classes = np.unique(gt)
    Y = label_binarize(gt, classes=classes)
    P = label_binarize(pred, classes=classes)
    fpr, tpr, _ = roc_curve(Y.ravel(), P.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{p} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("Macro‐ROC")
plt.savefig(os.path.join(OUTDIR,"roc_macro.png"))

# 5) average confusion matrices
for p in PIPELINES:
    cms = np.stack(df[df.pipeline==p]["cm"].values)
    avg_cm = cms.mean(axis=0)
    plt.figure()
    plt.imshow(avg_cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Avg Confusion Matrix: {p}")
    plt.colorbar()
    plt.savefig(os.path.join(OUTDIR, f"cm_{p}.png"))
