# lib/pipeline/mtl/mtlevaluator.py

import os
import pickle

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from lib.base.train       import BaseWrapper
from lib.evaluate.metrics import MetricsEvaluator
from lib.evaluate.visuals import VisualEvaluator
from lib.logging          import logger
import matplotlib.pyplot as plt
import seaborn as sns

logger = logger.get()

class MTLEvaluator:
    """
    Reports per‑run and mean±std metrics, and draws:
      - subject‑level (with cluster IDs)
      - cluster‑level
      - pooled
      - baseline comparisons at all levels
      - 2D scatter of subjects colored by cluster
      - Δ‑accuracy summaries by cluster and subject
      - aggregated-over‑runs confusion matrices
    """

    def __init__(self, mtl_wrapper, experiment_cfg):
        # Normalize to a DictConfig containing `evaluators`
        if not isinstance(experiment_cfg, DictConfig):
            experiment_cfg = OmegaConf.create(experiment_cfg)
        while isinstance(experiment_cfg, DictConfig) \
          and 'evaluators' not in experiment_cfg \
          and 'experiment' in experiment_cfg:
            experiment_cfg = experiment_cfg.experiment
        if not isinstance(experiment_cfg, DictConfig) or 'evaluators' not in experiment_cfg:
            raise ValueError(
                "MTLEvaluator expected config containing 'evaluators'; got:\n"
                f"{OmegaConf.to_yaml(experiment_cfg)}"
            )

        self.experiment_cfg = experiment_cfg
        self.mtl_wrapper    = mtl_wrapper

        # quantitative metrics
        qc = self.experiment_cfg.evaluators.quantitative
        self.metrics         = MetricsEvaluator({'metrics': qc.metrics})
        self.cluster_metrics = MetricsEvaluator({'metrics': qc.cluster_metrics})

        # qualitative visuals
        qv      = self.experiment_cfg.evaluators.qualitative
        out_dir = self.experiment_cfg.evaluators.mtl_output_dir
        self.visuals = VisualEvaluator({
            'visualizations':   qv.visualizations,
            'pca_n_components': qv.pca_n_components,
            'tsne':             qv.tsne,
            'output_dir':       out_dir,
        })
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # load baselines
        base_cfg       = OmegaConf.load('config/experiment/base.yaml')
        single_wrapped = pickle.load(open(base_cfg.logging.single_results_path, 'rb'))
        pooled_wrapped = pickle.load(open(base_cfg.logging.pooled_results_path, 'rb'))
        self.baseline_wrapper = BaseWrapper({
            'single': single_wrapped,
            'pooled': pooled_wrapped
        })

        logger.info("MTLEvaluator initialized. Writing outputs to %s", self.out_dir)
        
        # load subject representations for cluster_scatter
        feat_file = experiment_cfg.features_file
        with open(feat_file, "rb") as f:
            all_feats = pickle.load(f)      # same format: {subj: {session: {combined: …}}}
        # compute mean per subject
        self.subject_reprs = {
            subj: np.mean(
                np.concatenate([sess["combined"] for sess in sessions.values()], axis=0),
                axis=0
            )
            for subj, sessions in all_feats.items()
        }


    def evaluate(self):
        """
        subject_reprs: Optional dict {subject_id: feature_vector}
                        needed for cluster‑scatter plots.
        """
        viz_list = self.experiment_cfg.evaluators.qualitative.visualizations

        # —— 1) SUBJECT‑LEVEL ——
        subj_run = []
        for subj, runs in self.mtl_wrapper.results_by_subject.items():
            for ridx, run in enumerate(runs):
                gt = np.array(run['ground_truth'])
                pr = np.array(run['predictions'])
                m  = self.metrics.evaluate(gt, pr)
                row = {'run': ridx, 'subject': subj}
                row.update(m)
                subj_run.append(row)
                if 'confusion_matrix' in viz_list:
                    self.visuals.plot_confusion_matrix(
                        gt, pr,
                        filename=f"cm_mtl_subj_{subj}_run{ridx}.png"
                    )
        df_subj_run = pd.DataFrame(subj_run)
        df_subj_run.to_csv(os.path.join(self.out_dir, "mtl_subject_run_metrics.csv"), index=False)

        subj_stats = df_subj_run.groupby('subject').agg(['mean','std'])
        subj_stats.columns = [f"{met}_{st}" for met, st in subj_stats.columns]
        subj_stats = subj_stats.reset_index()
        subj_stats['cluster'] = subj_stats['subject'].map(self.mtl_wrapper.cluster_assignments)
        cols = ['subject','cluster'] + [c for c in subj_stats.columns if c not in ('subject','cluster')]
        subj_stats = subj_stats[cols]
        subj_stats.to_csv(os.path.join(self.out_dir, "mtl_subject_stats_metrics.csv"), index=False)

        # —— 2) CLUSTER‑LEVEL ——
        cl_run = []
        max_runs = max(len(runs) for runs in self.mtl_wrapper.results_by_subject.values())
        for ridx in range(max_runs):
            for subj, runs in self.mtl_wrapper.results_by_subject.items():
                if ridx < len(runs):
                    run = runs[ridx]
                    gt, pr = map(np.array, (run['ground_truth'], run['predictions']))
                    cl = self.mtl_wrapper.cluster_assignments.get(subj, "None")
                    m  = self.cluster_metrics.evaluate(gt, pr)
                    row = {'run': ridx, 'cluster': cl}
                    row.update(m)
                    cl_run.append(row)
                    if 'confusion_matrix' in viz_list:
                        self.visuals.plot_confusion_matrix(
                            gt, pr,
                            filename=f"cm_mtl_cluster_{cl}_run{ridx}.png"
                        )
        df_cl_run = pd.DataFrame(cl_run)
        df_cl_run.to_csv(os.path.join(self.out_dir, "mtl_cluster_run_metrics.csv"), index=False)

        cl_stats = df_cl_run.groupby('cluster').agg(['mean','std'])
        cl_stats.columns = [f"{met}_{st}" for met, st in cl_stats.columns]
        cl_stats = cl_stats.reset_index()
        cl_stats.to_csv(os.path.join(self.out_dir, "mtl_cluster_stats_metrics.csv"), index=False)

        # —— BASELINE CLUSTER‑LEVEL ——
        base_cl_run = []
        base_single = self.baseline_wrapper.get_experiment_results('single')
        max_base = max(len(r) for r in base_single.values())
        for run_idx in range(max_base):
            for subj, runs in base_single.items():
                if run_idx < len(runs):
                    run = runs[run_idx]
                    gt, pr = map(np.array, (run['ground_truth'], run['predictions']))
                    cl = self.mtl_wrapper.cluster_assignments.get(subj, 'None')
                    m  = self.cluster_metrics.evaluate(gt, pr)
                    row = {'run': run_idx, 'cluster': cl}
                    row.update(m)
                    base_cl_run.append(row)
                    if 'confusion_matrix' in viz_list:
                        self.visuals.plot_confusion_matrix(
                            gt, pr,
                            filename=f"cm_base_cluster_{cl}_run{run_idx}.png"
                        )
        df_base_cl_run = pd.DataFrame(base_cl_run)
        df_base_cl_run.to_csv(os.path.join(self.out_dir, 'baseline_cluster_run_metrics.csv'), index=False)

        base_cl_stats = df_base_cl_run.groupby('cluster').agg(['mean','std'])
        base_cl_stats.columns = [f"{met}_{st}" for met, st in base_cl_stats.columns]
        base_cl_stats = base_cl_stats.reset_index()
        base_cl_stats.to_csv(os.path.join(self.out_dir, 'baseline_cluster_stats_metrics.csv'), index=False)

        # —— 3) POOLED ——
        pool_run = []
        for run_idx in range(max_runs):
            all_gt, all_pr = [], []
            for runs in self.mtl_wrapper.results_by_subject.values():
                if run_idx < len(runs):
                    all_gt.extend(runs[run_idx]['ground_truth'])
                    all_pr.extend(runs[run_idx]['predictions'])
            gt, pr = map(np.array, (all_gt, all_pr))
            m  = self.metrics.evaluate(gt, pr)
            row = {'run': run_idx}
            row.update(m)
            pool_run.append(row)
            if 'confusion_matrix' in viz_list:
                self.visuals.plot_confusion_matrix(
                    gt, pr,
                    filename=f"cm_mtl_pooled_run{run_idx}.png"
                )
        df_pool_run = pd.DataFrame(pool_run)
        df_pool_run.to_csv(os.path.join(self.out_dir, 'mtl_pooled_run_metrics.csv'), index=False)

        pool_stats = df_pool_run.drop(columns=['run']).agg(['mean','std']).T
        pool_stats.columns = ['mean','std']
        pool_stats = pool_stats.reset_index().rename(columns={'index':'metric'})
        pool_stats.to_csv(os.path.join(self.out_dir, 'mtl_pooled_stats_metrics.csv'), index=False)

        # —— 4) BASELINE SINGLE‑SUBJECT ——
        base_subj_run = []
        for subj, runs in self.baseline_wrapper.get_experiment_results('single').items():
            for run_idx, run in enumerate(runs):
                gt, pr = map(np.array, (run['ground_truth'], run['predictions']))
                m  = self.metrics.evaluate(gt, pr)
                row = {'run': run_idx, 'subject': subj}
                row.update(m)
                base_subj_run.append(row)
        df_base_subj_run = pd.DataFrame(base_subj_run)
        df_base_subj_run.to_csv(os.path.join(self.out_dir, 'baseline_subject_run_metrics.csv'), index=False)

        # —— 5) BASELINE POOLED ——
        base_pool_run = []
        for run_idx, run in enumerate(self.baseline_wrapper.get_experiment_results('pooled')):
            gt, pr = map(np.array, (run['ground_truth'], run['predictions']))
            m  = self.metrics.evaluate(gt, pr)
            row = {'run': run_idx}
            row.update(m)
            base_pool_run.append(row)
        df_base_pool_run = pd.DataFrame(base_pool_run)
        df_base_pool_run.to_csv(os.path.join(self.out_dir, 'baseline_pooled_run_metrics.csv'), index=False)

        # —— 6) MERGE & DELTAS ——
        df_cmp_subj = df_subj_run.merge(df_base_subj_run,
                                        on=['subject','run'],
                                        suffixes=('_mtl','_base'))
        for metric in self.experiment_cfg.evaluators.quantitative.metrics:
            df_cmp_subj[f'{metric}_delta'] = (
                df_cmp_subj[f'{metric}_mtl'] - df_cmp_subj[f'{metric}_base']
            )
        df_cmp_subj.to_csv(os.path.join(self.out_dir, 'mtl_vs_baseline_subject_run.csv'), index=False)
        logger.info('Saved subject/run deltas to %s',
                    os.path.join(self.out_dir, 'mtl_vs_baseline_subject_run.csv'))

        df_cmp_cl = df_cl_run.merge(df_base_cl_run,
                                    on=['cluster','run'],
                                    suffixes=('_mtl','_base'))
        for metric in self.experiment_cfg.evaluators.quantitative.cluster_metrics:
            df_cmp_cl[f'{metric}_delta'] = (
                df_cmp_cl[f'{metric}_mtl'] - df_cmp_cl[f'{metric}_base']
            )
        df_cmp_cl.to_csv(os.path.join(self.out_dir, 'mtl_vs_baseline_cluster_run.csv'), index=False)
        logger.info('Saved cluster/run deltas to %s',
                    os.path.join(self.out_dir, 'mtl_vs_baseline_cluster_run.csv'))

        df_cmp_pool = df_pool_run.merge(df_base_pool_run,
                                        on='run',
                                        suffixes=('_mtl','_base'))
        for metric in self.experiment_cfg.evaluators.quantitative.metrics:
            df_cmp_pool[f'{metric}_delta'] = (
                df_cmp_pool[f'{metric}_mtl'] - df_cmp_pool[f'{metric}_base']
            )
        df_cmp_pool.to_csv(os.path.join(self.out_dir, 'mtl_vs_baseline_pooled_run.csv'), index=False)
        logger.info('Saved pooled/run deltas to %s',
                    os.path.join(self.out_dir, 'mtl_vs_baseline_pooled_run.csv'))

        # —— 7) PLOTTED COMPARISONS ——
        # 7.1 Pooled accuracy
        plt.figure(figsize=(6,4))
        plt.plot(df_cmp_pool['run'], df_cmp_pool['accuracy_mtl'],
                 marker='o', label='MTL', color='blue')
        plt.plot(df_cmp_pool['run'], df_cmp_pool['accuracy_base'],
                 marker='s', label='Baseline', color='green')
        plt.xlabel('Run'); plt.ylabel('Accuracy')
        plt.title('Pooled Accuracy: MTL vs Baseline'); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'pooled_accuracy_comparison.png'))
        plt.close()
        logger.info('Saved pooled accuracy comparison → %s',
                    os.path.join(self.out_dir, 'pooled_accuracy_comparison.png'))

        # 7.2 Cluster accuracy comparison (MTL left, Baseline right)
        cl_mt = df_cl_run.groupby('cluster')['accuracy'].agg(['mean','std']).rename(
            columns={'mean':'mtl_mean','std':'mtl_std'})
        cl_bs = df_base_cl_run.groupby('cluster')['accuracy'].agg(['mean','std']).rename(
            columns={'mean':'base_mean','std':'base_std'})
        cmpdf = cl_mt.join(cl_bs, how='outer').fillna(0)
        clusters = list(cmpdf.index)
        x = np.arange(len(clusters))
        w = 0.35
        plt.figure(figsize=(8,4))
        # MTL bars on left
        plt.bar(x - w/2,
                cmpdf['mtl_mean'],
                w,
                yerr=cmpdf['mtl_std'],
                label='MTL',
                color='blue',
                capsize=4)
        # Baseline bars on right
        plt.bar(x + w/2,
                cmpdf['base_mean'],
                w,
                yerr=cmpdf['base_std'],
                label='Baseline',
                color='green',
                capsize=4)
        plt.xticks(x, clusters)
        plt.xlabel('Cluster'); plt.ylabel('Accuracy')
        plt.title('Cluster Accuracy: MTL vs Baseline'); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'cluster_accuracy_comparison.png'))
        plt.close()
        logger.info('Saved cluster accuracy comparison → %s',
                    os.path.join(self.out_dir, 'cluster_accuracy_comparison.png'))

        # 7.3 Subject-level Δ boxplot
        plt.figure(figsize=(6,4))
        sns.boxplot(y=df_cmp_subj['accuracy_delta'], color='lightgray')
        plt.title('Subject-level Δ Accuracy (MTL − Baseline)')
        plt.ylabel('Δ Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'subject_accuracy_delta_boxplot.png'))
        plt.close()
        logger.info('Saved subject Δ boxplot → %s',
                    os.path.join(self.out_dir, 'subject_accuracy_delta_boxplot.png'))

        # —— 8) AGGREGATED confusion matrices ——
        if 'confusion_matrix' in viz_list:
            # pooled MTL
            all_gt, all_pr = [], []
            for runs in self.mtl_wrapper.results_by_subject.values():
                for run in runs:
                    all_gt.extend(run['ground_truth'])
                    all_pr.extend(run['predictions'])
            self.visuals.plot_confusion_matrix(
                np.array(all_gt), np.array(all_pr),
                filename="cm_mtl_pooled_agg.png"
            )
            # pooled Baseline
            all_gt_bs, all_pr_bs = [], []
            for run in self.baseline_wrapper.get_experiment_results('pooled'):
                all_gt_bs.extend(run['ground_truth'])
                all_pr_bs.extend(run['predictions'])
            self.visuals.plot_confusion_matrix(
                np.array(all_gt_bs), np.array(all_pr_bs),
                filename="cm_baseline_pooled_agg.png"
            )
            # per‑cluster aggregated
            for cl in cl_stats['cluster'].tolist():
                # MTL cluster
                gt_cl, pr_cl = [], []
                for subj, runs in self.mtl_wrapper.results_by_subject.items():
                    if self.mtl_wrapper.cluster_assignments.get(subj) == cl:
                        for run in runs:
                            gt_cl.extend(run['ground_truth'])
                            pr_cl.extend(run['predictions'])
                self.visuals.plot_confusion_matrix(
                    np.array(gt_cl), np.array(pr_cl),
                    filename=f"cm_mtl_cluster_{cl}_agg.png"
                )
                # Baseline cluster
                gt_bs_cl, pr_bs_cl = [], []
                for subj, runs in self.baseline_wrapper.get_experiment_results('single').items():
                    if self.mtl_wrapper.cluster_assignments.get(subj) == cl:
                        for run in runs:
                            gt_bs_cl.extend(run['ground_truth'])
                            pr_bs_cl.extend(run['predictions'])
                self.visuals.plot_confusion_matrix(
                    np.array(gt_bs_cl), np.array(pr_bs_cl),
                    filename=f"cm_baseline_cluster_{cl}_agg.png"
                )

        # —— 9) CLUSTER‑SCATTER ——
        if 'cluster_scatter' in viz_list:
            self.visuals.plot_cluster_scatter(
                self.subject_reprs,
                self.mtl_wrapper.cluster_assignments,
                method="pca",
                filename="cluster_scatter_pca.png"
            )
            self.visuals.plot_cluster_scatter(
                self.subject_reprs,
                self.mtl_wrapper.cluster_assignments,
                method="tsne",
                filename="cluster_scatter_tsne.png"
            )

        return {
            'mtl_subject_run':       df_subj_run,
            'mtl_subject_stats':     subj_stats,
            'mtl_cluster_run':       df_cl_run,
            'mtl_cluster_stats':     cl_stats,
            'baseline_cluster_run':  df_base_cl_run,
            'baseline_cluster_stats': base_cl_stats,
            'mtl_pooled_run':        df_pool_run,
            'mtl_pooled_stats':      pool_stats,
            'baseline_subject_run':  df_base_subj_run,
            'baseline_pooled_run':   df_base_pool_run,
            'delta_subject_run':     df_cmp_subj,
            'delta_cluster_run':     df_cmp_cl,
            'delta_pooled_run':      df_cmp_pool,
        }




"""# mtlevaluator.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lib.logging import logger
from lib.mtl.train import MTLWrapper
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logger = logger.get()

class MTLEvaluator:
    def __init__(self, mtl_wrapper, baseline_wrapper, config):
        self.mtl_wrapper = mtl_wrapper
        # If baseline_wrapper is a plain dict, wrap it.
        if isinstance(baseline_wrapper, dict):
            self.baseline_wrapper = type(mtl_wrapper)(results_by_subject=baseline_wrapper)
        else:
            self.baseline_wrapper = baseline_wrapper
        self.config = config

    @staticmethod
    def load_results(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        # If obj is already an instance of MTLWrapper, return it.
        if hasattr(obj, "results_by_subject"):
            return obj
        # If it's a dict with "ground_truth" and "predictions", wrap under key "pooled".
        if isinstance(obj, dict) and set(obj.keys()) == {"ground_truth", "predictions"}:
            wrapped = {"pooled": obj}
            print("[DEBUG] Loaded results as dict with keys ['ground_truth', 'predictions']. Wrapping under key 'pooled'.")
            from mtl_wrapper import MTLWrapper
            return MTLWrapper(results_by_subject=wrapped)
        # If it's a dict mapping subject IDs, wrap it.
        if isinstance(obj, dict):
            from mtl_wrapper import MTLWrapper
            return MTLWrapper(results_by_subject=obj)
        # If it's a list, wrap it as pooled.
        if isinstance(obj, list):
            from mtl_wrapper import MTLWrapper
            return MTLWrapper(results_by_subject={"pooled": obj})
        return obj

    def compute_overall_metrics(self, wrapper):
        all_gt = []
        all_pred = []
        for subj, res in wrapper.results_by_subject.items():
            if isinstance(res, list):
                res = res[-1]
            if not (isinstance(res, dict) and "ground_truth" in res and "predictions" in res):
                logger.warning(f"Results for subject {subj} are not in the expected format; skipping.")
                continue
            all_gt.extend(res["ground_truth"])
            all_pred.extend(res["predictions"])
        all_gt = np.array(all_gt)
        all_pred = np.array(all_pred)
        overall_acc = accuracy_score(all_gt, all_pred)
        overall_cm = confusion_matrix(all_gt, all_pred)
        overall_report = classification_report(all_gt, all_pred, zero_division=0)
        return {"accuracy": overall_acc, "confusion_matrix": overall_cm, "report": overall_report}

    def compute_cluster_metrics(self, wrapper):
        cluster_data = {}
        if not hasattr(wrapper, "cluster_assignments") or not wrapper.cluster_assignments:
            return {}
        for subj, res in wrapper.results_by_subject.items():
            if isinstance(res, list):
                res = res[-1]
            if not (isinstance(res, dict) and "ground_truth" in res and "predictions" in res):
                continue
            cl = wrapper.cluster_assignments.get(subj, "None")
            cluster_data.setdefault(cl, {"ground_truth": [], "predictions": []})
            cluster_data[cl]["ground_truth"].extend(res["ground_truth"])
            cluster_data[cl]["predictions"].extend(res["predictions"])
        cluster_metrics = {}
        for cl, data in cluster_data.items():
            gt = np.array(data["ground_truth"])
            preds = np.array(data["predictions"])
            acc = accuracy_score(gt, preds)
            cm = confusion_matrix(gt, preds)
            report = classification_report(gt, preds, zero_division=0)
            cluster_metrics[cl] = {"accuracy": acc, "confusion_matrix": cm, "report": report}
        return cluster_metrics

    def compute_subject_metrics(self):
        records = []
        baseline_dict = (self.baseline_wrapper.results_by_subject 
                         if hasattr(self.baseline_wrapper, "results_by_subject") 
                         else self.baseline_wrapper)
        for subj, res in self.mtl_wrapper.results_by_subject.items():
            if isinstance(res, list):
                res = res[-1]
            if not (isinstance(res, dict) and "ground_truth" in res and "predictions" in res):
                continue
            mtl_acc = accuracy_score(np.array(res["ground_truth"]), np.array(res["predictions"]))
            baseline_res = baseline_dict.get(subj, None)
            if baseline_res is None:
                continue
            if isinstance(baseline_res, list):
                baseline_res = baseline_res[-1]
            if not (isinstance(baseline_res, dict) and "ground_truth" in baseline_res and "predictions" in baseline_res):
                continue
            baseline_acc = accuracy_score(np.array(baseline_res["ground_truth"]), np.array(baseline_res["predictions"]))
            cluster = (self.mtl_wrapper.cluster_assignments.get(subj, "Unknown")
                       if hasattr(self.mtl_wrapper, "cluster_assignments") else "Unknown")
            diff = mtl_acc - baseline_acc
            records.append({
                "subject": subj,
                "cluster": cluster,
                "baseline_accuracy": baseline_acc,
                "mtl_accuracy": mtl_acc,
                "difference": diff
            })
        return pd.DataFrame(records)

    def plot_confusion_matrix(self, cm, title, output_file):
        plt.figure(figsize=(8,6))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Saved {title} to {output_file}")

    def plot_cluster_comparison(self, mtl_cluster, baseline_cluster, output_file="cluster_comparison.png"):
        clusters = sorted(list(mtl_cluster.keys()))
        mtl_acc = [mtl_cluster[cl]["accuracy"] for cl in clusters]
        baseline_acc = [baseline_cluster.get(cl, {"accuracy": 0})["accuracy"] for cl in clusters]
        x = np.arange(len(clusters))
        width = 0.35
        plt.figure(figsize=(10,6))
        plt.bar(x - width/2, baseline_acc, width, label="Baseline")
        plt.bar(x + width/2, mtl_acc, width, label="MTL")
        plt.xlabel("Cluster")
        plt.ylabel("Accuracy")
        plt.title("Cluster-Level Performance Comparison")
        plt.xticks(x, clusters)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Saved cluster comparison plot to {output_file}")

    def evaluate(self, verbose=True):
        # Compute overall metrics.
        mtl_overall = self.compute_overall_metrics(self.mtl_wrapper)
        baseline_overall = self.compute_overall_metrics(self.baseline_wrapper)
        if verbose:
            print("=== Overall Metrics ===")
            print("MTL Accuracy: {:.4f}".format(mtl_overall["accuracy"]))
            print("Baseline Accuracy: {:.4f}".format(baseline_overall["accuracy"]))
            print("\nMTL Classification Report:\n", mtl_overall["report"])
            print("\nBaseline Classification Report:\n", baseline_overall["report"])
            self.plot_confusion_matrix(mtl_overall["confusion_matrix"], "MTL Overall Confusion Matrix", "mtl_overall_cm.png")
            self.plot_confusion_matrix(baseline_overall["confusion_matrix"], "Baseline Overall Confusion Matrix", "baseline_overall_cm.png")
        
        # Compute cluster-level metrics.
        mtl_cluster = self.compute_cluster_metrics(self.mtl_wrapper)
        baseline_cluster = self.compute_cluster_metrics(self.baseline_wrapper)
        if verbose:
            print("\n=== Cluster-Level Metrics ===")
            for cl in mtl_cluster.keys():
                diff = mtl_cluster[cl]["accuracy"] - baseline_cluster.get(cl, {"accuracy": 0})["accuracy"]
                print(f"Cluster {cl}: Baseline = {baseline_cluster.get(cl, {'accuracy': 0})['accuracy']:.4f}, MTL = {mtl_cluster[cl]['accuracy']:.4f}, Diff = {diff:.4f}")
                self.plot_confusion_matrix(mtl_cluster[cl]["confusion_matrix"], f"MTL Confusion Matrix - Cluster {cl}", f"mtl_cluster_{cl}_cm.png")
                self.plot_confusion_matrix(baseline_cluster.get(cl, {"confusion_matrix": np.zeros((1,1), dtype=int)})["confusion_matrix"],
                                             f"Baseline Confusion Matrix - Cluster {cl}", f"baseline_cluster_{cl}_cm.png")
            self.plot_cluster_comparison(mtl_cluster, baseline_cluster)
        
        # Compute subject-level metrics.
        try:
            subject_df = self.compute_subject_metrics()
            if verbose:
                print("\n=== Subject-Level Metrics ===")
                print(subject_df)
        except Exception as e:
            print("Error computing subject-level metrics:", e)
            subject_df = None
        
        summary = {
            "overall": {"mtl": mtl_overall, "baseline": baseline_overall},
            "clusters": {"mtl": mtl_cluster, "baseline": baseline_cluster},
            "subjects": subject_df
        }
        return summary

if __name__ == "__main__":
    # Load evaluation configuration from config/mtl.yaml under the key 'evaluators'
    config = OmegaConf.load("config/experiment/mtl.yaml")
    eval_config = config.experiment.evaluators

    # Load the MTL and baseline results. They should be saved as MTLWrapper objects.
    mtl_results = MTLWrapper.load("mtl_training_results.pkl")
    baseline_results = MTLWrapper.load("baseline_training_results.pkl")
    
    evaluator = MTLEvaluator(mtl_results, baseline_results, eval_config)
    summary = evaluator.evaluate(verbose=True)
    
    with open("mtl_evaluation_summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    print("Evaluation summary saved to mtl_evaluation_summary.pkl")
"""