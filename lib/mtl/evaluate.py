import os
import pickle
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from lib.base.train import BaseWrapper
from lib.evaluate.metrics import MetricsEvaluator
from lib.evaluate.visuals import VisualEvaluator
from lib.logging import logger
import matplotlib.pyplot as plt
import seaborn as sns

logger = logger.get()


class MTLEvaluator:
    def __init__(self, mtl_wrapper, experiment_cfg):
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
        self.mtl_wrapper = mtl_wrapper

        # quantitative metrics
        qc = self.experiment_cfg.evaluators.quantitative
        self.metrics = MetricsEvaluator({'metrics': qc.metrics})
        self.cluster_metrics = MetricsEvaluator({'metrics': qc.cluster_metrics})

        # qualitative visuals
        qv = self.experiment_cfg.evaluators.qualitative
        out_dir = self.experiment_cfg.evaluators.mtl_output_dir
        self.visuals = VisualEvaluator({
            'visualizations': qv.visualizations,
            'pca_n_components': qv.pca_n_components,
            'tsne': qv.tsne,
            'output_dir': out_dir,
        })
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # load baseline results
        base_cfg = OmegaConf.load('config/experiment/base.yaml')
        single_wrapped = pickle.load(open(base_cfg.logging.single_results_path, 'rb'))
        pooled_wrapped = pickle.load(open(base_cfg.logging.pooled_results_path, 'rb'))
        self.baseline_wrapper = BaseWrapper({
            'single': single_wrapped,
            'pooled': pooled_wrapped
        })

        logger.info("MTLEvaluator initialized.")
        
        # load subject representations
        feat_file = experiment_cfg.features_file
        with open(feat_file, "rb") as f:
            all_feats = pickle.load(f)

        self.subject_reprs = {
            subj: np.mean(
                np.concatenate([sess["combined"] for sess in sessions.values()], axis=0),
                axis=0
            )
            for subj, sessions in all_feats.items()
        }

    def evaluate(self):
        viz_list = self.experiment_cfg.evaluators.qualitative.visualizations

        # Subject level
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

        # Cluster level
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

        # Pooled 
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
        
        # Baseline cluster level
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

        # Baseline single subject
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

        # Baseline pooled
        base_pool_run = []
        for run_idx, run in enumerate(self.baseline_wrapper.get_experiment_results('pooled')):
            gt, pr = map(np.array, (run['ground_truth'], run['predictions']))
            m  = self.metrics.evaluate(gt, pr)
            row = {'run': run_idx}
            row.update(m)
            base_pool_run.append(row)
        df_base_pool_run = pd.DataFrame(base_pool_run)
        df_base_pool_run.to_csv(os.path.join(self.out_dir, 'baseline_pooled_run_metrics.csv'), index=False)

        # Comparison
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


        cl_mt = df_cl_run.groupby('cluster')['accuracy'].agg(['mean','std']).rename(
            columns={'mean':'mtl_mean','std':'mtl_std'})
        cl_bs = df_base_cl_run.groupby('cluster')['accuracy'].agg(['mean','std']).rename(
            columns={'mean':'base_mean','std':'base_std'})
        cmpdf = cl_mt.join(cl_bs, how='outer').fillna(0)
        clusters = list(cmpdf.index)
        x = np.arange(len(clusters))
        w = 0.35
        plt.figure(figsize=(8,4))

        plt.bar(x - w/2,
                cmpdf['mtl_mean'],
                w,
                yerr=cmpdf['mtl_std'],
                label='MTL',
                color='blue',
                capsize=4)

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

        plt.figure(figsize=(6,4))
        sns.boxplot(y=df_cmp_subj['accuracy_delta'], color='lightgray')
        plt.title('Subject-level Δ Accuracy (MTL − Baseline)')
        plt.ylabel('Δ Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'subject_accuracy_delta_boxplot.png'))
        plt.close()

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
            all_gt_bs, all_pr_bs = [], []
            for run in self.baseline_wrapper.get_experiment_results('pooled'):
                all_gt_bs.extend(run['ground_truth'])
                all_pr_bs.extend(run['predictions'])
            self.visuals.plot_confusion_matrix(
                np.array(all_gt_bs), np.array(all_pr_bs),
                filename="cm_baseline_pooled_agg.png"
            )
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

        # Cluster scatter
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
            'mtl_subject_run': df_subj_run,
            'mtl_subject_stats': subj_stats,
            'mtl_cluster_run': df_cl_run,
            'mtl_cluster_stats': cl_stats,
            'baseline_cluster_run': df_base_cl_run,
            'baseline_cluster_stats': base_cl_stats,
            'mtl_pooled_run': df_pool_run,
            'mtl_pooled_stats': pool_stats,
            'baseline_subject_run': df_base_subj_run,
            'baseline_pooled_run': df_base_pool_run,
            'delta_subject_run': df_cmp_subj,
            'delta_cluster_run': df_cmp_cl,
            'delta_pooled_run': df_cmp_pool,
        }
