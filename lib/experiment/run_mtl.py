#!/usr/bin/env python
import os
import argparse
import pickle
import torch
from omegaconf import OmegaConf

from lib.pipeline.cluster.cluster import SubjectClusterer
from lib.mtl.train import MTLTrainer
from lib.utils.utils import convert_state_dict_keys

def parse_args():
    p = argparse.ArgumentParser("Pretrain MTL (global or per‑cluster)")
    p.add_argument("-c","--config",      type=str,
                   default="config/experiment/mtl.yaml",
                   help="Path to experiment mtl.yaml")
    p.add_argument("-m","--model-config",type=str,
                   default="config/model/deep4net.yaml",
                   help="Path to model deep4net.yaml")
    p.add_argument("-o","--out-dir",     type=str, default=None,
                   help="Where to save wrapper & weights")
    p.add_argument("-r","--restrict-to-cluster", action="store_true",
                   help="Only train on subjects in --cluster-id")
    p.add_argument("-k","--cluster-id",  type=int, default=None,
                   help="Cluster label to restrict (required if -r)")
    p.add_argument("-f","--features-file", type=str, default=None,
                   help="Override features.pkl used for clustering")
    return p.parse_args()

def main():
    args = parse_args()
    print("Flags:", args)

    # 1) load + patch experiment config
    exp_cfg = OmegaConf.load(args.config)
    if args.out_dir:
        exp_cfg.experiment.model_output_dir = args.out_dir
    if args.features_file:
        exp_cfg.experiment.features_file = args.features_file

    # inject cluster‐restriction into config
    OmegaConf.set_struct(exp_cfg, False)
    exp_cfg.experiment.restrict_to_cluster = args.restrict_to_cluster
    exp_cfg.experiment.cluster_id        = args.cluster_id
    OmegaConf.set_struct(exp_cfg, True)

    # 2) diagnostics: cluster vs preprocessed
    clusterer = SubjectClusterer(
        exp_cfg.experiment.features_file,
        exp_cfg.experiment.clustering
    )
    cw = clusterer.cluster_subjects(
        method=exp_cfg.experiment.clustering.method
    )
    subs_feat = [s for s,l in cw.labels.items()
                 if (not args.restrict_to_cluster) or l==args.cluster_id]
    print(f"[DIAG] Subjects in selected cluster: {subs_feat}")

    pp = pickle.load(open(exp_cfg.experiment.preprocessed_file,"rb"))
    subs_pp = list(pp.keys())
    inter = set(subs_feat) & set(subs_pp)
    print(f"[DIAG] Subjects in preprocessed_data: {subs_pp}")
    print(f"[DIAG] Intersection: {sorted(inter)}")
    if args.restrict_to_cluster and not inter:
        raise RuntimeError(f"No overlap for cluster {args.cluster_id}!")

    # 3) train!
    trainer = MTLTrainer(
        experiment_cfg=exp_cfg,
        model_cfg=args.model_config
    )
    wrapper = trainer.run()  # this does all the epochs / hyperparam runs

    # 4) save out the backbone weights
    suffix = "all" if not args.restrict_to_cluster else f"cluster{args.cluster_id}"
    out_dir = exp_cfg.experiment.model_output_dir
    os.makedirs(out_dir, exist_ok=True)
    wpath = os.path.join(out_dir, f"mtl_weights_{suffix}.pth")
    sd = convert_state_dict_keys(trainer.model.state_dict())
    torch.save(sd, wpath)
    print(f"[INFO] Pretrained MTL weights saved to {wpath}")

    # 5) wrapper
    print(f"[INFO] MTL wrapper saved to {trainer.wrapper_path}")

if __name__=="__main__":
    main()
