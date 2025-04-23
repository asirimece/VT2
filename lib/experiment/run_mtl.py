import os
import argparse
import pickle
import torch
from omegaconf import OmegaConf
from lib.pipeline.cluster.cluster import SubjectClusterer
from lib.mtl.train import MTLTrainer
from lib.utils.utils import convert_state_dict_keys
from lib.logging import logger

logger = logger.get()


def parse_args():
    p = argparse.ArgumentParser("Pretrain MTL")
    p.add_argument("-c","--config",      
                   type=str,
                   default="config/experiment/transfer.yaml")
    
    p.add_argument("-m","--model-config",
                   type=str,
                   default="config/model/deep4net.yaml")
    
    p.add_argument("-o","--out-dir",     
                   type=str, 
                   default=None)

    p.add_argument("-r","--restrict-to-cluster", 
                   action="store_true")

    p.add_argument("-k","--cluster-id",  
                   type=int, 
                   default=None)

    p.add_argument("-f","--features-file", 
                   type=str, 
                   default=None)
                   
    return p.parse_args()


def main():
    args = parse_args()

    exp_cfg = OmegaConf.load(args.config)
    if args.out_dir:
        exp_cfg.experiment.model_output_dir = args.out_dir
    if args.features_file:
        exp_cfg.experiment.features_file = args.features_file

    # inject cluster‐restriction
    OmegaConf.set_struct(exp_cfg, False)
    exp_cfg.experiment.restrict_to_cluster = args.restrict_to_cluster
    exp_cfg.experiment.cluster_id        = args.cluster_id
    OmegaConf.set_struct(exp_cfg, True)

    clusterer = SubjectClusterer(
        exp_cfg.experiment.features_file,
        exp_cfg.experiment.clustering
    )
    cw = clusterer.cluster_subjects(
        method=exp_cfg.experiment.clustering.method
    )
    subs_feat = [s for s,l in cw.labels.items()
                 if (not args.restrict_to_cluster) or l==args.cluster_id]
    logger.info(f"Subjects in selected cluster: {subs_feat}")

    pp = pickle.load(open(exp_cfg.experiment.preprocessed_file,"rb"))
    subs_pp = list(pp.keys())
    inter = set(subs_feat) & set(subs_pp)
    if args.restrict_to_cluster and not inter:
        raise RuntimeError(f"No overlap for cluster {args.cluster_id}!")

    trainer = MTLTrainer(
        experiment_cfg=exp_cfg,
        model_cfg=args.model_config
    )
    wrapper = trainer.run()

    suffix = "all" if not args.restrict_to_cluster else f"cluster{args.cluster_id}"
    out_dir = exp_cfg.experiment.model_output_dir
    os.makedirs(out_dir, exist_ok=True)
    wpath = os.path.join(out_dir, f"mtl_weights_{suffix}.pth")
    sd = convert_state_dict_keys(trainer.model.state_dict())
    torch.save(sd, wpath)
    logger.info(f"MTL weights saved to {wpath}")

if __name__=="__main__":
    main()
