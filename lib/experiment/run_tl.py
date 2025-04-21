#!/usr/bin/env python
import os
import argparse
import torch
from omegaconf import OmegaConf
from lib.tl.train import TLTrainer

def parse_args():
    p = argparse.ArgumentParser("Pretrain TL (scratch, global, or per‐cluster)")
    p.add_argument("-c","--config", type=str,
                   default="config/experiment/mtl.yaml",
                   help="Path to experiment config (with transfer section)")
    p.add_argument("-o","--out-dir", type=str, required=True,
                   help="Where to save TL outputs (models & wrappers)")
    p.add_argument("--preprocessed-data", type=str, required=True,
                   help="Path to preprocessed_data.pkl containing all subjects")
    p.add_argument("--pretrained-mtl-model", type=str, default=None,
                   help="Path to pretrained MTL weights (omit for scratch)")
    p.add_argument("--init-from-scratch", action="store_true",
                   help="Ignore pretrained backbone and train from random init.")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze shared backbone during TL training")
    p.add_argument("--lr", type=float, default=None,
                   help="Override TL learning rate")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override TL batch size")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override TL number of epochs")
    return p.parse_args()


def main():
    args = parse_args()
    print("Flags:", args)

    # 1) Load and wrap config so TLTrainer sees experiment.experiment.transfer
    raw_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.create({
        "experiment": {"experiment": raw_cfg.experiment}
    })

    # 2) Override data paths and TL settings
    # point at single‐subject data file
    cfg.experiment.experiment.preprocessed_file = args.preprocessed_data
    # configure transfer section
    tcfg = cfg.experiment.experiment.transfer
    tcfg.model_output_dir    = args.out_dir
    if args.pretrained_mtl_model:
        tcfg.pretrained_mtl_model = args.pretrained_mtl_model
    tcfg.init_from_scratch   = args.init_from_scratch
    tcfg.freeze_backbone     = args.freeze_backbone
    if args.lr is not None:
        tcfg.lr = args.lr
    if args.batch_size is not None:
        tcfg.batch_size = args.batch_size
    if args.epochs is not None:
        tcfg.epochs = args.epochs

    # 3) Run TL training
    os.makedirs(args.out_dir, exist_ok=True)
    trainer = TLTrainer(cfg)
    trainer.run()
    print(f"[INFO] TL complete. Models & results in {args.out_dir}")

if __name__ == "__main__":
    main()
