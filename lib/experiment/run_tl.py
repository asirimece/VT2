#!/usr/bin/env python
import os
import argparse
import torch
from omegaconf import OmegaConf
from lib.tl.train import TLTrainer
from lib.logging import logger

logger = logger.get()


def parse_args():
    p = argparse.ArgumentParser("Pretrain TL")
    p.add_argument("-c","--config", 
                   type=str,
                   default="config/experiment/transfer.yaml",
                   help="Path to experiment config (with transfer section)")
    
    p.add_argument("-o","--out-dir", 
                   type=str,
                   required=True)
    
    p.add_argument("--preprocessed-data", 
                   type=str, 
                   required=True)
                   
    p.add_argument("--pretrained-mtl-model", 
                   type=str, 
                   default=None)
                   
    p.add_argument("--init-from-scratch", 
                   action="store_true")
                   
    p.add_argument("--freeze-backbone", 
                   action="store_true")
                   
    p.add_argument("--lr", 
                   type=float, 
                   default=None)

    p.add_argument("--batch-size", 
                   type=int, 
                   default=None)
    
    p.add_argument("--epochs", 
                   type=int, 
                   default=None)

    return p.parse_args()


def main():
    args = parse_args()

    raw_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.create({
        "experiment": {"experiment": raw_cfg.experiment}
    })

    cfg.experiment.experiment.preprocessed_file = args.preprocessed_data
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

    os.makedirs(args.out_dir, exist_ok=True)
    trainer = TLTrainer(cfg)
    trainer.run()
    logger.info(f"TL weights saved to {args.out_dir}")

if __name__ == "__main__":
    main()
