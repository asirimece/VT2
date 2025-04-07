#!/usr/bin/env python
import hydra
from omegaconf import DictConfig, OmegaConf
from lib.steps import run_pipeline, save_features

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Run the full pipeline. The preprocessed_data.pkl already contains sub-epoching.
    features = run_pipeline(cfg)
    
    # Save the extracted features for later stages (e.g., clustering or classification)
    out_file = cfg.get("features_output", "./features.pkl")
    save_features(features, cfg, filename=out_file)
    
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
