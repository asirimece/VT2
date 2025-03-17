#!/usr/bin/env python
import os
import pickle
from lib.pipeline import run_preprocessing_pipeline, save_preprocessed_data
from lib.feature_extraction import run_feature_extraction

def run_preprocessing(cfg):
    print("Running preprocessing pipeline...")
    preprocessed_data = run_preprocessing_pipeline(cfg)
    out_file = cfg.get("preprocessed_output", "./outputs/preprocessed_data.pkl")
    save_preprocessed_data(preprocessed_data, out_file)
    return preprocessed_data

def run_feature_extraction_stage(cfg, preprocessed_data):
    print("Running feature extraction stage...")
    features = {}
    for subj, sessions in preprocessed_data.items():
        features[subj] = {}
        for sess_label, epochs in sessions.items():
            feat_dict = run_feature_extraction(epochs, cfg.feature_extraction)
            # Add labels from the epochs (from the 3rd column of events)
            labels = epochs.events[:, -1]
            feat_dict['labels'] = labels
            features[subj][sess_label] = feat_dict
            print(f"Extracted features for subject {subj}, session {sess_label}")
    return features

def run_pipeline(cfg):
    preprocessed_data = run_preprocessing(cfg)
    features = run_feature_extraction_stage(cfg, preprocessed_data)
    return features

def save_features(features, cfg, filename="./features.pkl"):
    methods_cfg = cfg.feature_extraction.get('methods', [])
    method_names = [str(method_cfg.get('name', 'unknown')) for method_cfg in methods_cfg]
    methods_str = "_".join(method_names)
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_{methods_str}{ext}"
    with open(new_filename, "wb") as f:
        pickle.dump(features, f)
    print(f"Features saved to {new_filename}")
