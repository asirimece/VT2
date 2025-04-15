#!/usr/bin/env python
import os
import pickle
import numpy as np
from omegaconf import OmegaConf
from lib.pipeline import run_preprocessing_pipeline, save_preprocessed_data
from lib.feature_extraction import run_feature_extraction
from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.linear_model import Lasso, ElasticNetCV

def run_preprocessing(cfg):
    print("Running preprocessing pipeline...")
    preprocessed_data = run_preprocessing_pipeline(cfg)
    out_file = cfg.get("preprocessed_output", "./outputs/preprocessed_data.pkl")
    save_preprocessed_data(preprocessed_data, out_file)
    return preprocessed_data

def run_feature_extraction_stage(feat_cfg, preprocessed_data):
    print("Running feature extraction stage...")
    features = {}
    # For each subject and session, compute the individual features,
    # then normalize, concatenate them, apply PCA, and then feature selection.
    for subj, sessions in preprocessed_data.items():
        features[subj] = {}
    
        for sess_label, epochs in sessions.items():
            # Get the raw feature matrices from each method
            feat_dict = run_feature_extraction(epochs,feat_cfg)
            
            # Check that at least one method produced features
            if not feat_dict:
                print(f"No features extracted for subject {subj}, session {sess_label}")
                continue

            # For each method, normalize features (z-score)
            scaler = StandardScaler()
            normalized_feats = {}
            for method, feats in feat_dict.items():
                normalized_feats[method] = scaler.fit_transform(feats)
                print(f"[DEBUG] Normalized features for method {method}: shape {normalized_feats[method].shape}")
            
            # Concatenate all normalized features along axis=1 (per trial)
            combined_features = None
            for method, feats in normalized_feats.items():
                if combined_features is None:
                    combined_features = feats
                else:
                    if feats.shape[0] != combined_features.shape[0]:
                        raise ValueError("Mismatch in number of trials between methods.")
                    combined_features = np.concatenate((combined_features, feats), axis=1)
            
            # Optionally apply PCA for initial dimensionality reduction
            if 'dimensionality_reduction' in feat_cfg:
                dim_red_cfg = feat_cfg.dimensionality_reduction
                name = dim_red_cfg['name']
                dr_kwargs = dim_red_cfg.get('kwargs', {})
                if name.lower() == 'pca':
                    explained_var = dr_kwargs.get('explained_variance', 0.95)
                    pca_model = PCA(n_components=explained_var)
                    combined_features = pca_model.fit_transform(combined_features)
            
            # Define labels (make sure they’re available for feature selection)
            labels = epochs.events[:, -1]
            
            # Now apply supervised feature selection if specified
            if 'feature_selection' in feat_cfg:
                fs_cfg = feat_cfg.feature_selection
                fs_method = fs_cfg.get('name', 'lasso')
                fs_kwargs = fs_cfg.get('kwargs', {})
                
                if fs_method.lower() == 'lasso':
                    alpha = fs_kwargs.get('alpha', 0.01)
                    from sklearn.linear_model import Lasso
                    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
                    lasso.fit(combined_features, labels)
                    selected_indices = np.where(lasso.coef_ != 0)[0]
                    if selected_indices.size == 0:
                        print(f"Lasso did not select any features for subject {subj}, session {sess_label}.")
                    else:
                        combined_features = combined_features[:, selected_indices]
                
                elif fs_method.lower() == 'elasticnet':
                    alphas = fs_kwargs.get('alphas', [0.001, 0.01, 0.1, 1.0, 10.0])
                    l1_ratios = fs_kwargs.get('l1_ratios', [0.1, 0.5, 0.9])
                    elasticnet_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5,
                                                 max_iter=10000, random_state=42)
                    elasticnet_cv.fit(combined_features, labels)
                    selected_indices = np.where(elasticnet_cv.coef_ != 0)[0]
                    if selected_indices.size == 0:
                        print(f"ElasticNet did not select any features for subject {subj}, session {sess_label}.")
                    else:
                        combined_features = combined_features[:, selected_indices]
                
                elif fs_method.lower() == 'rfecv':
                    from sklearn.svm import SVC
                    from sklearn.feature_selection import RFECV
                    from sklearn.model_selection import GridSearchCV

                    # Default RFECV parameters (can be overridden via config)
                    step = fs_kwargs.get('step', 1)
                    cv = fs_kwargs.get('cv', 5)
                    scoring = fs_kwargs.get('scoring', 'accuracy')
                    min_features_to_select = fs_kwargs.get('min_features_to_select', 1)
                    # Use a linear SVM as the estimator for RFECV
                    estimator = SVC(kernel='linear', random_state=42)

                    # Instantiate RFECV with the defaults
                    rfecv = RFECV(estimator=estimator, step=step, cv=cv,
                                  scoring=scoring, min_features_to_select=min_features_to_select,
                                  n_jobs=-1)
                    
                    # Optionally, perform grid search over RFECV parameters if a param_grid is provided
                    param_grid = fs_kwargs.get('param_grid', {})
                    if param_grid:
                        grid = GridSearchCV(rfecv, param_grid=param_grid, cv=cv,
                                            scoring=scoring, n_jobs=-1)
                        grid.fit(combined_features, labels)
                        best_rfecv = grid.best_estimator_
                        selected_indices = np.where(best_rfecv.support_)[0]
                        print(f"RFECV grid search best params: {grid.best_params_}")
                    else:
                        rfecv.fit(combined_features, labels)
                        selected_indices = np.where(rfecv.support_)[0]
                    
                    if selected_indices.size == 0:
                        print(f"RFECV did not select any features for subject {subj}, session {sess_label}.")
                    else:
                        combined_features = combined_features[:, selected_indices]
                
                else:
                    print(f"Warning: Unrecognized feature selection method '{fs_method}'. Skipping feature selection.")
            
            # Save the combined features and labels for this session
            feat_result = {'combined': combined_features, 'labels': labels}
            features[subj][sess_label] = feat_result
            print(f"Extracted and combined features for subject {subj}, session {sess_label}")
    return features

"""def save_final_data(final_data, filename="./outputs/preprocessed_data.pkl"):
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(final_data, f)
    print(f"Final preprocessed data (after feature selection) saved to {filename}")
    """
def run_pipeline(cfg):
    preprocessed_data = run_preprocessing(cfg)
    features = run_feature_extraction_stage(cfg, preprocessed_data)
    # Save the final features (post feature extraction & selection) to disk.
    #final_output_file = cfg.get("preprocessed_output", "./outputs/preprocessed_data.pkl")
    #save_final_data(features, final_output_file)
    return features

def save_features(features, feat_cfg, filename="./features.pkl"):
    feat_cfg = OmegaConf.select(feat_cfg, "feature_extraction")
    methods_cfg = feat_cfg.get('methods', [])
    method_names = [str(method_cfg.get('name', 'unknown')) for method_cfg in methods_cfg]
    # Include dimensionality reduction and feature selection steps in the filename
    dr = feat_cfg.get('dimensionality_reduction', {}).get('name', '')
    fs = feat_cfg.get('feature_selection', {}).get('name', '')
    methods_str = "_".join(method_names)
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_{methods_str}_{dr}_{fs}{ext}"
    with open(new_filename, "wb") as f:
        pickle.dump(features, f)
    print(f"Features saved to {new_filename}")

def aggregate_data(features, session):
    X_list = []
    y_list = []
    for subj, sessions in features.items():
        if session in sessions:
            X_list.append(sessions[session]['combined'])
            y_list.append(sessions[session]['labels'])
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y
    
def main():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/config.yaml")
    feat_cfg = OmegaConf.load("config/feature_extraction/feature_extraction.yaml")
    features = run_pipeline(cfg)
    save_features(features, feat_cfg)
    # (Further stages: clustering, classification, etc.)
    print("Pipeline complete.")

if __name__ == "__main__":
    main()

"""
    #!/usr/bin/env python
import os
import pickle
import numpy as np
from lib.pipeline import run_preprocessing_pipeline, save_preprocessed_data
from lib.feature_extraction import run_feature_extraction
from lib.cluster import compute_subject_level_representation, subject_level_clustering
from sklearn.preprocessing import StandardScaler

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
            if not feat_dict:
                print(f"No features extracted for subject {subj}, session {sess_label}")
                continue
            scaler = StandardScaler()
            normalized_feats = {}
            for method, feats in feat_dict.items():
                normalized_feats[method] = scaler.fit_transform(feats)
            combined_features = None
            for method, feats in normalized_feats.items():
                if combined_features is None:
                    combined_features = feats
                else:
                    if feats.shape[0] != combined_features.shape[0]:
                        raise ValueError("Mismatch in number of trials between methods.")
                    combined_features = np.concatenate((combined_features, feats), axis=1)
            labels = epochs.events[:, -1]
            features[subj][sess_label] = {'combined': combined_features, 'labels': labels}
            print(f"Extracted features for subject {subj}, session {sess_label}")
    return features

def aggregate_data(features, session):
    X_list = []
    y_list = []
    for subj, sessions in features.items():
        if session in sessions:
            X_list.append(sessions[session]['combined'])
            y_list.append(sessions[session]['labels'])
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

def run_pipeline(cfg):
    preprocessed_data = run_preprocessing(cfg)
    features = run_feature_extraction_stage(cfg, preprocessed_data)
    return features

def save_features(features, cfg, filename="./features.pkl"):
    methods_cfg = cfg.feature_extraction.get('methods', [])
    method_names = [str(method_cfg.get('name', 'unknown')) for method_cfg in methods_cfg]
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_{'_'.join(method_names)}{ext}"
    with open(new_filename, "wb") as f:
        pickle.dump(features, f)
    print(f"Features saved to {new_filename}")

def main():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("vt2/config/config.yaml")
    features = run_pipeline(cfg)
    save_features(features, cfg)
    # (Further stages: clustering, classification, etc.)
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
"""