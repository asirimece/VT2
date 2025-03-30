#!/usr/bin/env python
"""
csp_riemann_lda_check.py

This script loads the preprocessed sub-epoched data from a pickle file,
extracts features using two methods:
  (1) Filter Bank Common Spatial Patterns (FBCSP) via MNE’s CSP,
  (2) Riemannian feature extraction (covariance + Tangent Space mapping),
concatenates the features, applies RFECV for feature selection, and then
evaluates an LDA classifier using 5-fold cross-validation.

Usage:
    python csp_riemann_lda_check.py
"""

import os
import pickle
import numpy as np
import mne

# For CSP (FBCSP)
from mne.decoding import CSP

# For Riemannian feature extraction
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# For feature selection and classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

def extract_fbcsp_features(epochs, n_components=4):
    """
    Extract CSP (log-variance) features using MNE’s CSP.
    
    Parameters:
        epochs: an MNE Epochs object with shape (n_epochs, n_channels, n_times).
        n_components: number of CSP components to extract.
        
    Returns:
        X_csp: numpy array of shape (n_epochs, n_components)
        csp: fitted CSP object (for inspection, if needed)
    """
    # Convert data to double precision (float64) to avoid data-copy issues.
    X = epochs.get_data().astype(np.float64)
    y = epochs.events[:, -1]  # assuming labels are stored in the last column
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y)
    return X_csp, csp

def extract_riemann_features(epochs):
    """
    Extract Riemannian features from the epochs.
    Computes the covariance matrix for each epoch and maps them to the tangent space.
    
    Returns:
        X_riemann: numpy array of shape (n_epochs, n_features)
        ts: fitted TangentSpace object
    """
    X = epochs.get_data().astype(np.float64)
    cov_estimator = Covariances(estimator='oas')
    covmats = cov_estimator.fit_transform(X)
    ts = TangentSpace()
    X_riemann = ts.fit_transform(covmats)
    return X_riemann, ts

def main():
    # Load preprocessed sub-epoched data (from your pipeline)
    data_file = "./outputs/preprocessed_data.pkl"
    with open(data_file, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    # For this check, select a single subject (e.g., subject 1 from the training session)
    subj = 1
    if subj not in preprocessed_data:
        raise ValueError(f"Subject {subj} not found in preprocessed data.")
    
    # Use the training session sub-epochs (e.g., "0train")
    epochs = preprocessed_data[subj]["0train"]
    print("Using training epochs with shape:", epochs.get_data().shape)
    
    # --- Feature Extraction ---
    # 1. FBCSP features
    X_csp, csp_obj = extract_fbcsp_features(epochs, n_components=4)
    print("FBCSP features shape:", X_csp.shape)
    
    # 2. Riemannian features
    X_riemann, ts_obj = extract_riemann_features(epochs)
    print("Riemannian features shape:", X_riemann.shape)
    
    # Concatenate features along the feature axis
    X_features = np.concatenate([X_csp, X_riemann], axis=1)
    print("Concatenated feature vector shape:", X_features.shape)
    
    # Get labels from epochs (assumed to be already mapped to 0,1,2,3)
    y = epochs.events[:, -1]
    print("Unique labels in training data:", np.unique(y))
    
    # --- Feature Selection with RFECV using LDA ---
    lda = LinearDiscriminantAnalysis()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfecv = RFECV(estimator=lda, step=1, cv=cv, scoring='accuracy')
    rfecv.fit(X_features, y)
    print("RFECV selected features mask:", rfecv.support_)
    print("Number of selected features:", np.sum(rfecv.support_))
    
    X_selected = rfecv.transform(X_features)
    
    # --- LDA Classification via 5-fold CV on Selected Features ---
    cv_scores = cross_val_score(lda, X_selected, y, cv=cv, scoring='accuracy')
    print("LDA Classification Accuracy (5-fold CV):", np.mean(cv_scores) * 100, "%")
    
    # Train LDA on the full training set and display the confusion matrix
    lda.fit(X_selected, y)
    y_pred = lda.predict(X_selected)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)
    
if __name__ == "__main__":
    main()
