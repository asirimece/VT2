#!/usr/bin/env python
"""
ft_extract.py

Implements feature extraction methods for BCIC IV 2a (BNCI2014001) dataset:
  - ERD/ERS
  - CSP
  - FBCSP
  - Riemannian Geometry

Usage:
  1. Import the relevant functions (extract_erd_ers, extract_csp, etc.)
  2. Call run_feature_extraction(epochs, cfg) after your preprocessing pipeline
     has generated the 'epochs' for a subject/session.
     
Config Structure Example (bci_iv2a.yaml):
feature_extraction:
  methods:
    - name: erd_ers
      kwargs:
        baseline_window: [-0.5, 0.0]
        analysis_window: [0.0, 4.0]
        frequency_bands:
          mu: [8, 12]
          beta: [13, 30]
    - name: csp
      kwargs:
        frequency_band: [8, 30]
        n_components: 4
    - name: fbcsp
      kwargs:
        frequency_bands:
          - [4, 8]
          - [8, 12]
          - [12, 16]
          - [16, 20]
          - [20, 24]
          - [24, 28]
          - [28, 32]
          - [32, 38]
        n_components_per_band: 2
    - name: riemannian
      kwargs:
        estimator: "oas"
        mapping: "tangent"

  dimensionality_reduction:
    name: pca
    kwargs:
      explained_variance: 0.95
"""

import numpy as np
import mne
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

def extract_erd_ers(epochs, baseline_window, analysis_window, frequency_bands):
    sfreq = epochs.info['sfreq']
    times = epochs.times
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    tmin, tmax = times[0], times[-1]

    def time_to_index(sec):
        sec_clamped = np.clip(sec, tmin, tmax)
        idx = np.searchsorted(times, sec_clamped)
        return idx

    b_start_idx = time_to_index(baseline_window[0])
    b_end_idx   = time_to_index(baseline_window[1])
    a_start_idx = time_to_index(analysis_window[0])
    a_end_idx   = time_to_index(analysis_window[1])

    if b_end_idx - b_start_idx == 0:
        raise ValueError("Baseline window results in an empty slice.")

    n_epochs, _, _ = data.shape
    n_bands = len(frequency_bands)
    erd_ers_features = np.zeros((n_epochs, n_bands))
    band_names = list(frequency_bands.keys())

    for b_idx, band_name in enumerate(band_names):
        low_f, high_f = frequency_bands[band_name]
        data_band = data.copy()
        for ep in range(n_epochs):
            data_band[ep] = mne.filter.filter_data(data_band[ep], sfreq=sfreq, l_freq=low_f, h_freq=high_f, verbose=False)
        baseline_power = np.mean(data_band[:, :, b_start_idx:b_end_idx] ** 2, axis=(1,2))
        analysis_power = np.mean(data_band[:, :, a_start_idx:a_end_idx] ** 2, axis=(1,2))
        baseline_power = np.maximum(baseline_power, 1e-12)
        erd_ers_features[:, b_idx] = ((analysis_power - baseline_power) / baseline_power) * 100.0

    return erd_ers_features

def extract_csp(epochs, frequency_band, n_components):
    labels = epochs.events[:, -1]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError(f"CSP requires at least 2 classes. Found labels: {unique_labels}")
    data = epochs.get_data()
    csp = CSP(n_components=n_components, reg=None, norm_trace=False)
    csp_features = csp.fit_transform(data, labels)
    return csp_features

def extract_fbcsp(epochs, freq_bands, n_components_per_band):
    sfreq = epochs.info['sfreq']
    labels = epochs.events[:, -1]
    raw_data = epochs.get_data()
    all_features = []
    for (low_f, high_f) in freq_bands:
        data_band = raw_data.copy()
        n_epochs, _, _ = data_band.shape
        for ep in range(n_epochs):
            data_band[ep] = mne.filter.filter_data(data_band[ep], sfreq=sfreq, l_freq=low_f, h_freq=high_f, verbose=False)
        csp = CSP(n_components=n_components_per_band, norm_trace=False)
        feats = csp.fit_transform(data_band, labels)
        all_features.append(feats)
    fbcsp_features = np.concatenate(all_features, axis=1)
    return fbcsp_features

def extract_riemannian(epochs, estimator, mapping):
    data = epochs.get_data()
    cov_estimator = Covariances(estimator=estimator)
    cov_mats = cov_estimator.fit_transform(data)
    if mapping == "tangent":
        ts = TangentSpace()
        riemann_features = ts.fit_transform(cov_mats)
    else:
        n_epochs = cov_mats.shape[0]
        tri_indices = np.triu_indices(cov_mats.shape[1])
        n_features = len(tri_indices[0])
        riemann_features = np.zeros((n_epochs, n_features))
        for i in range(n_epochs):
            riemann_features[i, :] = cov_mats[i][tri_indices]
    return riemann_features

def run_feature_extraction(epochs, pipeline_cfg):
    if 'feature_extraction' not in pipeline_cfg:
        print("No feature_extraction config found, returning empty dict.")
        return {}
    methods_cfg = pipeline_cfg.feature_extraction.get('methods', [])
    feature_dict = {}
    for method_cfg in methods_cfg:
        method_name = method_cfg['name']
        kwargs = method_cfg.get('kwargs', {})
        if method_name == 'erd_ers':
            feats = extract_erd_ers(
                epochs=epochs,
                baseline_window=kwargs.get('baseline_window', [0.0, 0.5]),
                analysis_window=kwargs.get('analysis_window', [0.5, 4.0]),
                frequency_bands=kwargs.get('frequency_bands', {'mu':[8,12], 'beta':[13,30]})
            )
            feature_dict['erd_ers'] = feats
        elif method_name == 'csp':
            feats = extract_csp(
                epochs=epochs,
                frequency_band=kwargs.get('frequency_band', [8, 30]),
                n_components=kwargs.get('n_components', 4)
            )
            feature_dict['csp'] = feats
        elif method_name == 'fbcsp':
            feats = extract_fbcsp(
                epochs=epochs,
                freq_bands=kwargs.get('frequency_bands', [[4,8],[8,12],[12,16]]),
                n_components_per_band=kwargs.get('n_components_per_band', 2)
            )
            feature_dict['fbcsp'] = feats
        elif method_name == 'riemannian':
            feats = extract_riemannian(
                epochs=epochs,
                estimator=kwargs.get('estimator', 'oas'),
                mapping=kwargs.get('mapping', 'tangent')
            )
            feature_dict['riemannian'] = feats
        else:
            print(f"Warning: Unrecognized method '{method_name}'")
    # Apply dimensionality reduction if specified (applied per subject/session)
    if 'dimensionality_reduction' in pipeline_cfg.feature_extraction:
        dim_red_cfg = pipeline_cfg.feature_extraction.dimensionality_reduction
        name = dim_red_cfg['name']
        dr_kwargs = dim_red_cfg.get('kwargs', {})
        if name.lower() == 'pca':
            from sklearn.decomposition import PCA
            explained_var = dr_kwargs.get('explained_variance', 0.95)
            # Apply PCA to each method's features individually.
            for m_name, feats in feature_dict.items():
                pca_model = PCA(n_components=explained_var)
                feature_dict[m_name] = pca_model.fit_transform(feats)
    return feature_dict
