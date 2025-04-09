import numpy as np
from lib.logging import logger
import mne
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

logger = logger.get()

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
    
    def extract_erd_ers(self, epochs):
        config = self.config.feature_extraction
        kwargs = config.methods[0].get('kwargs', {}) if config.methods else {}
        baseline_window = kwargs.get('baseline_window', [0.0, 0.5])
        analysis_window = kwargs.get('analysis_window', [0.5, 4.0])
        frequency_bands = kwargs.get('frequency_bands', {'mu': [8, 12], 'beta': [13, 30]})
        
        sfreq = epochs.info['sfreq']
        times = epochs.times
        data = epochs.get_data()
        tmin, tmax = times[0], times[-1]
        
        def time_to_index(sec):
            sec_clamped = np.clip(sec, tmin, tmax)
            return np.searchsorted(times, sec_clamped)
        
        b_start_idx = time_to_index(baseline_window[0])
        b_end_idx = time_to_index(baseline_window[1])
        a_start_idx = time_to_index(analysis_window[0])
        a_end_idx = time_to_index(analysis_window[1])
        
        n_epochs = data.shape[0]
        n_bands = len(frequency_bands)
        erd_ers_features = np.zeros((n_epochs, n_bands))
        band_names = list(frequency_bands.keys())
        
        for b_idx, band_name in enumerate(band_names):
            low_f, high_f = frequency_bands[band_name]
            data_band = data.copy()
            for ep in range(n_epochs):
                data_band[ep] = mne.filter.filter_data(data_band[ep], sfreq=sfreq,
                                                       l_freq=low_f, h_freq=high_f, verbose=False)
            baseline_power = np.mean(data_band[:, :, b_start_idx:b_end_idx] ** 2, axis=(1, 2))
            analysis_power = np.mean(data_band[:, :, a_start_idx:a_end_idx] ** 2, axis=(1, 2))
            baseline_power = np.maximum(baseline_power, 1e-12)
            erd_ers_features[:, b_idx] = ((analysis_power - baseline_power) / baseline_power) * 100.0
        return erd_ers_features

    def extract_csp(self, epochs):
        kwargs = self.config.feature_extraction.methods[0].get('kwargs', {}) if self.config.feature_extraction.methods else {}
        frequency_band = kwargs.get('frequency_band', [8, 30])
        n_components = kwargs.get('n_components', 4)
        labels = epochs.events[:, -1]
        if len(np.unique(labels)) < 2:
            raise ValueError("CSP requires at least 2 classes.")
        data = epochs.get_data()
        csp = CSP(n_components=n_components, reg=None, norm_trace=False)
        csp_features = csp.fit_transform(data, labels)
        return csp_features

    def extract_fbcsp(self, epochs):
        kwargs = self.config.feature_extraction.methods[0].get('kwargs', {}) if self.config.feature_extraction.methods else {}
        freq_bands = kwargs.get('frequency_bands', [[4, 8], [8, 12], [12, 16]])
        n_components_per_band = kwargs.get('n_components_per_band', 2)
        sfreq = epochs.info['sfreq']
        labels = epochs.events[:, -1]
        raw_data = epochs.get_data()
        all_features = []
        for (low_f, high_f) in freq_bands:
            data_band = raw_data.copy()
            n_epochs = data_band.shape[0]
            for ep in range(n_epochs):
                data_band[ep] = mne.filter.filter_data(data_band[ep], sfreq=sfreq,
                                                       l_freq=low_f, h_freq=high_f, verbose=False)
            csp = CSP(n_components=n_components_per_band, norm_trace=False)
            feats = csp.fit_transform(data_band, labels)
            all_features.append(feats)
        fbcsp_features = np.concatenate(all_features, axis=1)
        return fbcsp_features

    def extract_riemannian(self, epochs):
        kwargs = self.config.feature_extraction.methods[0].get('kwargs', {}) if self.config.feature_extraction.methods else {}
        estimator = kwargs.get('estimator', 'oas')
        mapping = kwargs.get('mapping', 'tangent')
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

    def run(self, epochs):
        """
        Execute all configured feature extraction methods.
        Returns a dictionary with method names as keys and feature matrices as values.
        """
        if not hasattr(self.config.feature_extraction, "methods") or not self.config.feature_extraction.methods:
            print("No feature extraction methods configured. Returning empty dict.")
            return {}
        
        feature_dict = {}
        for method_config in self.config.feature_extraction.methods:
            method_name = method_config['name']
            if method_name == 'erd_ers':
                feature_dict['erd_ers'] = self.extract_erd_ers(epochs)
            elif method_name == 'csp':
                feature_dict['csp'] = self.extract_csp(epochs)
            elif method_name == 'fbcsp':
                feature_dict['fbcsp'] = self.extract_fbcsp(epochs)
            elif method_name == 'riemannian':
                feature_dict['riemannian'] = self.extract_riemannian(epochs)
            else:
                logger.warning(f"Unrecognized feature extraction method '{method_name}'.")
        return feature_dict
