import pickle
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def load_preprocessed_data(filename, top_key=None, nested_key=None, final_key=None):
    """
    Load preprocessed data from a pickle file by drilling down through nested dictionaries.
    
    Parameters:
      - filename: Path to the pickle file.
      - top_key: Top-level key in the dictionary (e.g. 1).
      - nested_key: Nested key within the top-level value (e.g. "0train").
      - final_key: Final key that holds the actual data (e.g. "combined").
    
    Returns:
      The extracted data.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        if top_key is not None:
            if top_key in data:
                print(f"Extracting data using top-level key '{top_key}' from {filename}.")
                data = data[top_key]
            else:
                raise ValueError(f"Top-level key '{top_key}' not found in {filename}. Available keys: {list(data.keys())}")
        if nested_key is not None:
            if nested_key in data:
                print(f"Extracting nested key '{nested_key}' from {filename}.")
                data = data[nested_key]
            else:
                raise ValueError(f"Nested key '{nested_key}' not found in {filename}. Available keys: {list(data.keys())}")
        if final_key is not None:
            if final_key in data:
                print(f"Extracting final key '{final_key}' from {filename}.")
                data = data[final_key]
            else:
                raise ValueError(f"Final key '{final_key}' not found in {filename}. Available keys: {list(data.keys())}")
        return data
    else:
        return data

def compare_psd_mne(data1, data2, channel, fmax=50):
    """
    Compare PSD using MNE's built-in compute_psd method.
    (This branch is used if data1 and data2 are MNE Raw or Epochs objects.)
    """
    psd1, freqs1 = data1.compute_psd(picks=[channel], fmax=fmax, method='welch', verbose=False)
    psd2, freqs2 = data2.compute_psd(picks=[channel], fmax=fmax, method='welch', verbose=False)
    
    psd1_mean = np.mean(psd1, axis=0)
    psd2_mean = np.mean(psd2, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs1, psd1_mean, label="ICA 22 Components")
    plt.semilogy(freqs2, psd2_mean, label="ICA 20 Components")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (Power/Hz)")
    plt.title(f"PSD Comparison for {channel}")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_eog_variance_mne(data1, data2, eog_channels):
    """
    Compare variance in EOG channels using MNE objects.
    """
    d1 = data1.get_data(picks=eog_channels)
    d2 = data2.get_data(picks=eog_channels)
    var1 = np.var(d1)
    var2 = np.var(d2)
    
    print("Variance in EOG channels:")
    print(f"ICA 22 Components: {var1:.4f}")
    print(f"ICA 20 Components: {var2:.4f}")

def compare_psd_custom(data1, data2, channel_idx, fs=250, fmax=50):
    """
    Compute and plot PSD for numpy array data (assumed shape (n_channels, n_times)).
    
    Parameters:
      - data1, data2: NumPy arrays from the two pipelines.
      - channel_idx: The index of the channel to compare (e.g. 0 for EOG1).
      - fs: Sampling frequency.
      - fmax: Maximum frequency for plotting.
    """
    psd1, freqs1 = welch(data1[channel_idx, :], fs=fs)
    psd2, freqs2 = welch(data2[channel_idx, :], fs=fs)
    
    # Limit to frequencies <= fmax.
    valid1 = freqs1 <= fmax
    valid2 = freqs2 <= fmax
    psd1 = psd1[valid1]
    psd2 = psd2[valid2]
    freqs1 = freqs1[valid1]
    freqs2 = freqs2[valid2]
    
    plt.figure(figsize=(10,6))
    plt.semilogy(freqs1, psd1, label="ICA 22 Components")
    plt.semilogy(freqs2, psd2, label="ICA 20 Components")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (Power/Hz)")
    plt.title(f"PSD Comparison for channel index {channel_idx}")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_eog_variance_custom(data1, data2, eog_channel_indices):
    """
    Compute and print overall variance for specified EOG channel indices from numpy array data.
    """
    var1 = np.var(data1[eog_channel_indices, :])
    var2 = np.var(data2[eog_channel_indices, :])
    print("Variance in EOG channels:")
    print(f"ICA 22 Components: {var1:.4f}")
    print(f"ICA 20 Components: {var2:.4f}")

def main():
    # Paths to your saved pickle files.
    file_22 = "outputs/22ica/22ica_features.pkl"
    file_20 = "outputs/20ica/20ica_features.pkl"
    
    # Drill down through the dictionaries:
    # Top-level key is 1, nested key is "0train", and the final key "combined" holds the data.
    data_22 = load_preprocessed_data(file_22, top_key=1, nested_key="0train", final_key="combined")
    data_20 = load_preprocessed_data(file_20, top_key=1, nested_key="0train", final_key="combined")
    
    print("Type of data from 22ica:", type(data_22))
    print("Type of data from 20ica:", type(data_20))
    
    if isinstance(data_22, (mne.io.BaseRaw, mne.Epochs)) and isinstance(data_20, (mne.io.BaseRaw, mne.Epochs)):
        # Use MNE methods if the data are MNE objects.
        compare_psd_mne(data_22, data_20, channel="EOG1", fmax=50)
        eog_channels = ["EOG1", "EOG2", "EOG3"]
        compare_eog_variance_mne(data_22, data_20, eog_channels)
    elif isinstance(data_22, np.ndarray) and isinstance(data_20, np.ndarray):
        # For numpy arrays, use custom PSD and variance calculations.
        fs = 250  # Set the sampling frequency (adjust as needed).
        # Assume that channel index 0 corresponds to EOG1.
        compare_psd_custom(data_22, data_20, channel_idx=0, fs=fs, fmax=50)
        # Assume that the EOG channels are indices 0, 1, and 2.
        compare_eog_variance_custom(data_22, data_20, eog_channel_indices=[0, 1, 2])
    else:
        raise ValueError("Data types from 22ica and 20ica do not match or are unsupported.")

if __name__ == "__main__":
    main()
