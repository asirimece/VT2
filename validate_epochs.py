#!/usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import mne

def load_preprocessed_data(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

def plot_label_distribution(preprocessed_data, subject=1, session="0train"):
    epochs = preprocessed_data[subject][session]
    labels = epochs.events[:, 2]
    label_counts = Counter(labels)
    labels_sorted = sorted(label_counts.keys())
    counts = [label_counts[l] for l in labels_sorted]
    
    fig, ax = plt.subplots()
    ax.bar([str(l) for l in labels_sorted], counts, color='skyblue')
    ax.set_xlabel("Event Code")
    ax.set_ylabel("Number of Epochs")
    ax.set_title(f"Label Distribution for Subject {subject}, Session {session}")
    fig.savefig("epoch_label_distribution.png")
    plt.close(fig)
    print("Saved label distribution plot as 'epoch_label_distribution.png'.")

def plot_random_epochs(preprocessed_data, subject=1, session="0train", n_epochs=6):
    epochs = preprocessed_data[subject][session]
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    times = epochs.times
    n_total = data.shape[0]
    random_indices = np.random.choice(n_total, n_epochs, replace=False)
    
    fig, axs = plt.subplots(n_epochs, 1, figsize=(10, 2*n_epochs), sharex=True)
    if n_epochs == 1:
        axs = [axs]
    for i, idx in enumerate(random_indices):
        axs[i].plot(times, data[idx, 0, :], color='tab:blue')
        axs[i].set_title(f"Subject {subject}, Session {session}, Epoch {idx} (Channel 0)")
        axs[i].set_ylabel("Amplitude (µV)")
    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig("epoch_sample.png")
    plt.close(fig)
    print("Saved sample epochs plot as 'epoch_sample.png'.")

def main():
    # Path to your preprocessed data (adjust the path as needed)
    filepath = "./outputs/2025-03-12/11-39-55/outputs/preprocessed_data.pkl"
    preprocessed_data = load_preprocessed_data(filepath)
    
    print("\n----- Validating Labeling -----")
    plot_label_distribution(preprocessed_data, subject=1, session="0train")
    
    print("\n----- Visual Inspection of Sample Epochs -----")
    plot_random_epochs(preprocessed_data, subject=1, session="0train", n_epochs=6)

if __name__ == '__main__':
    main()
