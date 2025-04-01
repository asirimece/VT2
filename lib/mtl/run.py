# run_mtl.py

import torch
from torch.utils.data import DataLoader
from lib.mtl.trainer import EEGMultiTaskDataset, train_mtl_model, evaluate_mtl_model
from lib.mtl.model import MultiTaskDeep4Net
from lib.cluster.cluster import ClusterWrapper, SubjectClusterer  # Import your implemented ClusterWrapper
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def load_preprocessed_data(preprocessed_file="outputs/preprocessed_data.pkl"):
    with open(preprocessed_file, "rb") as f:
        preprocessed = pickle.load(f)
    
    data_list = []
    labels_list = []
    subject_ids_list = []
    for subj, sessions in preprocessed.items():
        for sess, epochs in sessions.items():
            data = epochs.get_data()  # shape: [n_trials, channels, time]
            labels = epochs.events[:, -1]
            data_list.append(data)
            labels_list.append(labels)
            subject_ids_list.extend([subj] * data.shape[0])
    
    X = np.concatenate(data_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y, subject_ids_list

def load_cluster_wrapper(config_path="vt2/config/dataset/bci_iv2a.yaml", features_file="outputs/2025-03-28/both_ems/features___.pkl"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    clustering_config = config.get('clustering', {})
    subject_clusterer = SubjectClusterer(features_file, clustering_config)
    cluster_wrapper = subject_clusterer.cluster_subjects(method=clustering_config.get('method', 'kmeans'))
    return cluster_wrapper

def plot_cluster_distribution(cluster_wrapper, output_file="cluster_distribution.png"):
    assignments = cluster_wrapper.labels  # dict: subject_id -> cluster label (or dict)
    cleaned_labels = []
    for value in assignments.values():
        if isinstance(value, dict):
            label = value.get("cluster", None)
        else:
            label = value
        cleaned_labels.append("None" if label is None else str(label))
    
    unique, counts = np.unique(cleaned_labels, return_counts=True)
    plt.figure(figsize=(8, 6))
    plt.bar(unique, counts, color="skyblue")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Subjects")
    plt.title("Cluster Distribution")
    plt.savefig(output_file)
    plt.close()

def plot_subject_scatter(cluster_wrapper, output_file="subject_scatter.png"):
    subject_ids = list(cluster_wrapper.subject_representations.keys())
    X = np.array([cluster_wrapper.subject_representations[sid] for sid in subject_ids])
    numeric_labels = []
    for sid in subject_ids:
        label = cluster_wrapper.get_cluster_for_subject(sid)
        if isinstance(label, dict):
            label = label.get("cluster", None)
        try:
            numeric_labels.append(float(label))
        except (TypeError, ValueError):
            numeric_labels.append(-1)
    labels = np.array(numeric_labels)
    
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap="viridis", s=100)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Subject-level Representations (PCA)")
    plt.colorbar(scatter, label="Cluster Label")
    plt.savefig(output_file)
    plt.close()

def main():
    # Load preprocessed EEG data.
    data, labels, subject_ids = load_preprocessed_data("outputs/preprocessed_data.pkl")
    
    # Load clustering results.
    cluster_wrapper = load_cluster_wrapper()
    n_clusters = cluster_wrapper.get_num_clusters()
    
    # Generate intermediary clustering plots.
    plot_cluster_distribution(cluster_wrapper, output_file="cluster_distribution.png")
    plot_subject_scatter(cluster_wrapper, output_file="subject_scatter.png")
    
    # Create dataset and dataloader.
    from lib.mtl.trainer import EEGMultiTaskDataset 
    dataset = EEGMultiTaskDataset(data, labels, subject_ids, cluster_wrapper)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pass backbone_kwargs with input_window_samples.
    backbone_kwargs = {"input_window_samples": data.shape[2]}  # data.shape[2] should be 500
    model = MultiTaskDeep4Net(n_chans=data.shape[1], n_outputs=4, n_clusters=n_clusters, backbone_kwargs=backbone_kwargs)
    
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Starting MTL training...")
    model = train_mtl_model(model, dataloader, criterion, optimizer, device, num_epochs=100)
    
    print("Evaluating MTL model...")
    all_labels, all_preds = evaluate_mtl_model(model, dataloader, device)
    
    # Save training results.
    results_dict = {"ground_truth": all_labels, "predictions": all_preds}
    with open("mtl_training_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)
    print("MTL training results saved to mtl_training_results.pkl")
    
if __name__ == "__main__":
    main()
