import torch
from torch.utils.data import DataLoader
from lib.dataset.dataset import EEGMultiTaskDataset
from lib.mtl.train import train_mtl_model, evaluate_mtl_model
from lib.mtl.model import MultiTaskDeep4Net
from lib.pipeline.cluster.cluster import ClusterWrapper, SubjectClusterer  # Import your implemented ClusterWrapper
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lib.mtl.train import MTLWrapper  # MTLWrapper class from your multitask_trainer.py
from lib.mtl.evaluate import MTLEvaluator   # Your evaluator module
from lib.utils.utils import convert_state_dict_keys

def load_preprocessed_data(preprocessed_file="dump/preprocessed_data.pkl"):
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

def load_cluster_wrapper(config_path="config/dataset/bci_iv2a.yaml", features_file="dump/features.pkl"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    clustering_config = config.get('clustering', {})
    subject_clusterer = SubjectClusterer(features_file, clustering_config)
    cluster_wrapper = subject_clusterer.cluster_subjects(method=clustering_config.get('method', 'kmeans'))
    return cluster_wrapper

def plot_cluster_distribution(cluster_wrapper, output_file="cluster_distribution.png"):
    assignments = cluster_wrapper.labels  # dict: subject_id -> cluster label
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
    
    from sklearn.decomposition import PCA
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
    data, labels, subject_ids = load_preprocessed_data("dump/preprocessed_data.pkl")
    
    # Load clustering results.
    cluster_wrapper = load_cluster_wrapper()
    n_clusters = cluster_wrapper.get_num_clusters()
    
    # Generate intermediary clustering plots.
    plot_cluster_distribution(cluster_wrapper, output_file="cluster_distribution.png")
    plot_subject_scatter(cluster_wrapper, output_file="subject_scatter.png")
    
    # Create dataset and dataloader.
    # Note: EEGMultiTaskDataset now returns (sample, label, subject_id, cluster_id)
    dataset = EEGMultiTaskDataset(data, labels, subject_ids, cluster_wrapper)
    # Use shuffle=False for evaluation so subject_ids remain in order.
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pass backbone_kwargs with n_times.
    backbone_kwargs = {"n_times": data.shape[2]}
    model = MultiTaskDeep4Net(n_chans=data.shape[1], n_outputs=4, n_clusters=n_clusters, backbone_kwargs=backbone_kwargs)
    
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Starting MTL training...")
    model = train_mtl_model(model, 
                            dataloader, 
                            criterion, 
                            optimizer, 
                            device, 
                            epochs=mtl_config["epochs"],
                             lambda_bias=mtl_config["lambda_bias"][0])
    
    print("Evaluating MTL model...")
    # Run evaluation; our evaluate_mtl_model now returns subject_ids, ground truth, and predictions.
    subjects, all_labels, all_preds = evaluate_mtl_model(model, dataloader, device)
    
    # ----------------------------
    # Aggregate Subject-Level Results
    # ----------------------------
    subject_results = {}
    for subj, gt, pred in zip(subjects, all_labels, all_preds):
        if subj not in subject_results:
            subject_results[subj] = {"ground_truth": [], "predictions": []}
        subject_results[subj]["ground_truth"].append(gt)
        subject_results[subj]["predictions"].append(pred)
    
    # Use your cluster_wrapper to get cluster assignments (adjust if stored differently)
    cluster_assignments = cluster_wrapper.labels
    
    # Create the MTLWrapper instance and save it.
    mtl_wrapper = MTLWrapper(subject_results, cluster_assignments=cluster_assignments)
    mtl_wrapper.save("mtl_wrapper_results.pkl")
    print("MTLWrapper results saved to mtl_wrapper_results.pkl")
    
    # Save the entire MTL model
    #torch.save(model, "pretrained_mtl_model.pth")
    # Save only weights
    # Convert the state dict keys to a consistent format and re-save the weights.
    converted_state_dict = convert_state_dict_keys(model.state_dict())
    torch.save(converted_state_dict, "pretrained_mtl_model_weights.pth")
    print("MTL model weights saved to pretrained_mtl_model_weights.pth")
    
    # ----------------------------
    # Integrate the Evaluator
    # ----------------------------
    # Load your baseline results (make sure training_results.pkl exists)
    with open("./trained_models/training_results.pkl", "rb") as f:
        baseline_results = pickle.load(f)
    
    # Load evaluation configuration from config/experiment/mtl.yaml
    with open("config/experiment/mtl.yaml", "r") as f:
        mtl_config = yaml.safe_load(f)
    
    # Now pass the configuration as the third argument.
    evaluator = MTLEvaluator(mtl_wrapper, baseline_results, mtl_config)
    evaluator.evaluate(verbose=True)

if __name__ == "__main__":
    main()
