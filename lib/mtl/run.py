# multitask_trainer.py (Updated Main Evaluation Section)
import torch
from torch.utils.data import DataLoader
from lib.mtl.trainer import EEGMultiTaskDataset, train_mtl_model, evaluate_mtl_model, MTLWrapper
from lib.mtl.model import MultiTaskDeep4Net
from lib.cluster.cluster import ClusterWrapper, SubjectClusterer
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.mtl.utils import convert_state_dict_keys
from lib.base.trainer import BaseWrapper

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
            # Convert subject ID to string.
            subject_ids_list.extend([str(subj)] * data.shape[0])
    
    X = np.concatenate(data_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y, subject_ids_list

def load_cluster_wrapper(config_path="config/experiment/mtl.yaml", features_file="./outputs/features.pkl"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # If your file has clustering nested under "experiment", get it like this:
    clustering_config = config.get('experiment', {}).get('clustering', {})
    subject_clusterer = SubjectClusterer(features_file, clustering_config)
    cluster_wrapper = subject_clusterer.cluster_subjects(method=clustering_config.get('method', 'kmeans'))
    return cluster_wrapper

def plot_cluster_distribution(cluster_wrapper, output_file="cluster_distribution.png"):
    assignments = cluster_wrapper.labels
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
    
    """from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap="viridis", s=100)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Subject-level Representations (PCA)")
    plt.colorbar(scatter, label="Cluster Label")
    plt.savefig(output_file)
    plt.close()"""

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
    dataset = EEGMultiTaskDataset(data, labels, subject_ids, cluster_wrapper)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up the model.
    backbone_kwargs = {"n_times": data.shape[2]}
    model = MultiTaskDeep4Net(n_chans=data.shape[1], n_outputs=4, n_clusters=n_clusters, backbone_kwargs=backbone_kwargs)
    
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Starting MTL training...")
    model = train_mtl_model(model, dataloader, criterion, optimizer, device, num_epochs=100)
    
    print("Evaluating MTL model...")
    subjects, all_labels, all_preds = evaluate_mtl_model(model, dataloader, device)
    
    # ----------------------------
    # Aggregate Subject-Level Results
    # ----------------------------
    subject_results = {}
    for subj, gt, pred in zip(subjects, all_labels, all_preds):
        subj_str = str(subj)
        if subj_str not in subject_results:
            subject_results[subj_str] = {"ground_truth": [], "predictions": []}
        subject_results[subj_str]["ground_truth"].append(gt)
        subject_results[subj_str]["predictions"].append(pred)
    
    # Get cluster assignments from cluster_wrapper.
    cluster_assignments = cluster_wrapper.labels
    
    # Create and save the MTLWrapper instance.
    mtl_wrapper = MTLWrapper(subject_results, cluster_assignments=cluster_assignments)
    mtl_wrapper.save("mtl_wrapper_results.pkl")
    print("MTLWrapper results saved to mtl_wrapper_results.pkl")
    
    # Save model weights.
    converted_state_dict = convert_state_dict_keys(model.state_dict())
    torch.save(converted_state_dict, "pretrained_mtl_model_weights.pth")
    print("MTL model weights saved to pretrained_mtl_model_weights.pth")
    
    # ----------------------------
    # Evaluate Against Baselines
    # ----------------------------
    # Load evaluation configuration.
    with open("config/experiment/mtl.yaml", "r") as f:
        eval_config = yaml.safe_load(f)
    
    from lib.mtl.evaluate import MTLEvaluator
    
    # Load both baseline wrappers.
    baseline_pooled = MTLWrapper.load("./trained_models/pooled_baseline_results.pkl")
    baseline_single = MTLWrapper.load("./trained_models/single_baseline_results.pkl")
    
    # Evaluate MTL results against the pooled baseline.
    print("\nEvaluating MTL results against the pooled baseline:")
    evaluator_pooled = MTLEvaluator(mtl_wrapper, baseline_pooled, eval_config)
    summary_pooled = evaluator_pooled.evaluate(verbose=True)
    
    # Evaluate MTL results against the single-subject baseline.
    print("\nEvaluating MTL results against the single-subject baseline:")
    evaluator_single = MTLEvaluator(mtl_wrapper, baseline_single, eval_config)
    summary_single = evaluator_single.evaluate(verbose=True)
    
    # Combine and save summaries.
    combined_summary = {
        "pooled_baseline": summary_pooled,
        "single_baseline": summary_single
    }
    with open("mtl_evaluation_summary.pkl", "wb") as f:
        pickle.dump(combined_summary, f)
    print("Combined evaluation summary saved to mtl_evaluation_summary.pkl")

if __name__ == "__main__":
    main()
