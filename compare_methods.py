#!/usr/bin/env python
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score

def load_features(features_file):
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

def inspect_features(features_dict):
    print("Inspecting loaded features dictionary...")
    subjects = list(features_dict.keys())
    print("Subjects available:", subjects)
    for subj in subjects:
        sessions = list(features_dict[subj].keys())
        print(f"Subject {subj} sessions:", sessions)
        for sess in sessions:
            keys = list(features_dict[subj][sess].keys())
            print(f"  Session '{sess}' keys:", keys)
    print("Inspection complete.\n")

def evaluate_method(X, y, n_splits=5):
    clf = SVC(kernel='linear', C=1, random_state=42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, kappas = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        accuracies.append(accuracy_score(y[test_idx], y_pred))
        kappas.append(cohen_kappa_score(y[test_idx], y_pred))
    return np.mean(accuracies), np.mean(kappas)

def evaluate_method_for_subject(subject_features, method_name, n_splits=5):
    """
    Evaluates a feature extraction method for one subject by
    processing each session separately and then averaging the results.
    
    Parameters:
      subject_features : dict
          Dictionary for a subject, e.g., {'0train': {method1: array, 'labels': array}, '1test': {...}}
      method_name : str
          The name of the feature extraction method to evaluate.
      n_splits : int
          Number of splits for cross-validation.
          
    Returns:
      mean_acc : float
          Mean accuracy across sessions.
      mean_kappa : float
          Mean Cohen's kappa across sessions.
    """
    session_results = []
    for sess, data in subject_features.items():
        if method_name in data and 'labels' in data:
            X = data[method_name]
            y = data['labels']
            # Evaluate for this session
            acc, kappa = evaluate_method(X, y, n_splits=n_splits)
            print(f"  Session '{sess}': Accuracy = {acc:.2f}, Kappa = {kappa:.2f}")
            session_results.append((acc, kappa))
        else:
            print(f"Method {method_name} not found for this session: {sess}")
    if not session_results:
        raise ValueError(f"No features or labels found for method {method_name} in this subject.")
    # Average results across sessions for this subject
    mean_acc = np.mean([r[0] for r in session_results])
    mean_kappa = np.mean([r[1] for r in session_results])
    return mean_acc, mean_kappa

if __name__ == "__main__":
    # Change the path to your saved features file.
    # For example, if you ran riemannian + PCA, it might be:
    features_file = "outputs/2025-03-12/fbcsp_pca/fbcsp_pca_features.pkl"
    features_dict = load_features(features_file)
    
    inspect_features(features_dict)
    
    method = 'fbcsp'  # Change to the method you wish to evaluate
    all_subject_acc = []
    all_subject_kappa = []
    
    for subj in sorted(features_dict.keys()):
        try:
            print(f"Evaluating subject {subj} for method {method}:")
            acc, kappa = evaluate_method_for_subject(features_dict[subj], method, n_splits=5)
            print(f"Subject {subj} - Mean Accuracy: {acc:.2f}, Mean Kappa: {kappa:.2f}\n")
            all_subject_acc.append(acc)
            all_subject_kappa.append(kappa)
        except Exception as e:
            print(f"Error for subject {subj}: {e}")
    
    if all_subject_acc:
        print("Overall Mean Accuracy:", np.mean(all_subject_acc))
        print("Overall Mean Cohen's Kappa:", np.mean(all_subject_kappa))




    