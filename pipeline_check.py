import os
import pickle
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

def run_classical_pipeline(preprocessed_data_file):
    """
    Loads the preprocessed epochs data, then for each subject,
    applies a CSP+LDA pipeline to classify the trials.

    Parameters
    ----------
    preprocessed_data_file : str
        Path to the pickle file containing preprocessed data.

    Returns
    -------
    results : dict
        A dictionary with per-subject accuracy and confusion matrix.
    """
    if not os.path.exists(preprocessed_data_file):
        raise FileNotFoundError(f"Preprocessed data file not found: {preprocessed_data_file}")
    
    with open(preprocessed_data_file, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    results = {}
    
    for subj, sessions in preprocessed_data.items():
        print(f"\n=== Subject {subj} ===")
        if "0train" not in sessions or "1test" not in sessions:
            print(f"Subject {subj}: Missing train or test session, skipping.")
            continue

        # Extract training and test epochs (assuming they are MNE Epochs objects)
        epochs_train = sessions["0train"]
        epochs_test = sessions["1test"]

        # Get the data arrays: shape (n_trials, n_channels, n_times)
        X_train = epochs_train.get_data()
        X_test  = epochs_test.get_data()
        # Convert to double precision (float64) to avoid precision issues in CSP
        X_train = X_train.astype(np.float64)
        X_test  = X_test.astype(np.float64)
        
        # Convert raw event labels (assumed to be 1-4) to 0-3
        y_train = epochs_train.events[:, -1]
        y_test  = epochs_test.events[:, -1]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print("Unique training labels (raw):", np.unique(epochs_train.events[:, -1]))
        print("Unique training labels (converted):", np.unique(y_train))
        
        # Initialize CSP; here we set n_components=4 (you can adjust this number)
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        try:
            csp.fit(X_train, y_train)
        except Exception as e:
            print("Error during CSP fitting:", e)
            continue
        
        # Transform the data to obtain features
        X_train_csp = csp.transform(X_train)
        X_test_csp  = csp.transform(X_test)
        
        # Train an LDA classifier on the CSP features
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_csp, y_train)
        y_pred = clf.predict(X_test_csp)
        
        # Evaluate the classifier
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"LDA accuracy for subject {subj}: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        results[subj] = {"accuracy": acc, "confusion_matrix": cm}
    
    return results

if __name__ == "__main__":
    # Adjust the path to your preprocessed data file as needed.
    preprocessed_data_file = os.path.join(os.getcwd(), "outputs", "preprocessed_data.pkl")
    print(f"Using preprocessed data file: {preprocessed_data_file}")
    
    results = run_classical_pipeline(preprocessed_data_file)
    
    print("\n=== Overall Results ===")
    for subj, res in results.items():
        print(f"Subject {subj}: Accuracy = {res['accuracy']:.4f}")
