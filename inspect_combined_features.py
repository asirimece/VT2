#!/usr/bin/env python
import pickle
import numpy as np

def load_features(features_file):
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

def inspect_features(features_dict):
    print("Inspecting features dictionary...")
    subjects = list(features_dict.keys())
    print("Subjects available:", subjects)
    
    for subj in subjects:
        print(f"\nSubject {subj}:")
        sessions = features_dict[subj]
        print("  Sessions:", list(sessions.keys()))
        for sess, data in sessions.items():
            print(f"  Session '{sess}':")
            keys = list(data.keys())
            print("    Keys:", keys)
            if 'combined' in data:
                combined = data['combined']
                labels = data.get('labels', None)
                print("    Combined features shape:", combined.shape)
                if labels is not None:
                    print("    Labels shape:", np.array(labels).shape)
                else:
                    print("    No labels found.")
    print("\nInspection complete.")

if __name__ == "__main__":
    # Change this path to the location of your features pickle file.
    features_file = "outputs/22ica/22ica_features.pkl"
    features_dict = load_features(features_file)
    inspect_features(features_dict)
