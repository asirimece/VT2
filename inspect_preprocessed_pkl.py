# inspect_preprocessed_data.py

import pickle

def main():
    # Update this path if needed.
    preprocessed_data_path = "/home/ubuntu/VT2/outputs/preprocessed_data.pkl"
    
    try:
        with open(preprocessed_data_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return

    print("Subjects found in the preprocessed data:")
    for subject in data.keys():
        print(f"\nSubject: {subject}")
        subject_data = data[subject]
        print("  Available keys:", list(subject_data.keys()))
        for epoch_key in subject_data.keys():
            try:
                epoch = subject_data[epoch_key]
                # Assuming the epoch object has a method get_data() and an attribute events.
                X = epoch.get_data()
                y = epoch.events[:, -1]
                print(f"  {epoch_key}: X shape = {X.shape}, y shape = {y.shape}")
            except Exception as e:
                print(f"  {epoch_key}: Could not extract details ({e})")
        print("-" * 40)

if __name__ == "__main__":
    main()
