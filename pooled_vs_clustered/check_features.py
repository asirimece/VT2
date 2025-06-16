import pickle

with open("dump/features.pkl", "rb") as f:
    features = pickle.load(f)

# Print one example
for subj_id, val in features.items():
    print("Subject ID:", subj_id)
    print("Type of value:", type(val))
    if isinstance(val, dict):
        for sess_name, sess_data in val.items():
            print(f"  Session {sess_name} keys: {list(sess_data.keys())}")
            print(f"  Shape of 'combined': {sess_data['combined'].shape}")
        break
    elif isinstance(val, np.ndarray):
        print(f"Shape of array: {val.shape}")
        break
    else:
        print("Unexpected format:", val)
        break
