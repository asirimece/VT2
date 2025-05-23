import pickle

with open("dump/preprocessed_data_custom.pkl", "rb") as f:
    preprocessed_data = pickle.load(f)

print("\n--- Data Inspection ---")
for subj, v in preprocessed_data.items():
    print(f"Subject {subj}: type={type(v)}")
    if isinstance(v, dict):
        print(f"  Keys: {list(v.keys())}")
    else:
        print(f"  Shape: {v.get_data().shape if hasattr(v, 'get_data') else 'N/A'}")
    # Only print the first 2 subjects for brevity
    if int(subj) > 2:
        break

"""
import pickle

with open('./dump/preprocessed_data_custom.pkl', 'rb') as f:
    data = pickle.load(f)

# Pick one subject, one split (e.g. 'train'), and check shape:
subj = list(data.keys())[0]
ep = data[subj]['train']  # or 'test'
print(ep.get_data().shape)  # (n_epochs, n_channels, n_times)
"""