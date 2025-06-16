import pickle

path = "/home/ubuntu/VT2/dump/preprocessed_data_custom.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print("Outer type:", type(data))     # should be dict
print("Subjects available:", list(data.keys())[:5], "...")

# Inspect a single subject’s entry, e.g. subject "300"
subj_entry = data["300"]
print("Type of data['300']:", type(subj_entry))

if isinstance(subj_entry, dict):
    print("  Keys inside subject 300 entry:", subj_entry.keys())
    # Now inspect one of those keys, for example:
    for k in subj_entry:
        print(f"    Key '{k}' → type:", type(subj_entry[k]))
else:
    print("  Entry is not a dict but:", type(subj_entry))
