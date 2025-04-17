import pickle

with open("./dump/features.pkl", "rb") as f:
    features = pickle.load(f)

for subject, sessions in features.items():
    for sess, feat_dict in sessions.items():
        X = feat_dict["combined"]
        y = feat_dict["labels"]
        print(f"Subject {subject}, session {sess}:")
        print(f"  • feature matrix shape: {X.shape}  (trials×dims)")
        print(f"  • labels length      : {len(y)}")

import pickle

with open("./dump/features.pkl","rb") as f:
    feats = pickle.load(f)

for subj, sessions in feats.items():
    for sess, d in sessions.items():
        X, y = d['combined'], d['labels']
        assert X.shape[0] == len(y), "Mismatch in trials vs labels!"
        print(f"{subj}/{sess}: {X.shape[1]} features, {len(y)} trials")
