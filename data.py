import mirdata
import re
import mirdata
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import pandas as pd




PITCHES = "C C# D D# E F F# G G# A A# B".split()
CHORD_VOCAB = [f"{p}:maj" for p in PITCHES] + [f"{p}:min" for p in PITCHES] + ["N"]
LE = LabelEncoder()
LE.fit(CHORD_VOCAB)



def canonical_root(root):
    root = root.strip()
    flat2sharp = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"
    }
    if root in flat2sharp:
        return flat2sharp[root]
    return root

def chord2label(label):

    lab = label.strip()
    if lab.upper() in {"N", "NO_CHORD"}:
        return "N"
    root = re.split(r"[:/]", lab)[0]
    root = canonical_root(root)
    if root not in PITCHES:
        return "N"
    if re.search(r"\bmin\b", lab, flags=re.IGNORECASE):
        quality = "min"
    else:
        quality = "maj"
    return f"{root}:{quality}"



class BillboardChromaDataset(Dataset):
    def __init__(self, billboard, track_ids=None, label_encoder=LE):
        self.billboard = billboard
        self.track_ids = track_ids or self.billboard.track_ids
        self.label_encoder = label_encoder
        assert self.label_encoder is not None, "You must provide a fitted LabelEncoder!"

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        try:
            tid = self.track_ids[idx]
            track = self.billboard.track(tid)

            # Load chroma features: shape (n_frames, 24)
            chroma = pd.read_csv(track.bothchroma_path, header=None).values
            times = chroma[:, 1].astype(np.float32)
            chroma = chroma[:, 2:].astype(np.float32)

            # Convert chord labels to frame-wise label list
            n_frames = chroma.shape[0]
            y_str = self.get_frame_labels(track, n_frames, times)
            # Transform chord labels to index
            y_idx = self.label_encoder.transform(y_str)

            return torch.tensor(chroma), torch.tensor(y_idx, dtype=torch.long), tid
        except ValueError:
            return None

    def get_frame_labels(self, track, n_frames, times):
        labels = np.array(["N"] * n_frames, dtype=object)
        if track.chords_majmin is None:
            raise ValueError(f"track {track.track_id} has no chord annotations.")
        for (start, end), lab in zip(track.chords_majmin.intervals, track.chords_majmin.labels):
            simp = chord2label(lab)
            mask = (times >= start) & (times < end)
            labels[mask] = simp
        return labels
    


def split_dataset(billboard, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    np.random.seed(seed)
    track_ids = billboard.track_ids.copy()
    np.random.shuffle(track_ids)
    n = len(track_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    train_ids = track_ids[:n_train]
    val_ids = track_ids[n_train:n_train + n_val]
    test_ids = track_ids[n_train + n_val:n_train + n_val + n_test]
    trainset = BillboardChromaDataset(billboard, track_ids=train_ids)
    valset = BillboardChromaDataset(billboard, track_ids=val_ids)
    testset = BillboardChromaDataset(billboard, track_ids=test_ids)
    

    return trainset, valset, testset


if __name__ == "__main__":
    billboard = mirdata.initialize('billboard')
    billboard.download()
    billboard.validate()
    track = billboard.choice_track()
    print(track)
    print(len(billboard.track_ids))



    ds = BillboardChromaDataset(billboard, track_ids=[track.track_id])
    X, y_idx, tid = ds[0]
    print("X:", X.shape)
    print("y shape:", tuple(y_idx.shape), " sample 20:", y_idx[1020:1040].tolist())

