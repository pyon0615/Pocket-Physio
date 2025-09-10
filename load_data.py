import os
import numpy as np
from pathlib import Path

def check_pairs(data_dir: str, exercise_filter: str = None):
    data_dir = Path(data_dir)
    angle_files = sorted(f for f in data_dir.iterdir() if f.name.endswith("_angles.npy"))

    valid_keys = []

    for angle_path in angle_files:
        key = angle_path.name.replace("_angles.npy", "")
        label_path = data_dir / f"{key}_label.npy"

        if not label_path.exists():
            print(f"❌ Missing label: {key}_label.npy")
            continue

        if exercise_filter and exercise_filter.lower() not in key.lower():
            continue

        valid_keys.append(key)

    if not valid_keys:
        print("❌ No valid (angle, label) pairs found.")
    else:
        print(f"✅ Found {len(valid_keys)} valid sample pairs.")
    return valid_keys


def load_data(data_dir: str, exercise_filter: str = None):
    data_dir = Path(data_dir)
    valid_keys = check_pairs(data_dir, exercise_filter)

    features, labels = [], []

    for key in valid_keys:
        angle_path = data_dir / f"{key}_angles.npy"
        label_path = data_dir / f"{key}_label.npy"

        try:
            feat_arr = np.load(angle_path, allow_pickle=True)
            label_arr = np.load(label_path, allow_pickle=True).flatten()

            if feat_arr.shape[0] != 60:
                print(f"⛔ Skipping {key}: unexpected shape {feat_arr.shape}")
                continue

            label = int(label_arr[0]) if label_arr.size > 0 else None
            if label is None:
                print(f"⚠️ Skipping {key}: empty label")
                continue

            features.append(feat_arr)
            labels.append(label)

        except Exception as e:
            print(f"🚫 Error loading {key}: {e}")
            continue

    if not features:
        raise RuntimeError("❌ No valid samples loaded.")

    X = np.stack(features)
    y = np.array(labels, dtype=int)
    print(f"\n📊 Loaded dataset:")
    print(f"   X shape = {X.shape}")
    print(f"   y shape = {y.shape}")
    print(f"   Label counts = {np.bincount(y)}")
    return X, y


if __name__ == "__main__":
    # Example: load all data
    X, y = load_data("processed_data")

    # Example: load only squat samples
    # X, y = load_data("processed_data", exercise_filter="squat")
