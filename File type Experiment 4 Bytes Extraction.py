import os
import numpy as np
from collections import Counter

# ============================================================
# EXTRACT BYTE HISTOGRAM + ENTROPY FROM A SINGLE FILE
# ============================================================

def extract_byte_features(file_path):
    """
    Extracts:
      - Raw bytes
      - 256-bin normalized histogram
      - Shannon entropy
    Returns: 257-dim feature vector
    """
    # Read file
    with open(file_path, "rb") as f:
        data = f.read()

    # Empty file → return zeros
    if len(data) == 0:
        histogram = np.zeros(256, dtype=np.float32)
        entropy = 0.0
        return np.append(histogram, entropy)

    # Convert raw bytes to uint8 array
    byte_arr = np.frombuffer(data, dtype=np.uint8)

    # Compute histogram
    counts = Counter(byte_arr)
    hist = np.array([counts.get(i, 0) for i in range(256)], dtype=np.float32)

    total = hist.sum()
    hist_norm = hist / total

    # Shannon entropy
    p = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(p * np.log2(p)))

    # Final feature vector (256 bins + entropy)
    feature_vector = np.append(hist_norm, entropy)
    return feature_vector


# ============================================================
# 
# ============================================================

def load_histogram_entropy_dataset(root_dir):
    features = []
    labels = []
    class_map = {}
    class_id = 0

    # Loop through class folders
    for folder in sorted(os.listdir(root_dir)):
        class_folder = os.path.join(root_dir, folder)
        if not os.path.isdir(class_folder):
            continue

        # Assign label
        class_map[folder] = class_id
        print(f"[INFO] Loading class: {folder}  →  {class_id}")

        # Loop through files in class
        for fname in os.listdir(class_folder):
            fpath = os.path.join(class_folder, fname)

            if not os.path.isfile(fpath):
                continue

            try:
                fv = extract_byte_features(fpath)
                features.append(fv)
                labels.append(class_id)
            except Exception as e:
                print(f"[ERROR] Skipping {fpath}: {e}")

        class_id += 1

    # Convert to numpy arrays
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print("\n[INFO] Dataset loaded")
    print("X shape:", X.shape)      # (N, 257)
    print("y shape:", y.shape)      # (N,)
    print("Classes:", class_map)

    return X, y, class_map


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(X, y, out_prefix="byte_features"):
    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_y.npy", y)
    print(f"[DONE] Saved {out_prefix}_X.npy and {out_prefix}_y.npy")


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":
    DATASET_PATH = "C:\Users\moizz\Downloads\File Type Indentification"   # folder containing 15 class folders

    # Step 1 — Load and extract features
    X, y, class_map = load_histogram_entropy_dataset(DATASET_PATH)

    # Step 2 — Save dataset
    save_dataset(X, y, out_prefix="hist_entropy")

    # Step 3 — Save class mapping
    import json
    with open("class_map.json", "w") as f:
        json.dump(class_map, f, indent=4)

    print("\n[COMPLETE] Histogram + Entropy dataset ready.")
