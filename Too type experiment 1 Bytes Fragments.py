import os
import numpy as np
from collections import Counter
import json
=================

def extract_histogram(byte_arr):
    """
    Takes a uint8 array and returns:
        - normalized 256-bin histogram
    Output shape: (256,)
    """
    if len(byte_arr) == 0:
        return np.zeros(256, dtype=np.float32)

    counts = Counter(byte_arr)
    hist = np.array([counts.get(i, 0) for i in range(256)], dtype=np.float32)
    hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
    return hist_norm


# ============================================================
# 2. Extract byte fragments for all fragment sizes
# ============================================================

FRAGMENT_SIZES = [64, 128, 1024, 2048, 4096]

def extract_all_fragments(file_path):
    """
    For each fragment size:
        - Reads first N bytes
        - Pads if needed
        - Extracts histogram ONLY
    Returns: dict {size: (256-dim vector)}
    """
    with open(file_path, "rb") as f:
        raw = f.read()

    results = {}

    for size in FRAGMENT_SIZES:
        if len(raw) < size:
            fragment = raw + bytes(size - len(raw))
        else:
            fragment = raw[:size]

        arr = np.frombuffer(fragment, dtype=np.uint8)
        fv = extract_histogram(arr)   # NO entropy
        results[size] = fv

    return results


# ============================================================
# 3. Load dataset with 15 classes
# ============================================================

def load_fragment_dataset(root_dir):
    """
    Output:
        X64, X128, X1024, X2048, X4096, y, class_map
    """
    X64, X128, X1024, X2048, X4096 = [], [], [], [], []
    labels = []
    class_map = {}
    label_id = 0

    for folder in sorted(os.listdir(root_dir)):
        class_folder = os.path.join(root_dir, folder)
        if not os.path.isdir(class_folder):
            continue

        class_map[folder] = label_id
        print(f"[INFO] Class: {folder} → {label_id}")

        for fname in os.listdir(class_folder):
            fpath = os.path.join(class_folder, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                frags = extract_all_fragments(fpath)

                X64.append(frags[64])
                X128.append(frags[128])
                X1024.append(frags[1024])
                X2048.append(frags[2048])
                X4096.append(frags[4096])

                labels.append(label_id)

            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")

        label_id += 1

    return (
        np.array(X64, dtype=np.float32),
        np.array(X128, dtype=np.float32),
        np.array(X1024, dtype=np.float32),
        np.array(X2048, dtype=np.float32),
        np.array(X4096, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        class_map
    )


# ============================================================
# 4. Save datasets
# ============================================================

def save_all(
    X64, X128, X1024, X2048, X4096, y, class_map,
    prefix="frag"
):
    np.save(f"{prefix}_64.npy", X64)
    np.save(f"{prefix}_128.npy", X128)
    np.save(f"{prefix}_1024.npy", X1024)
    np.save(f"{prefix}_2048.npy", X2048)
    np.save(f"{prefix}_4096.npy", X4096)
    np.save(f"{prefix}_y.npy", y)

    with open(f"{prefix}_class_map.json", "w") as f:
        json.dump(class_map, f, indent=4)

    print("[DONE] All fragment-based histogram datasets saved.")


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

if __name__ == "__main__":
    DATASET = r"C:\Users\moizz\Downloads\File Type Indentification"

    X64, X128, X1024, X2048, X4096, y, cmap = load_fragment_dataset(DATASET)

    print("Shapes:")
    print("64 bytes   →", X64.shape)
    print("128 bytes  →", X128.shape)
    print("1024 bytes →", X1024.shape)
    print("2048 bytes →", X2048.shape)
    print("4096 bytes →", X4096.shape)
    print("Labels →", y.shape)

    save_all(X64, X128, X1024, X2048, X4096, y, cmap)
