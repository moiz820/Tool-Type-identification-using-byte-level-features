import os
import numpy as np
from PIL import Image

# ============================================================
# 1) Remove header & trailer, extract 4096-byte content fragment
# ============================================================

def extract_content_fragment(arr, fragment_size=4096, skip_header=1024, skip_footer=1024):
    """
    Remove first 1024 bytes (header) and last 1024 bytes (trailer).
    Then extract a 4096-byte content-only segment.
    If file is smaller, pad with zeros.
    """
    length = len(arr)

    # If file is too small, fallback
    if length < skip_header + skip_footer:
        core = arr
    else:
        core = arr[skip_header : length - skip_footer]

    # Trim or pad to fragment_size
    if len(core) < fragment_size:
        core = np.pad(core, (0, fragment_size - len(core)), mode='constant')
    else:
        core = core[:fragment_size]

    return core


# ============================================================
# 2) Convert byte array → grayscale image (64×64)
# ============================================================

def bytes_to_img(arr, height=64, width=64):
    """
    Convert 4096 bytes → 64×64 grayscale image.
    """
    arr = arr[: height * width]
    if len(arr) < height * width:
        arr = np.pad(arr, (0, height * width - len(arr)), mode="constant")
    return arr.reshape((height, width))


# ============================================================
# 3) Create ONE 4096-byte image for a file (content only)
# ============================================================

def file_to_image_4096(file_path):
    with open(file_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    core = extract_content_fragment(raw, fragment_size=4096)
    img = bytes_to_img(core, 64, 64)
    return img


# ============================================================
# 4) Load full dataset (folder-per-class)
# ============================================================

def load_dataset(root_dir):
    X = []
    y = []
    class_map = {}
    label_id = 0

    for folder in sorted(os.listdir(root_dir)):
        class_folder = os.path.join(root_dir, folder)
        if not os.path.isdir(class_folder):
            continue

        class_map[folder] = label_id
        print(f"[INFO] Loading class: {folder} → {label_id}")

        for fname in os.listdir(class_folder):
            file_path = os.path.join(class_folder, fname)
            if not os.path.isfile(file_path):
                continue

            try:
                img = file_to_image_4096(file_path)
                X.append(img)
                y.append(label_id)

            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")

        label_id += 1

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int32)

    print("\nDataset loaded successfully:")
    print("X shape:", X.shape)       # (N, 64, 64)
    print("y shape:", y.shape)
    print("Classes:", class_map)

    return X, y, class_map


# ============================================================
# 5) Save PNG images for inspection
# ============================================================

def save_images(X, y, class_map, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    reverse_map = {v: k for k, v in class_map.items()}

    for idx, img in enumerate(X):
        cls = reverse_map[y[idx]]
        folder = os.path.join(out_dir, cls)
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img).save(os.path.join(folder, f"{idx}.png"))


# ============================================================
# 6) Main pipeline
# ============================================================

if __name__ == "__main__":
    DATASET_DIR = r"C:\Users\moizz\Downloads\File Type Indentification"

    X, y, cmap = load_dataset(DATASET_DIR)

    np.save("X_4096_fragment.npy", X)
    np.save("y_labels.npy", y)

    save_images(X, y, cmap, "img_4096_clean")

    print("\n[DONE] 4096-byte content-only dataset created successfully.")
