import os
import numpy as np
from PIL import Image

# ================================================================
# 1) EXACT BYTE → IMAGE SHAPES
# ================================================================

def bytes_to_img(arr, height, width):
    """
    Takes a uint8 byte array and converts to (H,W) grayscale image.
    Pads or trims to exact H*W.
    """
    target_len = height * width

    if len(arr) < target_len:
        arr = np.pad(arr, (0, target_len - len(arr)), mode='constant')
    else:
        arr = arr[:target_len]

    return arr.reshape((height, width))


# ================================================================
#
# ================================================================
# 4096 bytes (64 × 64)
def make_4096_img(file_path):
    with open(file_path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    return bytes_to_img(arr, 64, 64)


# 10,000 bytes (100 × 100)
def make_10000_img(file_path):
    with open(file_path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    return bytes_to_img(arr, 100, 100)


# 20,000 bytes (200 × 100)
def make_20000_img(file_path):
    with open(file_path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    return bytes_to_img(arr, 200, 100)


# ================================================================
# 3) Load dataset and generate all 3 image types
# dataset
# ================================================================

def load_dataset(root_dir):
    X_4096 = []
    X_10000 = []
    X_20000 = []
    y = []
    class_map = {}
    label_id = 0

    for folder in sorted(os.listdir(root_dir)):
        class_folder = os.path.join(root_dir, folder)
        if not os.path.isdir(class_folder):
            continue

        class_map[folder] = label_id
        print(f"[INFO] Loading: {folder} → {label_id}")

        for fname in os.listdir(class_folder):
            fpath = os.path.join(class_folder, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                img1 = make_4096_img(fpath)
                img2 = make_10000_img(fpath)
                img3 = make_20000_img(fpath)

                X_4096.append(img1)
                X_10000.append(img2)
                X_20000.append(img3)
                y.append(label_id)
            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")

        label_id += 1

    return (
        np.array(X_4096, dtype=np.uint8),
        np.array(X_10000, dtype=np.uint8),
        np.array(X_20000, dtype=np.uint8),
        np.array(y),
        class_map
    )


# ================================================================
# 4) Save PNGs for manual inspection
# ================================================================

def save_images(images, labels, class_map, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    class_reverse = {v: k for k, v in class_map.items()}

    for i, img in enumerate(images):
        cname = class_reverse[labels[i]]
        folder = os.path.join(out_dir, cname)
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img).save(os.path.join(folder, f"{i}.png"))


# ================================================================
# 5) Main pipeline
# ================================================================

if __name__ == "__main__":
    DATASET = ""C:\Users\moizz\Downloads\Tool Type Indentification""   

    X4, X10, X20, y, cmap = load_dataset(DATASET)

    print("Shapes:")
    print("4096 →", X4.shape)     # (N, 64, 64)
    print("10,000 →", X10.shape)  # (N, 100, 100)
    print("20,000 →", X20.shape)  # (N, 200, 100)

    # save numpy arrays
    np.save("X_4096.npy", X4)
    np.save("X_10000.npy", X10)
    np.save("X_20000.npy", X20)
    np.save("y.npy", y)

    # : save images
    save_images(X4, y, cmap, "img_4096")
    save_images(X10, y, cmap, "img_10000")
    save_images(X20, y, cmap, "img_20000")

    print("\n[DONE] All datasets created.")

