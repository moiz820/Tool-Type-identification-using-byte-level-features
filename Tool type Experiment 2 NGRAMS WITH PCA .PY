import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json

import tensorflow as tf
from tensorflow.keras import layers, models

# ============================
# CONFIG
# ============================

DATASET_ROOT = ""C:\Users\moizz\Downloads\Tool Type Indentification""   # root folder with class subfolders
FRAGMENT_SIZES = [10000, 20000, 50000, 70000]
NGRAMS = [1, 2, 3, 4]

HASH_VECTOR_SIZE = 4096     # feature dim for hashed n-gram
PCA_COMPONENTS = 256        # PCA output dim

CNN_EPOCHS = 100
CNN_BATCH_SIZE = 64
CNN_LR = 0.0005
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================
# UTILS
# ============================

def read_fragment(file_path, fragment_size):
    """Read first fragment_size bytes, pad with zeros if needed."""
    with open(file_path, "rb") as f:
        data = f.read()

    if len(data) < fragment_size:
        data = data + bytes(fragment_size - len(data))
    else:
        data = data[:fragment_size]

    return np.frombuffer(data, dtype=np.uint8)


def hashed_ngram_vector(byte_arr, n, vec_size=HASH_VECTOR_SIZE):
    """
    Create a hashed n-gram frequency vector of fixed length vec_size.
    - byte_arr: uint8 array
    - n: n-gram length (1,2,3,4)
    """
    vec = np.zeros(vec_size, dtype=np.float32)
    L = len(byte_arr)

    if L < n:
        return vec

    for i in range(L - n + 1):
        gram = tuple(byte_arr[i:i+n])
        h = hash(gram) % vec_size
        vec[h] += 1.0

    # normalize
    total = vec.sum()
    if total > 0:
        vec /= total

    return vec


def build_dataset_for_combo(root_dir, fragment_size, ngram):
    """
    Build X, y for given fragment_size and ngram.
    Returns:
        X: (N, HASH_VECTOR_SIZE)
        y: (N,)
        class_map: dict name -> label
    """
    X = []
    y = []
    class_map = {}
    label_id = 0

    for folder in sorted(os.listdir(root_dir)):
        class_folder = os.path.join(root_dir, folder)
        if not os.path.isdir(class_folder):
            continue

        class_map[folder] = label_id
        print(f"[INFO] frag={fragment_size}, ngram={ngram}: Class {folder} -> {label_id}")

        for fname in os.listdir(class_folder):
            fpath = os.path.join(class_folder, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                frag_bytes = read_fragment(fpath, fragment_size)
                feature_vec = hashed_ngram_vector(frag_bytes, ngram, vec_size=HASH_VECTOR_SIZE)
                X.append(feature_vec)
                y.append(label_id)
            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")

        label_id += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[INFO] Built dataset: frag={fragment_size}, ngram={ngram}, X shape={X.shape}, y shape={y.shape}")
    return X, y, class_map


def build_cnn_model(input_length):
    """
     PCA features (treated as a 1D sequence).
    """
    model = models.Sequential([
        layers.Input(shape=(input_length, 1)),
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(15, activation='softmax')  # 15 classes
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=CNN_LR)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ============================
# MAIN LOOP: all fragment sizes × n-grams
# ============================

if __name__ == "__main__":
    results = {}   # store accuracy per (frag, ngram)

    for frag_size in FRAGMENT_SIZES:
        for ngram in NGRAMS:
            print("\n" + "="*70)
            print(f"[RUN] Fragment size = {frag_size} bytes | n-gram = {ngram}")
            print("="*70)

            # 1) Build base feature dataset
            X, y, class_map = build_dataset_for_combo(DATASET_ROOT, frag_size, ngram)

            # 2) Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )

            # 3) Standardize before PCA
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 4) PCA
            pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            # 5) Reshape for CNN: (samples, timesteps, channels)
            X_train_cnn = X_train_pca.reshape((X_train_pca.shape[0], X_train_pca.shape[1], 1))
            X_test_cnn = X_test_pca.reshape((X_test_pca.shape[0], X_test_pca.shape[1], 1))

            # 6) Build & train CNN
            model = build_cnn_model(input_length=PCA_COMPONENTS)

            model.fit(
                X_train_cnn, y_train,
                epochs=CNN_EPOCHS,
                batch_size=CNN_BATCH_SIZE,
                validation_split=0.1,
                verbose=1
            )

            # 7) Evaluate
            y_pred_probs = model.predict(X_test_cnn)
            y_pred = np.argmax(y_pred_probs, axis=1)

            acc = accuracy_score(y_test, y_pred)
            print(f"\n[RESULT] frag={frag_size}, ngram={ngram}, Accuracy = {acc:.4f}")
            print(classification_report(y_test, y_pred))

            # store result
            results[(frag_size, ngram)] = float(acc)

            #  save model
            model_name = f"cnn_frag{frag_size}_ngram{ngram}.keras"
            model.save(model_name)
            print(f"[INFO] Saved model: {model_name}")

    # Save results summary
    res_dict = {f"{frag}_{ng}": acc for (frag, ng), acc in results.items()}
    with open("cnn_pca_ngram_results.json", "w") as f:
        json.dump(res_dict, f, indent=4)

    print("\n[SUMMARY] Accuracies per (fragment_size, ngram):")
    for key, v in res_dict.items():
        print(f"{key}: {v:.4f}")

