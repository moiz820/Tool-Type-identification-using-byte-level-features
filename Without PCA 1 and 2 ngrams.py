import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
import json

# =====================================================================
# CONFIG
# =====================================================================

DATASET_ROOT = ""C:\Users\moizz\Downloads\File Type Indentification"'"
NGRAMS = [1, 2]
HASH_VECTOR_SIZE = 4096

TEST_SIZE = 0.2
RANDOM_STATE = 42

CNN_LR = 0.0005
CNN_BATCH = 64
CNN_EPOCHS = 50   # 100 possible but long

# =====================================================================
# BYTE READING
# =====================================================================

def read_full_file(file_path):
    """Read full file as uint8 array."""
    with open(file_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8)

# =====================================================================
# HASHED N-GRAM FEATURE VECTOR
# =====================================================================

def hashed_ngram_vector(byte_arr, n, vec_size=HASH_VECTOR_SIZE):
    vec = np.zeros(vec_size, dtype=np.float32)
    L = len(byte_arr)

    if L < n:
        return vec

    for i in range(L - n + 1):
        gram = tuple(byte_arr[i:i+n])
        h = hash(gram) % vec_size
        vec[h] += 1.0

    total = vec.sum()
    if total > 0:
        vec /= total

    return vec

# =====================================================================
# DATASET BUILDER (1-gram or 2-gram on FULL FILE)
# =====================================================================

def build_ngram_dataset(root_dir, ngram):
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
            fpath = os.path.join(class_folder, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                bytes_arr = read_full_file(fpath)
                vec = hashed_ngram_vector(bytes_arr, ngram)

                X.append(vec)
                y.append(label_id)

            except Exception as e:
                print(f"[ERROR] Failed on {fpath}: {e}")

        label_id += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[DONE] n={ngram} → X={X.shape}, y={y.shape}")
    return X, y, class_map

# =====================================================================
# CNN MODEL (1D)
# =====================================================================

def build_cnn(input_len):
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(15, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(CNN_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =====================================================================
# MAIN EXECUTION
# =====================================================================

results = {}

for ngram in NGRAMS:
    print("\n" + "="*70)
    print(f"     RUNNING EXPERIMENT → NGRAM = {ngram}")
    print("="*70)

    # ---------------------------
    # 1) Dataset
    # ---------------------------
    X, y, class_map = build_ngram_dataset(DATASET_ROOT, ngram)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ---------------------------
    # 2) CNN
    # ---------------------------
    print("\n[TRAIN] CNN Model")

    X_train_cnn = X_train.reshape((X_train.shape[0], HASH_VECTOR_SIZE, 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], HASH_VECTOR_SIZE, 1))

    cnn = build_cnn(HASH_VECTOR_SIZE)

    cnn.fit(
        X_train_cnn, y_train,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH,
        validation_split=0.1,
        verbose=1
    )

    pred_cnn = np.argmax(cnn.predict(X_test_cnn), axis=1)

    acc_cnn = accuracy_score(y_test, pred_cnn)
    print(f"[CNN ACC] n={ngram}: {acc_cnn:.4f}")

    results[(ngram, "cnn")] = acc_cnn

    # ---------------------------
    # 3) SVM LINEAR
    # ---------------------------
    print("\n[TRAIN] SVM Linear")

    scaler = StandardScaler()
    X_train_svm = scaler.fit_transform(X_train)
    X_test_svm = scaler.transform(X_test)

    svm_linear = SVC(kernel="linear", C=5)
    svm_linear.fit(X_train_svm, y_train)

    pred_lin = svm_linear.predict(X_test_svm)
    acc_lin = accuracy_score(y_test, pred_lin)

    print(f"[SVM-LINEAR ACC] n={ngram}: {acc_lin:.4f}")
    results[(ngram, "svm_linear")] = acc_lin

    # ---------------------------
    # 4) SVM RBF
    # ---------------------------
    print("\n[TRAIN] SVM RBF")

    svm_rbf = SVC(kernel="rbf", C=5, gamma=0.001)
    svm_rbf.fit(X_train_svm, y_train)

    pred_rbf = svm_rbf.predict(X_test_svm)
    acc_rbf = accuracy_score(y_test, pred_rbf)

    print(f"[SVM-RBF ACC] n={ngram}: {acc_rbf:.4f}")
    results[(ngram, "svm_rbf")] = acc_rbf

    # ---------------------------
    # 5) XGBoost
    # ---------------------------
    print("\n[TRAIN] XGBoost")

    xgb = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        objective="multi:softmax",
        num_class=15,
        eval_metric="mlogloss"
    )

    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, pred_xgb)

    print(f"[XGBOOST ACC] n={ngram}: {acc_xgb:.4f}")
    results[(ngram, "xgboost")] = acc_xgb

# Save Results


print("\n==== FINAL RESULTS ====")
for k, v in res.items():
    print(k, ":", v)
