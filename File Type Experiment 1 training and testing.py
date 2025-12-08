
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models
===========================================================

FRAGMENT_SIZES = [64, 128, 1024, 2048, 4096]


# ============================================================
# HISTOGRAM 
# ============================================================

def hist256(arr):
    counts = np.bincount(arr, minlength=256)
    hist = counts.astype(np.float32)
    return hist / hist.sum()


# ============================================================
# BEGIN, MIDDLE, END WINDOWS
# ============================================================

def begin_window(data, size):
    if len(data) < size:
        data += bytes(size - len(data))
    return np.frombuffer(data[:size], dtype=np.uint8)

def middle_window(data, size):
    if len(data) < size:
        data += bytes(size - len(data))
    start = (len(data) - size) // 2
    return np.frombuffer(data[start:start+size], dtype=np.uint8)

def end_window(data, size):
    if len(data) < size:
        data = bytes(size - len(data)) + data
    return np.frombuffer(data[-size:], dtype=np.uint8)


# ============================================================
# LOAD DATASET FOR ONE WINDOW MODE
# ============================================================

def load_dataset_mode(root, mode_func, size):
    X = []
    y = []
    class_map = {}
    label = 0

    for folder in sorted(os.listdir(root)):
        fpath = os.path.join(root, folder)
        if not os.path.isdir(fpath):
            continue

        class_map[folder] = label

        for fname in os.listdir(fpath):
            fp = os.path.join(fpath, fname)
            if not os.path.isfile(fp):
                continue

            with open(fp, "rb") as f:
                raw = f.read()

            frag = mode_func(raw, size)
            fv = hist256(frag)
            X.append(fv)
            y.append(label)

        label += 1

    return np.array(X), np.array(y), class_map


# ============================================================
# CNN Definition
# ============================================================

def build_cnn():
    model = tf.keras.Sequential([
        layers.Input((256,1)),
        layers.Conv1D(32,3,activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64,3,activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(15,activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # CNN
    cnn = build_cnn()
    cnn.fit(
        X_train.reshape(-1,256,1), y_train,
        epochs=50, batch_size=64, verbose=0
    )
    pred_cnn = cnn.predict(X_test.reshape(-1,256,1))
    acc_cnn = accuracy_score(y_test, pred_cnn.argmax(1))

    # SVM
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    svm = SVC(C=5, gamma=0.001, kernel='rbf')
    svm.fit(X_train_s, y_train)
    acc_svm = accuracy_score(y_test, svm.predict(X_test_s))

    # XGBoost
    xgb = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        objective='multi:softmax',
        num_class=15,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train, y_train)
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test))

    return acc_cnn*100, acc_svm*100, acc_xgb*100


def run_all(dataset_path):
    results = {}

    for size in FRAGMENT_SIZES:
        print(f"\n========== SIZE {size} ==========")

        # Beginning window
        Xb, yb, _ = load_dataset_mode(dataset_path, begin_window, size)
        b_cnn, b_svm, b_xgb = evaluate_models(Xb, yb)

        # Middle window
        Xm, ym, _ = load_dataset_mode(dataset_path, middle_window, size)
        m_cnn, m_svm, m_xgb = evaluate_models(Xm, ym)

        # End window
        Xe, ye, _ = load_dataset_mode(dataset_path, end_window, size)
        e_cnn, e_svm, e_xgb = evaluate_models(Xe, ye)

        results[size] = {
            "CNN":  [b_cnn, m_cnn, e_cnn],
            "SVM":  [b_svm, m_svm, e_svm],
            "XGB":  [b_xgb, m_xgb, e_xgb],
        }

    return results

================================================

if __name__ == "__main__":
    DATASET = r"C:\Users\moizz\Downloads\File Type Indentification/Fragments"
    output = run_all(DATASET)
    print("\nFINAL RESULTS:\n", output)

