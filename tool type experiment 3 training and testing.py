import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# LOAD THE DATA
# ============================================================

X = np.load("X.npy")   # shape (N, features)
y = np.load("y.npy")   # shape (N,)

num_classes = len(np.unique(y))
input_dim = X.shape[1]

print("[INFO] Loaded dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", num_classes)

# ============================================================
# CNN MODEL
# ============================================================

def build_cnn(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(32, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
# TRAIN–TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ============================================================
# CNN TRAINING
# ============================================================

print("\n================ CNN TRAINING ================")

X_train_cnn = X_train.reshape((X_train.shape[0], input_dim, 1))
X_test_cnn  = X_test.reshape((X_test.shape[0], input_dim, 1))

cnn = build_cnn(input_dim, num_classes)

cnn.fit(
    X_train_cnn, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

cnn_pred = np.argmax(cnn.predict(X_test_cnn), axis=1)
cnn_acc = accuracy_score(y_test, cnn_pred)

print("\n[CNN RESULTS]")
print("Accuracy:", cnn_acc)
print(classification_report(y_test, cnn_pred))


# ============================================================
# SVM (Linear & RBF)
# ============================================================

print("\n================ SVM TRAINING ================")

scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm  = scaler.transform(X_test)

# ---- Linear ----
svm_linear = SVC(kernel="linear", C=5)
svm_linear.fit(X_train_svm, y_train)
pred_lin = svm_linear.predict(X_test_svm)
acc_lin = accuracy_score(y_test, pred_lin)

print("\n[SVM LINEAR]")
print("Accuracy:", acc_lin)
print(classification_report(y_test, pred_lin))

# ---- RBF ----
svm_rbf = SVC(kernel="rbf", C=5, gamma=0.001)
svm_rbf.fit(X_train_svm, y_train)
pred_rbf = svm_rbf.predict(X_test_svm)
acc_rbf = accuracy_score(y_test, pred_rbf)

print("\n[SVM RBF]")
print("Accuracy:", acc_rbf)
print(classification_report(y_test, pred_rbf))


# ============================================================
# XGBOOST TRAINING
# ============================================================

print("\n================ XGBOOST TRAINING ================")

xgb = XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=300,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss"
)

xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, pred_xgb)

print("\n[XGBOOST RESULTS]")
print("Accuracy:", acc_xgb)
print(classification_report(y_test, pred_xgb))
