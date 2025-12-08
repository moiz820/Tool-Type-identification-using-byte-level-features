import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

==========

X = np.load("X_vfa.npy")      # shape (N, 256) 
y = np.load("y_vfa.npy")      # shape (N,)

num_classes = len(np.unique(y))
input_dim = X.shape[1]

print("Dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", num_classes)

# ============================================================
# CNN MODEL
# ============================================================

def build_cnn(input_dim, num_classes):
    model = tf.keras.Sequential([
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
# TRAIN–TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)


# ============================================================
# TRAIN CNN
# ============================================================

print("\n================ CNN TRAINING ================")

cnn = build_cnn(input_dim, num_classes)

cnn.fit(
    X_train.reshape(-1, input_dim, 1),
    y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

cnn_pred = cnn.predict(X_test.reshape(-1, input_dim, 1)).argmax(axis=1)
cnn_acc = accuracy_score(y_test, cnn_pred)

print("[CNN ACCURACY]", cnn_acc)
print(classification_report(y_test, cnn_pred))


# ============================================================
# PREPROCESS FOR SVM / XGBoost
# ============================================================

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# ============================================================
# SVM MODELS
# ============================================================

svm_models = {
    "svm_linear":   SVC(kernel="linear", C=5),
    "svm_rbf":      SVC(kernel="rbf", C=5, gamma=0.001),
    "svm_poly":     SVC(kernel="poly", C=5, degree=3, gamma="scale"),
    "svm_sigmoid":  SVC(kernel="sigmoid", C=5, gamma="scale"),
}

svm_results = {}

print("\n================ SVM TRAINING ================")

for name, clf in svm_models.items():
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    svm_results[name] = acc

    print(f"[{name.upper()} ACCURACY]", acc)
    print(classification_report(y_test, pred))


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
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print("[XGBOOST ACCURACY]", xgb_acc)
print(classification_report(y_test, xgb_pred))


# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n================ FINAL SUMMARY ================")
print("CNN:", cnn_acc)
print("SVM Linear:", svm_results["svm_linear"])
print("SVM RBF:", svm_results["svm_rbf"])
print("SVM Polynomial:", svm_results["svm_poly"])
print("SVM Sigmoid:", svm_results["svm_sigmoid"])
print("XGBoost:", xgb_acc)
