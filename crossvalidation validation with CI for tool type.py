import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# (X=257 features, y=labels 0-4 for 5 classes)
# ============================================================

X = np.load("X.npy")      # shape (N, 257)
y = np.load("y.npy")      # shape (N,)
X = X.astype(np.float32)

# ============================================================
# CONFIDENCE INTERVAL FUNCTION
# ============================================================

def confidence_interval(acc_list):
    acc_list = np.array(acc_list)
    mean = acc_list.mean()
    sd = acc_list.std()
    ci_low = mean - 1.96 * (sd / np.sqrt(len(acc_list)))
    ci_high = mean + 1.96 * (sd / np.sqrt(len(acc_list)))
    return mean, sd, ci_low, ci_high

# ============================================================
# CNN MODEL BUILDER
# ============================================================

def build_cnn(input_len=257):
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(5, activation='softmax')   # 5 classes
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================
# DEFINE MODELS (SVM + XGBoost)
# ============================================================

models_svm = {
    "svm_linear":   SVC(kernel="linear", C=5),
    "svm_poly":     SVC(kernel="poly", degree=3, C=5, gamma="scale"),
    "svm_sigmoid":  SVC(kernel="sigmoid", C=5, gamma="scale"),
}

xgb = XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=300,
    objective="multi:softmax",
    num_class=5,
    eval_metric="mlogloss"
)

# ============================================================
# 10-FOLD CROSS VALIDATION
# ============================================================

kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = {
    "cnn": [],
    "svm_linear": [],
    "svm_poly": [],
    "svm_sigmoid": [],
    "xgboost": []
}

fold_idx = 1

for train_idx, test_idx in kf.split(X):

    print(f"\n================= FOLD {fold_idx} =================")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------
    print("[CNN] Training...")

    X_train_cnn = X_train.reshape((-1, 257, 1))
    X_test_cnn  = X_test.reshape((-1, 257, 1))

    model_cnn = build_cnn()

    model_cnn.fit(
        X_train_cnn, y_train,
        epochs=40,
        batch_size=64,
        verbose=0
    )

    pred_cnn = np.argmax(model_cnn.predict(X_test_cnn), axis=1)
    acc_cnn = accuracy_score(y_test, pred_cnn)
    results["cnn"].append(acc_cnn)

    print(f"[CNN Accuracy] {acc_cnn:.4f}")

    # --------------------------------------------------------
    # Standardize for SVM and XGBoost
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --------------------------------------------------------
    # SVM MODELS
    # --------------------------------------------------------
    for name, clf in models_svm.items():
        print(f"[{name}] Training...")
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, pred)
        results[name].append(acc)
        print(f"[{name} Accuracy] {acc:.4f}")

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------
    print("[XGBoost] Training...")
    xgb.fit(X_train, y_train)
    pred_x = xgb.predict(X_test)
    acc_x = accuracy_score(y_test, pred_x)
    results["xgboost"].append(acc_x)
    print(f"[XGBoost Accuracy] {acc_x:.4f}")

    fold_idx += 1


# ============================================================
# FINAL RESULTS WITH CONFIDENCE INTERVALS
# ============================================================

print("\n==================== FINAL 10-FOLD RESULTS ====================\n")

for model_name, accs in results.items():
    mean, sd, ci_low, ci_high = confidence_interval(accs)
    print(f"{model_name.upper()}:")
    print("Fold Accuracies:", np.round(accs, 4))
    print(f"Mean Accuracy: {mean:.4f}")
    print(f"Std Dev: {sd:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n")
