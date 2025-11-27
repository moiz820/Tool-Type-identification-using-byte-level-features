import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------
# Load Data (15 classes)
# --------------------------------------------------
X = np.load("features.npy")
y = np.load("labels.npy")

# --------------------------------------------------
# Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================================================
# =============== 1) CNN MODEL =====================
# ==================================================

# Reshape for Conv1D
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

cnn = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(15, activation='softmax')
])

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn.fit(
    X_train_cnn, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

cnn_pred = np.argmax(cnn.predict(X_test_cnn), axis=1)

print("\n================ CNN RESULTS ================")
print("Accuracy:", accuracy_score(y_test, cnn_pred))
print(classification_report(y_test, cnn_pred))

# ==================================================
# =============== 2) SVM MODELS ====================
# ==================================================

# Standardization
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

C_val = 5
gamma_val = 0.001

# ---- RBF ----
svm_rbf = SVC(C=C_val, gamma=gamma_val, kernel='rbf')
svm_rbf.fit(X_train_svm, y_train)
pred_rbf = svm_rbf.predict(X_test_svm)

print("\n============= SVM (RBF) RESULTS =============")
print("Accuracy:", accuracy_score(y_test, pred_rbf))
print(classification_report(y_test, pred_rbf))

# ---- LINEAR ----
svm_linear = SVC(C=C_val, kernel='linear')
svm_linear.fit(X_train_svm, y_train)
pred_linear = svm_linear.predict(X_test_svm)

print("\n=========== SVM (LINEAR) RESULTS ============")
print("Accuracy:", accuracy_score(y_test, pred_linear))
print(classification_report(y_test, pred_linear))

# ---- POLYNOMIAL ----
svm_poly = SVC(C=C_val, gamma=gamma_val, kernel='poly', degree=3)
svm_poly.fit(X_train_svm, y_train)
pred_poly = svm_poly.predict(X_test_svm)

print("\n=========== SVM (POLY) RESULTS ==============")
print("Accuracy:", accuracy_score(y_test, pred_poly))
print(classification_report(y_test, pred_poly))

# ---- SIGMOID ----
svm_sigmoid = SVC(C=C_val, gamma=gamma_val, kernel='sigmoid')
svm_sigmoid.fit(X_train_svm, y_train)
pred_sig = svm_sigmoid.predict(X_test_svm)

print("\n=========== SVM (SIGMOID) RESULTS ===========")
print("Accuracy:", accuracy_score(y_test, pred_sig))
print(classification_report(y_test, pred_sig))

# ==================================================
# =============== 3) XGBOOST MODEL =================
# ==================================================

xgb = XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=300,
    objective='multi:softmax',
    num_class=15,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='mlogloss'
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("\n================ XGBOOST RESULTS ================")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))
