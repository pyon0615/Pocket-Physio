#!/usr/bin/env python3
# train_model2A.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === Load Preprocessed Data ===
X = np.load("pushup_X.npy")  # shape: (samples, timesteps, features)
y = np.load("pushup_y.npy")  # one-hot: [1,0]=Correct, [0,1]=Incorrect

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Model Definition ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # 2-class classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Training ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# === Evaluation ===
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# === Predictions ===
y_pred_probs  = model.predict(X_test)
y_true_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# === Classification Report ===
print("\n📋 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=["Correct", "Incorrect"]))

# === Confusion Matrix ===
cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Correct", "Incorrect"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix – Push-up Classifier (Model 2A)")
plt.show()

# === Evaluation Metrics ===
acc  = accuracy_score(y_true_labels, y_pred_labels)
prec = precision_score(y_true_labels, y_pred_labels)
rec  = recall_score(y_true_labels, y_pred_labels)
f1   = f1_score(y_true_labels, y_pred_labels)
auc  = roc_auc_score(y_true_labels, y_pred_probs[:,1])  # class 1 = Incorrect

print("\n📊 Detailed Evaluation Metrics:")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC AUC   : {auc:.4f}")

# === Save Model ===
model.save("model2A_pushup.keras")
print("💾 Model saved as model2A_pushup.keras")
