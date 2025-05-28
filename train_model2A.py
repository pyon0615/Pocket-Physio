#!/usr/bin/env python3
# train_model2A.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# === Load Data ===
X = np.load("pushup_X.npy")  # shape: (samples, timesteps, features)
y = np.load("pushup_y.npy")  # shape: (samples, 2) one-hot encoded: [1, 0] = Correct, [0, 1] = Incorrect

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Build LSTM Model ===
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # 2 output classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train Model ===
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# === Save Model ===
model.save("model2A_pushup.h5")

# === Evaluate and Plot Confusion Matrix ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Correct", "Incorrect"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Greens, ax=ax)
plt.title("Confusion Matrix: Push-up Correct vs Incorrect", fontsize=14)
plt.tight_layout()
plt.savefig("model2A_confusion_matrix.png")
plt.show()
