#!/usr/bin/env python3
# train_model2B.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical

# ==== STEP 1: Load Data ====
X = np.load("squat_X.npy")  # shape: (samples, 60, features)
y = np.load("squat_y.npy")  # shape: (samples, 2)
print(f"✅ Loaded squat_X: {X.shape}, squat_y: {y.shape}")

# ==== STEP 2: Train-Test Split ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ==== STEP 3: Build LSTM Model ====
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # output: correct vs incorrect
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==== STEP 4: Compute Class Weights ====
y_train_labels = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights_dict = dict(enumerate(class_weights))
print("📊 Class Weights:", class_weights_dict)

# ==== STEP 5: Train ====
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    callbacks=[early_stop],
    class_weight=class_weights_dict,
    verbose=1
)

# ==== STEP 6: Evaluate ====
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {acc:.4f}")

# ==== STEP 7: Save Model ====
model.save("model2B_squat.h5")
print("✅ Model saved as model2B_squat.h5")

# ==== STEP 8: Optional Model Plot ====
try:
    plot_model(model, to_file="model2B_structure.png", show_shapes=True)
except:
    print("🛈 Skipped saving model structure plot (optional dependency missing)")

# ==== STEP 9: Confusion Matrix ====
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Correct", "Incorrect"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Greens, ax=ax)
plt.title("Confusion Matrix: Squat Correct vs Incorrect", fontsize=14)
plt.tight_layout()
plt.savefig("model2B_confusion_matrix.png")
plt.show()
