import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# Load Data
# =========================
def load_dataset(data_folder='processed_data', label_folder='labels'):
    data = []
    labels = []

    for file in os.listdir(data_folder):
        if file.endswith('.npy'):
            x = np.load(os.path.join(data_folder, file))
            label_file = os.path.join(label_folder, file.replace('.npy', '.txt'))

            if not os.path.exists(label_file):
                continue

            with open(label_file, 'r') as f:
                label = eval(f.read().strip())

            data.append(x)
            labels.append(label)

    return np.array(data), np.array(labels)

# =========================
# Build LSTM Model
# =========================
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# =========================
# Training Setup
# =========================
if __name__ == "__main__":
    # Load data
    X, y = load_dataset()
    print("Loaded:", X.shape, y.shape)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Build model
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), num_classes=y.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test))

    # Save model
    model.save("lstm_model.h5")
    print("✅ Model saved as lstm_model.h5")
    

#To evaluate LSMT model 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

