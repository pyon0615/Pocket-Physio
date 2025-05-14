import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Form Feature Data ===
def load_form_dataset(data_folder='form_data'):
    data = []
    labels = []

    for file in os.listdir(data_folder):
        if file.endswith(".npy"):
            x = np.load(os.path.join(data_folder, file))
            label_file = os.path.join(data_folder, file.replace(".npy", ".txt"))

            if not os.path.exists(label_file):
                continue

            with open(label_file, 'r') as f:
                label = eval(f.read().strip())

            data.append(x)
            labels.append(label)

    return np.array(data), np.array(labels)

# === Build LSTM Model ===
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

# === Train Model ===
if __name__ == "__main__":
    # Load data
    X, y = load_form_dataset()
    print("Loaded:", X.shape, y.shape)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Build and compile model
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), num_classes=y.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test))

    # Save model
    model.save("form_model.h5")
    print("✅ Model saved as form_model.h5")

    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Evaluation
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred_classes))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Form Model Confusion Matrix")
    plt.show()
