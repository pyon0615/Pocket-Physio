import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load model ===
model = load_model("lstm_model.h5")

# === Load all test samples ===
data_folder = "processed_data"
label_folder = "labels"

X = []
y_true = []
filenames = []

for file in os.listdir(data_folder):
    if file.endswith(".npy"):
        x = np.load(os.path.join(data_folder, file))
        x = np.expand_dims(x, axis=0)

        label_file = os.path.join(label_folder, file.replace(".npy", ".txt"))
        if not os.path.exists(label_file):
            continue

        with open(label_file, 'r') as f:
            label = eval(f.read().strip())
        
        X.append(x)
        y_true.append(np.argmax(label))  # Convert one-hot to class index
        filenames.append(file)

X = np.vstack(X)  # (num_samples, 60, 99)

# === Predict ===
y_pred_prob = model.predict(X)
y_pred = np.argmax(y_pred_prob, axis=1)

# === Evaluation ===
print("✅ Accuracy:", accuracy_score(y_true, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
