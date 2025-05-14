from tensorflow.keras.models import load_model
import numpy as np
import os

# Load model
model = load_model("lstm_model.h5")

# Load sample
x = np.load("processed_data/correct_1.npy")  # Example file
x = np.expand_dims(x, axis=0)  # Add batch dimension: (1, 60, 99)

# Predict
y_pred = model.predict(x)
print("Prediction (softmax):", y_pred)
print("Predicted class index:", np.argmax(y_pred))
