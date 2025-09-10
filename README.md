# Pocket-Physio
Pocket Physio 🏋️‍♀️🤖
Machine learning approach to improving home-based physiotherapy for musculoskeletal disorders
📖 Overview
Pocket Physio is a machine learning project designed to support patients performing physiotherapy exercises at home. The system analyzes video input from a standard webcam to:
Identify the exercise type (e.g., squat, push-up).
Evaluate the correctness of form using pose detection and deep learning models.
By reducing reliance on in-person sessions, Pocket Physio aims to improve accessibility, enhance adherence, and lower costs for physiotherapy treatment programs
ENGG2112_report
.
🎯 Objectives
Exercise Identification – Classify the exercise type using a 2D CNN model.
Form Evaluation – Assess movement quality (correct vs. incorrect) with CNN + LSTM models.
Accessibility – Deliver real-time feedback using only a webcam (no wearables or special sensors).
User-Friendly Interface – Provide simple, intuitive feedback to patients during exercise.
🛠 Methodology
Data Acquisition
Model 1 dataset: Push-up and squat images (Kaggle + Google + YouTube + iStock).
Model 2 dataset: Videos of squats and push-ups, labeled as correct/incorrect (Kaggle, Waseda University, and joint-angle datasets).
Models
Model 1 (Exercise Classifier):
2D CNN (TensorFlow/Keras) trained on 256×256 RGB images.
Achieved ~99% accuracy on validation set.
Model 2 (Form Evaluator):
CNN + LSTM pipeline using MediaPipe keypoints.
Push-ups (Model 2A): ~83% accuracy.
Squats (Model 2B): ~77% accuracy for incorrect detection, but weaker on correct poses.
User Interface
Displays the detected exercise type.
Provides real-time corrective or encouraging feedback.
📊 Results
Model 1: High performance on training/validation but reduced accuracy (~60%) on live webcam input (dataset bias, overfitting).
Model 2A (push-ups): Good at detecting correct form, weaker at spotting incorrect poses.
Model 2B (squats): Stronger at spotting incorrect squats than correct ones (dataset imbalance).
