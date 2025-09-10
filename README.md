# Pocket-Physio

---

## Overview  
The system analyzes video input from a standard webcam to:  
1. **Identify the exercise type** (push-up or squat)  
2. **Evaluate correctness of form** (correct vs. incorrect posture)  

Our vision: make physiotherapy **more accessible, affordable, and effective** through remote AI-assisted feedback.  

---

## 🎯 Objectives  
- **Preprocessing** – Standardize data, normalize pose keypoints, and augment images  
- **Exercise Identification** – Classify type of exercise (Model 1: 2D CNN)  
- **Form Feedback** – Assess correctness of form (Model 2: CNN + LSTM)  
- **Accessibility** – Works with only a webcam, no wearables or special sensors  
- **User Interface** – Real-time corrective or encouraging feedback  

---

## Methodology  

### Data  
- **Model 1** – Push-up & squat images (Kaggle, Google Images, YouTube, iStock)  
- **Model 2** – Correct/incorrect exercise videos (Kaggle & Waseda University datasets)  

### Models  
- **Model 1 (Exercise Classifier)**  
  - 2D CNN (TensorFlow/Keras)  
  - Achieved **~99% validation accuracy**  
- **Model 2 (Form Evaluator)**  
  - CNN + LSTM on MediaPipe pose keypoints  
  - Push-ups: **83% accuracy**  
  - Squats: Better at detecting incorrect form (**77.6% recall**) than correct form  

### User Interface  
- Displays:  
  - Current exercise type  
  - Real-time feedback on form quality  

---

## 📊 Results  

| Model   | Accuracy | Precision | Recall | F1 | ROC AUC |
|---------|----------|-----------|--------|----|---------|
| Model 1 | 99.1%    | 1.00      | 0.993  | 0.996 | 1.0 |
| Model 2A (Push-ups) | 83.3% | 0.71 | 1.0 | 0.83 | 0.92 |
| Model 2B (Squats)  | 77.6% (incorrect) | 0.87 | 0.78 | 0.82 | 0.85 |

**Limitations:** Performance drops with real webcam data due to dataset bias, lighting, and camera angle variation.  

---

## Issues & Limitations  
- Dataset imbalance & limited diversity  
- Live webcam performance gap (vs. curated datasets)  
- Latency with sequential pipelines (CNN ➝ LSTM)  
- Ethical & legal risks if misclassifications cause injury  

---

## Further development
- Expand datasets with physiotherapist-labeled data  
- Build a mobile app for accessibility  
- Explore **TGA Class IIa medical device certification**  
- Train on more exercises beyond squats/push-ups  
