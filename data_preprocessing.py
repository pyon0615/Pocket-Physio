#!/usr/bin/env python3
# data_preprocessing.py

import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import mediapipe as mp

# === CONFIG ===
VIDEO_DIRS  = ["Video_Dataset", "Correct sequence", "Wrong sequence"]
ANGLE_DIR   = "processed_data"
LABEL_DIR   = "processed_data"
MAX_FRAMES  = 60

# Mediapipe setup
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(static_image_mode=False)

# === JOINT COMBINATIONS FOR ANGLE FEATURES ===
JOINT_ANGLES = [
    (11,13,15), (12,14,16),
    (23,25,27), (24,26,28),
    (11,23,25), (12,24,26),
    (13,15,19), (14,16,20),
    (23,27,31), (24,28,32),
    (0,11,13),  (0,12,14),
    (0,23,25),  (0,24,26),
    (15,19,21), (16,20,22),
    (11,12,14), (23,24,26),
    (11,13,25), (12,14,26),
    (13,25,27), (14,26,28),
]

GROUND_JOINTS = [
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
    31, 32
]

VERTICAL = np.array([0.0, -1.0, 0.0], dtype=float)

def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def extract_frame_features(landmarks):
    feats = []
    for (i, j, k) in JOINT_ANGLES:
        feats.append(angle_between(landmarks[i], landmarks[j], landmarks[k]))
    for j in GROUND_JOINTS:
        up = landmarks[j] + VERTICAL
        down = landmarks[j] - VERTICAL
        feats.append(angle_between(up, landmarks[j], down))
    return np.array(feats, dtype=float)

def extract_angles_from_video(path):
    cap = cv2.VideoCapture(path)
    seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640,480))
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            lm = np.array([[p.x, p.y, p.z] for p in results.pose_landmarks.landmark])
            seq.append(lm)
    cap.release()
    if len(seq) < 1:
        return None
    if len(seq) < MAX_FRAMES:
        last = seq[-1]
        for _ in range(MAX_FRAMES - len(seq)):
            seq.append(last.copy())
    return seq[:MAX_FRAMES]

def infer_label_from_folder(folder_name):
    base = folder_name.lower()
    if base in ("good", "correct sequence"):
        return np.array([1,0], dtype=int)
    return np.array([0,1], dtype=int)

def infer_type_from_filename(filename):
    name = filename.lower()
    # also catch "push up" with a space
    # we remove spaces so "push up" -> "pushup"
    compact = name.replace(" ", "")
    if "pushup" in compact:
        return "pushup"
    elif "squat" in name:
        return "squat"
    else:
        return "unknown"

def main():
    os.makedirs(ANGLE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    scaler = StandardScaler()

    X_all = []
    y_all = []
    types = []

    for video_root in VIDEO_DIRS:
        for root, _, files in os.walk(video_root):
            mp4s = [f for f in files if f.lower().endswith(".mp4")]
            if not mp4s:
                continue

            rel = os.path.relpath(root, video_root)
            folder_label = rel.split(os.sep)[0] if rel != '.' else os.path.basename(video_root)
            label_vec = infer_label_from_folder(folder_label)

            for fname in mp4s:
                fp = os.path.join(root, fname)
                lm_seq = extract_angles_from_video(fp)
                if lm_seq is None:
                    print(f"⚠️ Too few landmarks, skipping: {fp}")
                    continue

                feats = np.stack([extract_frame_features(lm) for lm in lm_seq])
                vel = np.vstack([np.zeros((1, feats.shape[1])), np.diff(feats, axis=0)])
                X = np.concatenate([feats, vel], axis=1)

                T, D = X.shape
                flat = X.reshape(-1, D)
                flat = scaler.fit_transform(flat)
                Xs = flat.reshape(T, D)

                base = os.path.splitext(os.path.basename(fp))[0]
                sample_type = infer_type_from_filename(base)

                np.save(os.path.join(ANGLE_DIR, base + "_angles.npy"), Xs)
                np.save(os.path.join(LABEL_DIR, base + "_label.npy"), label_vec)
                np.save(os.path.join(ANGLE_DIR, base + "_type.npy"), sample_type)
                print(f"✅ {fname}: saved *_angles.npy, *_label.npy, and *_type.npy")

                X_all.append(Xs)
                y_all.append(label_vec)
                types.append(sample_type)

    # ========= Combine and Save for Model 2A & 2B =========
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    types  = np.array(types)

    # Model 2A: Push-up
    idx_push = np.where(types == "pushup")[0]
    np.save("pushup_X.npy", X_all[idx_push])
    np.save("pushup_y.npy", y_all[idx_push])
    print(f"📦 Saved pushup_X.npy and pushup_y.npy — Model 2A ✅")

    # Model 2B: Squat
    idx_squat = np.where(types == "squat")[0]
    np.save("squat_X.npy", X_all[idx_squat])
    np.save("squat_y.npy", y_all[idx_squat])
    print(f"📦 Saved squat_X.npy and squat_y.npy — Model 2B ✅")

if __name__ == "__main__":
    main()
