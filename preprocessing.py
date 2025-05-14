import os
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mediapipe as mp

# =========================
# Pose Extraction Function
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_all_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            frame_keypoints = []
            for lm in results.pose_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
            keypoints_all_frames.append(frame_keypoints)

    cap.release()
    return np.array(keypoints_all_frames)

# =========================
# Normalization Function
# =========================
def normalize_sequence(sequence):
    scaler = MinMaxScaler()
    return scaler.fit_transform(sequence)

# =========================
# Preprocess All Videos
# =========================
def process_videos(input_folder, label, label_name, output_folder='processed_data', label_folder='labels'):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4') or f.endswith('.mov')]

    for idx, filename in enumerate(video_files):
        video_path = os.path.join(input_folder, filename)
        keypoints = extract_keypoints(video_path)

        if keypoints.shape[0] == 0:
            print(f"⚠️ Skipped: {filename} (no keypoints)")
            continue

        normalized = normalize_sequence(keypoints)
        save_name = f"{label_name}_{idx+1}.npy"
        np.save(os.path.join(output_folder, save_name), normalized)

        with open(os.path.join(label_folder, f"{label_name}_{idx+1}.txt"), 'w') as f:
            f.write(str(label))

        print(f"✅ Saved: {save_name}")

# =========================
# Run Processing
# =========================
if __name__ == "__main__":
    process_videos("Correct sequence", [1, 0], "correct")
    process_videos("Wrong sequence", [0, 1], "wrong")

