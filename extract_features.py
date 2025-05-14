import os
import numpy as np

# === Angle Calculation ===
def calculate_angle(a, b, c):
    """Calculate the angle at point b (in degrees) formed by points a-b-c"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# === Feature Extraction from 1 Video ===
def extract_features_from_sequence(sequence):
    features = []
    for frame in sequence:
        joints = np.reshape(frame, (33, 3))

        # Select key joints
        l_shoulder = joints[11]
        l_elbow    = joints[13]
        l_wrist    = joints[15]

        l_hip   = joints[23]
        l_knee  = joints[25]
        l_ankle = joints[27]

        r_shoulder = joints[12]
        r_elbow    = joints[14]
        r_wrist    = joints[16]

        # Calculate angles
        angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
        angle_l_knee  = calculate_angle(l_hip, l_knee, l_ankle)
        angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)

        # Calculate distances
        dist_hip_ankle = np.linalg.norm(l_hip - l_ankle)
        dist_shoulder_wrist = np.linalg.norm(l_shoulder - l_wrist)

        # Add to frame feature
        features.append([
            angle_l_elbow,
            angle_r_elbow,
            angle_l_knee,
            dist_hip_ankle,
            dist_shoulder_wrist
        ])

    return np.array(features)  # shape: (60, 5)

# === Run Extraction on Folder ===
def process_form_data(input_folder='processed_data', label_folder='labels', output_folder='form_data'):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if not file.endswith(".npy"):
            continue

        sequence = np.load(os.path.join(input_folder, file))
        features = extract_features_from_sequence(sequence)

        # Save features
        np.save(os.path.join(output_folder, file), features)

        # Copy matching label
        label_file = file.replace('.npy', '.txt')
        label_path = os.path.join(label_folder, label_file)
        if os.path.exists(label_path):
            os.system(f"cp {label_path} {os.path.join(output_folder, label_file)}")

        print(f"✅ Processed {file} → shape {features.shape}")

# === Run this file directly ===
if __name__ == "__main__":
    process_form_data()
