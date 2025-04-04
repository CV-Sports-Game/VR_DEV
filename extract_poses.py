import cv2
import os
import numpy as np
import mediapipe as mp

video_path = "fencing.mp4"
output_dir = "pose_data"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
pose = mp.solutions.pose.Pose()
frame_rate = 5
frame_count = 0
pose_vectors = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_rate == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark]).flatten()
        else:
            coords = np.zeros(33 * 3)  # 99 zeros if no detection
        pose_vectors.append(coords)
    frame_count += 1

cap.release()
np.save(os.path.join(output_dir, "pose_sequences.npy"), np.array(pose_vectors))
print(f"Saved {len(pose_vectors)} pose frames.")
