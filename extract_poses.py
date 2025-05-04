import cv2
import os
import numpy as np
import mediapipe as mp

video_folder = "frame"  # Folder with all your .mp4 files
output_dir = "pose_data"
os.makedirs(output_dir, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

frame_rate = 5

for video_file in os.listdir(video_folder):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"🚨 Couldn't open {video_file}")
        continue

    print(f"🎥 Processing {video_file}...")
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
                coords = np.zeros(33 * 3)
            pose_vectors.append(coords)

        frame_count += 1

    cap.release()
    npy_name = os.path.splitext(video_file)[0] + "_pose.npy"
    np.save(os.path.join(output_dir, npy_name), np.array(pose_vectors))
    print(f"✅ Saved {len(pose_vectors)} pose frames to {npy_name}")
