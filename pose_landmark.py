import cv2
import os
import random

pose_frames_folder = "pose_frames"
sample_frames = random.sample(os.listdir(pose_frames_folder), 5)  # Pick 5 random frames

for frame_name in sample_frames:
    frame_path = os.path.join(pose_frames_folder, frame_name)
    img = cv2.imread(frame_path)

    if img is None:
        print(f"Error loading {frame_name}")
        continue

    cv2.imshow("Pose Frame", img)
    cv2.waitKey(500)  # Show each frame for 500ms

cv2.destroyAllWindows()
