import cv2
import os
import random

frames_folder = "frames"
sample_frames = random.sample(os.listdir(frames_folder), 5)  # Pick 5 random frames

for frame_name in sample_frames:
    frame_path = os.path.join(frames_folder, frame_name)
    img = cv2.imread(frame_path)

    if img is None:
        print(f"Error loading {frame_name}")
        continue

    cv2.imshow("Frame", img)
    cv2.waitKey(500)  # Show for 500ms

cv2.destroyAllWindows()
