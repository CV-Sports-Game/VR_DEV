import cv2
import os

video_path = "fencing.mp4"
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 5  # Capture 1 frame every 5 frames

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_rate == 0:
        frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames and saved in '{output_folder}'")
