import cv2
import os

video_folder = "frame"  # Folder with your .mp4 videos
output_base = "frames"  # Where extracted frames go
frame_rate = 5          # Save 1 out of every 5 frames

os.makedirs(output_base, exist_ok=True)

for video_file in os.listdir(video_folder):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"🚨 Couldn't open {video_file}")
        continue

    print(f"🎥 Extracting frames from {video_file}...")

    # Create a subfolder for each video
    video_name = os.path.splitext(video_file)[0]
    output_folder = os.path.join(output_base, video_name)
    os.makedirs(output_folder, exist_ok=True)

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
    print(f"✅ Saved {saved_count} frames to '{output_folder}'")

cv2.destroyAllWindows()
