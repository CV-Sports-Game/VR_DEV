import cv2
import os
import random
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

video_folder = "frame"
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

sample_videos = random.sample(video_files, 5)

with mp_pose.Pose(static_image_mode=True) as pose:
    for video_name in sample_videos:
        video_path = os.path.join(video_folder, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error loading {video_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"No frames in {video_name}")
            continue

        random_frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Failed to read frame from {video_name}")
            continue

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Draw landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # Show the frame with overlays
        cv2.imshow(f"Pose in {video_name}", frame)
        cv2.waitKey(1000)  # Display for 1 second

cv2.destroyAllWindows()
