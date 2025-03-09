import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

frames_folder = "frames"
output_folder = "pose_frames"
os.makedirs(output_folder, exist_ok=True)

for frame_name in os.listdir(frames_folder):
    frame_path = os.path.join(frames_folder, frame_name)
    frame = cv2.imread(frame_path)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imwrite(os.path.join(output_folder, frame_name), frame)

print("Pose estimation completed! Processed frames saved in 'pose_frames/'")
