import os
import cv2
import mediapipe as mp
import json
from pathlib import Path

VIDEOS_DIR = "scraped_videos"
POSE_DATA_DIR = "pose_data"

os.makedirs(POSE_DATA_DIR, exist_ok=True)

mp_pose = mp.solutions.pose


def extract_poses_from_video(video_path, output_json_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frame_idx = 0
    pose_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        keypoints = []
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                keypoints.append({
                    "id": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })
        pose_data.append({
            "frame": frame_idx,
            "keypoints": keypoints
        })
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break
    
    cap.release()
    pose.close()
    
    # Save pose data to JSON
    with open(output_json_path, 'w') as f:
        json.dump(pose_data, f, indent=2)
    print(f"✅ Saved pose data: {output_json_path} ({frame_idx} frames)")
    return True

def process_all_videos():
    print("🚀 Extracting poses from all videos in scraped_videos/")
    for file in os.listdir(VIDEOS_DIR):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(VIDEOS_DIR, file)
            video_name = Path(file).stem
            output_json = os.path.join(POSE_DATA_DIR, f"{video_name}_poses.json")
            if os.path.exists(output_json):
                print(f"⏩ Skipping {file} (already processed)")
                continue
            extract_poses_from_video(video_path, output_json)

if __name__ == "__main__":
    process_all_videos()
