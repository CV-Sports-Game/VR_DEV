import os
import numpy as np

pose_dir = "pose_data"
all_data = []

for file in os.listdir(pose_dir):
    if file.endswith("_pose.npy"):
        path = os.path.join(pose_dir, file)
        data = np.load(path)
        all_data.append(data)

if all_data:
    merged = np.concatenate(all_data, axis=0)
    np.save(os.path.join(pose_dir, "pose_sequences.npy"), merged)
    print(f"✅ Combined {len(all_data)} files into pose_sequences.npy with shape {merged.shape}")
else:
    print("🚨 No pose data found to merge.")
