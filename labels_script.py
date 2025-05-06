import os
import csv

pose_dir = "pose_data"
output_csv = os.path.join(pose_dir, "labels.csv")

# Get all pose files
pose_files = [f for f in os.listdir(pose_dir) if f.endswith("_pose.npy")]

# Parse labels from filenames (assumes format: label_something_pose.npy)
rows = []
for f in pose_files:
    parts = f.split("_")
    if len(parts) >= 2:
        label = parts[0]  # 'parry', 'thrust', etc.
        rows.append((f, label))
    else:
        print(f"⚠️ Skipped: {f} (unrecognized format)")

# Write to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print(f"✅ Generated {len(rows)} labeled entries in {output_csv}")
