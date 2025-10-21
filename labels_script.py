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
        print(f" Skipped: {f} (unrecognized format)")

# Write to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print(f" Generated {len(rows)} labeled entries in {output_csv}")

# --- New: Generate image_labels.csv for images/ directory ---

def generate_image_labels(images_dir="images", output_csv="images/image_labels.csv"):
    image_rows = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                rel_dir = os.path.relpath(root, images_dir)
                label = rel_dir.split(os.sep)[0] if rel_dir != "." else "unknown"
                rel_path = os.path.join(rel_dir, file) if rel_dir != "." else file
                image_rows.append((rel_path, label))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])
        writer.writerows(image_rows)
    print(f" Generated {len(image_rows)} labeled entries in {output_csv}")

# Run the new function
generate_image_labels()
