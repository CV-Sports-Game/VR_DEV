# 📘 Fencing Pose Classification Pipeline Documentation

This guide helps you set up and run a fencing move classification project using MediaPipe for pose extraction and a Transformer-based PyTorch model for classification.

---

## 🔧 1. Installation Requirements

### Python Version:

* Python 3.8 or newer

### Install required libraries:

```bash
pip install mediapipe opencv-python torch torchvision yt-dlp pandas scikit-learn matplotlib
```

---

## 📁 2. Folder Structure

```
project_root/
├── frame/               # Input folder for .mp4 videos
├── pose_data/           # Stores extracted pose .npy files + labels.csv
├── scraped_videos/      # Stores YouTube-scraped videos (optional)
├── extract_poses.py
├── fencing_scrapper.py
├── labels_script.py
├── merge_pose_data.py
├── pose_dataset.py
├── pose_landmark.py
├── pose_transformer.py
├── train_model.py
├── video.py
├── visualize_frames.py
```

---

## 📹 3. \[Optional] Scrape Fencing Videos from YouTube

Run this to auto-download and label fencing videos:

```bash
python3 fencing_scrapper.py
```

This saves videos to `scraped_videos/` and labels to `scraped_videos/labels.csv`.

Then move videos to the main video folder:

```bash
mv scraped_videos/*.mp4 frame/
cp scraped_videos/labels.csv pose_data/
```

---

## 🕴️ 4. Extract Poses from Video Frames

Run this to extract 3D pose landmarks from all `.mp4` videos:

```bash
python3 extract_poses.py
```

This saves pose arrays (`*.npy`) into the `pose_data/` folder.

---

## 🏷️ 5. Create a Labels File

If you named videos like `thrust_clip1.mp4`, run:

```bash
python3 labels_script.py
```

This generates `pose_data/labels.csv` with auto-labeled rows:

```
filename,label
thrust_clip1_pose.npy,thrust
parry_clip2_pose.npy,parry
```

---

## 📦 6. Merge All Pose Files

Run this script to combine all individual `.npy` pose files into a single sequence file:

```bash
python3 merge_pose_data.py
```

It creates `pose_data/pose_sequences.npy`

---

## 🧠 7. Train the Pose Transformer

```bash
python3 train_model.py
```

This:

* Loads labeled `.npy` files
* Trains a Transformer classifier
* Saves the model as `pose_model.pth`
* Prints label mapping and sample predictions

---

## 🧪 8. Evaluate Results

The training script automatically prints predictions on sample inputs. If everything works:

```
Sample 1: True = thrust, Predicted = thrust
Sample 2: True = parry, Predicted = roll
...
```

---

## 👀 9. Visualize Frames

Use this script to view random frames:

```bash
python3 visualize_frames.py
```

---

## 🛠 Troubleshooting

* If all predictions are one label → check `labels.csv` balance
* If pose extraction fails → ensure videos are readable `.mp4` and not corrupted
* Update `yt-dlp` if scraping fails:

```bash
pip install -U yt-dlp
```

---
run sequence 
1. extract_poses
2. labels_script
3. pose_dataset
4. pose_landmark
5. merge_pose
6. train_model
8. video.py and viualize
9. pose_transformer

May need to download 
1. pytorch

2. other dependencies

