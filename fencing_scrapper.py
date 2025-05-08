import os
import yt_dlp
import csv

# Create label-to-query mapping
search_queries = {
    "thrust": "fencing thrust slow motion",
    "parry": "fencing parry slow motion",
    "slash": "fencing slash technique",
    "block": "fencing block tutorial",
    "roll": "fencing evasion roll",
    "jump": "fencing jump footwork"
}

output_dir = "scraped_videos"
os.makedirs(output_dir, exist_ok=True)

# This will store (filename, label) pairs
labeled_entries = []

# yt-dlp options
ydl_opts = {
    "format": "best[ext=mp4]",
    "outtmpl": os.path.join(output_dir, "%(title).40s.%(ext)s"),
    "quiet": True,
    "noplaylist": True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for label, query in search_queries.items():
        print(f"🔍 Searching for '{query}' (label: {label})...")
        try:
            result = ydl.extract_info(f"ytsearch5:{query}", download=True)
            for entry in result['entries']:
                video_title = entry.get('title', 'video').replace(" ", "_").replace("/", "_")
                filename = f"{video_title[:40]}.mp4"
                labeled_entries.append((filename, label))
        except Exception as e:
            print(f"⚠️ Error downloading for label '{label}': {e}")

# Save labels.csv
csv_path = os.path.join(output_dir, "labels.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(labeled_entries)

print(f"\n✅ Downloaded {len(labeled_entries)} videos and saved labels to {csv_path}")
