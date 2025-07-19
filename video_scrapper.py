import os
import yt_dlp
import csv
import time
from pathlib import Path

class SportsVideoScraper:
    """
    A video scraper for downloading sports technique videos from YouTube
    for AI analysis and movement sequence analysis.
    """
    
    def __init__(self, output_dir: str = "scraped_videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # yt-dlp options for high-quality video download
        self.ydl_opts = {
            "format": "best[height<=720][ext=mp4]",  # 720p max for reasonable file size
            "outtmpl": os.path.join(output_dir, "%(title).40s.%(ext)s"),
            "quiet": False,
            "noplaylist": True,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ignoreerrors": True,
            "no_warnings": False,
            "extractaudio": False,
            "audioformat": "mp3",
        }
        
        # Sports technique search queries
        self.search_queries = {
            # Boxing techniques
            "boxing_jab": "boxing jab technique tutorial",
            "boxing_cross": "boxing cross punch technique",
            "boxing_hook": "boxing hook punch tutorial",
            "boxing_uppercut": "boxing uppercut technique",
            "boxing_defense": "boxing defense techniques",
            "boxing_footwork": "boxing footwork drills",
            
            # Fencing techniques
            "fencing_lunge": "fencing lunge technique tutorial",
            "fencing_parry": "fencing parry defense",
            "fencing_riposte": "fencing riposte attack",
            "fencing_footwork": "fencing footwork drills",
            "fencing_guard": "fencing guard position",
            "fencing_attack": "fencing attack techniques",
        }
    
    def search_and_download_videos(self, query: str, label: str, max_videos: int = 3):
        """
        Search for videos and download them
        
        Args:
            query: Search query for YouTube
            label: Label for the technique
            max_videos: Maximum number of videos to download
        """
        print(f"🔍 Searching for '{query}' (label: {label})...")
        
        labeled_entries = []
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Search for videos
                search_query = f"ytsearch{max_videos}:{query}"
                result = ydl.extract_info(search_query, download=True)
                
                if result and 'entries' in result:
                    for entry in result['entries']:
                        if entry:
                            video_title = entry.get('title', 'video').replace(" ", "_").replace("/", "_")
                            filename = f"{video_title[:40]}.mp4"
                            labeled_entries.append((filename, label))
                            print(f"  ✅ Downloaded: {filename}")
                
        except Exception as e:
            print(f"⚠️ Error downloading for label '{label}': {e}")
        
        return labeled_entries
    
    def scrape_all_sports_videos(self, max_videos_per_technique: int = 3):
        """
        Scrape videos for all sports techniques
        
        Args:
            max_videos_per_technique: Number of videos to download per technique
        """
        print("🚀 Starting Sports Video Scraper")
        print("=" * 50)
        
        all_labeled_entries = []
        
        for label, query in self.search_queries.items():
            print(f"\n📹 Processing: {label}")
            print("-" * 30)
            
            entries = self.search_and_download_videos(query, label, max_videos_per_technique)
            all_labeled_entries.extend(entries)
            
            # Add delay between searches to be respectful
            time.sleep(2)
        
        # Save labels to CSV
        csv_path = os.path.join(self.output_dir, "video_labels.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(all_labeled_entries)
        
        print(f"\n🎉 Video scraping complete!")
        print(f"📊 Total videos downloaded: {len(all_labeled_entries)}")
        print(f"📁 Videos saved in: {self.output_dir}")
        print(f"📋 Labels saved to: {csv_path}")
        
        return all_labeled_entries
    
    def get_video_info(self, video_path: str):
        """
        Get information about a downloaded video
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = yt_dlp.YoutubeDL.extract_info(ydl, video_path, download=False)
                return {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "resolution": info.get("resolution", "Unknown"),
                    "filesize": info.get("filesize", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "view_count": info.get("view_count", 0),
                }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}

def main():
    """Main function to run the video scraper"""
    scraper = SportsVideoScraper()
    
    # Scrape videos for all techniques
    labeled_videos = scraper.scrape_all_sports_videos(max_videos_per_technique=2)
    
    # Print summary
    print("\n📋 Summary of downloaded videos:")
    for filename, label in labeled_videos:
        print(f"  {label}: {filename}")

if __name__ == "__main__":
    main() 