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
        
        # yt-dlp options for high-quality, short video download
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
            # Filter for shorter videos (better for technique analysis)
            "match_filter": "duration < 180",  # Only videos under 3 minutes
        }
        
        # Sports technique search queries - optimized for short, focused videos
        self.search_queries = {
            # Boxing techniques - short tutorial focus
            "boxing_jab": "boxing jab technique tutorial short",
            "boxing_cross": "boxing cross punch tutorial short",
            "boxing_hook": "boxing hook punch tutorial short",
            "boxing_uppercut": "boxing uppercut tutorial short",
            "boxing_defense": "boxing defense tutorial short",
            "boxing_footwork": "boxing footwork tutorial short",
            
            # Fencing techniques - short tutorial focus
            "fencing_lunge": "fencing lunge tutorial short",
            "fencing_parry": "fencing parry tutorial short",
            "fencing_riposte": "fencing riposte tutorial short",
            "fencing_footwork": "fencing footwork tutorial short",
            "fencing_guard": "fencing guard tutorial short",
            "fencing_attack": "fencing attack tutorial short",
        }
    
    def check_video_quality(self, video_info: dict) -> tuple[bool, str]:
        """
        Check if video meets quality criteria
        
        Args:
            video_info: Video information from yt-dlp
            
        Returns:
            Tuple of (is_good_quality, reason)
        """
        try:
            # Check duration (prefer short videos)
            duration = video_info.get('duration', 0)
            if duration > 180:  # Longer than 3 minutes
                return False, f"Too long ({duration}s)"
            if duration < 10:  # Too short
                return False, f"Too short ({duration}s)"
            
            # Check resolution
            height = video_info.get('height', 0)
            if height < 360:  # Too low resolution
                return False, f"Low resolution ({height}p)"
            
            # Check view count (prefer popular videos)
            view_count = video_info.get('view_count', 0)
            if view_count < 1000:  # Too few views
                return False, f"Low views ({view_count})"
            
            # Check title relevance
            title = video_info.get('title', '').lower()
            relevant_keywords = ['tutorial', 'technique', 'how to', 'training', 'drill']
            if not any(keyword in title for keyword in relevant_keywords):
                return False, "Title not relevant"
            
            return True, "Good quality"
            
        except Exception as e:
            return False, f"Quality check error: {e}"
    
    def search_and_download_videos(self, query: str, label: str, max_videos: int = 3):
        """
        Search for videos and download them with quality filtering
        
        Args:
            query: Search query for YouTube
            label: Label for the technique
            max_videos: Maximum number of videos to download
        """
        print(f"🔍 Searching for '{query}' (label: {label})...")
        
        labeled_entries = []
        attempts = 0
        max_attempts = max_videos * 3  # Try more videos to find good ones
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Search for more videos to filter
                search_query = f"ytsearch{max_attempts}:{query}"
                result = ydl.extract_info(search_query, download=False)  # Don't download yet
                
                if result and 'entries' in result:
                    for entry in result['entries']:
                        if entry and attempts < max_attempts:
                            attempts += 1
                            
                            # Check video quality
                            is_good_quality, reason = self.check_video_quality(entry)
                            
                            if is_good_quality:
                                print(f"  [{attempts}] ✅ Good quality: {entry.get('title', 'Unknown')}")
                                
                                # Now download the good quality video
                                try:
                                    ydl.download([entry['webpage_url']])
                                    
                                    video_title = entry.get('title', 'video').replace(" ", "_").replace("/", "_")
                                    filename = f"{video_title[:40]}.mp4"
                                    labeled_entries.append((filename, label))
                                    print(f"  ✅ Downloaded: {filename}")
                                    
                                    if len(labeled_entries) >= max_videos:
                                        break
                                        
                                except Exception as e:
                                    print(f"  ❌ Download failed: {e}")
                            else:
                                print(f"  [{attempts}] ❌ Skipped: {reason}")
                
        except Exception as e:
            print(f"⚠️ Error searching for label '{label}': {e}")
        
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
    
    # Scrape videos for all techniques (fewer but higher quality)
    labeled_videos = scraper.scrape_all_sports_videos(max_videos_per_technique=1)
    
    # Print summary
    print("\n📋 Summary of downloaded videos:")
    for filename, label in labeled_videos:
        print(f"  {label}: {filename}")
    
    print(f"\n🎯 Total high-quality videos: {len(labeled_videos)}")
    print("💡 These short, focused videos are perfect for AI analysis!")

if __name__ == "__main__":
    main() 