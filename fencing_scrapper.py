import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
import io
import time
import random

def download_image(url, filepath):
    """Download image from URL and save to filepath"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Open image with PIL to verify it's valid
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')  # Convert to RGB if needed
        
        # Save the image
        img.save(filepath, 'JPEG', quality=85)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def scrape_images(search_query, folder_name, num_images=30):
    """Scrape images for a specific search query"""
    print(f"🔍 Searching for '{search_query}'...")
    
    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Search for images
    with DDGS() as ddgs:
        try:
            results = list(ddgs.images(search_query, max_results=num_images + 10))
            
            downloaded = 0
            for i, result in enumerate(results):
                if downloaded >= num_images:
                    break
                    
                image_url = result['image']
                filename = f"{os.path.basename(folder_name)}_{downloaded}.jpg"
                filepath = os.path.join(folder_name, filename)
                
                print(f"  Downloading {filename}...")
                if download_image(image_url, filepath):
                    downloaded += 1
                    print(f"  ✅ Downloaded {filename}")
                else:
                    print(f"  ❌ Failed to download {filename}")
                
                # Small delay to be respectful
                time.sleep(random.uniform(0.5, 1.5))
            
            print(f"✅ Downloaded {downloaded} images for {folder_name}")
            return downloaded
            
        except Exception as e:
            print(f"❌ Error searching for {search_query}: {e}")
            return 0

def main():
    """Main function to scrape all boxing and fencing images"""
    print("🚀 Starting Boxing and Fencing Image Scraper")
    print("=" * 50)
    
    # Define search queries for boxing poses
    boxing_queries = {
        "boxing_punch": "boxing punch technique",
        "boxing_uppercut": "boxing uppercut technique", 
        "boxing_straight_punch": "boxing straight punch jab",
        "boxing_fast_punch": "boxing fast punch combination",
        "boxing_hook": "boxing hook punch technique",
        "boxing_block": "boxing block defense technique",
        "boxing_guard": "boxing guard stance position",
        "boxing_footwork": "boxing footwork movement"
    }
    
    # Define search queries for fencing poses
    fencing_queries = {
        "fencing_lunge": "fencing lunge technique",
        "fencing_slide": "fencing slide footwork",
        "fencing_parry": "fencing parry defense",
        "fencing_block": "fencing block technique",
        "fencing_guard": "fencing guard position",
        "fencing_en_garde": "fencing en garde stance",
        "fencing_riposte": "fencing riposte attack",
        "fencing_footwork": "fencing footwork movement"
    }
    
    # Create images directory
    os.makedirs("images", exist_ok=True)
    
    total_downloaded = 0
    
    # Scrape boxing images
    print("\n🥊 Scraping Boxing Images...")
    for folder, query in boxing_queries.items():
        folder_path = os.path.join("images", folder)
        downloaded = scrape_images(query, folder_path, 30)
        total_downloaded += downloaded
        print(f"📁 {folder}: {downloaded} images")
        print("-" * 30)
    
    # Scrape fencing images
    print("\n🤺 Scraping Fencing Images...")
    for folder, query in fencing_queries.items():
        folder_path = os.path.join("images", folder)
        downloaded = scrape_images(query, folder_path, 30)
        total_downloaded += downloaded
        print(f"📁 {folder}: {downloaded} images")
        print("-" * 30)
    
    print(f"\n🎉 Scraping Complete!")
    print(f"📊 Total images downloaded: {total_downloaded}")
    print(f"📁 Images saved in: images/")
    
    # Create a summary of what was downloaded
    print("\n📋 Summary:")
    for folder in list(boxing_queries.keys()) + list(fencing_queries.keys()):
        folder_path = os.path.join("images", folder)
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {folder}: {count} images")

if __name__ == "__main__":
    main()
