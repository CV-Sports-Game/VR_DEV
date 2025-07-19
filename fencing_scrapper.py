import os
import requests
from duckduckgo_search import DDGS
from PIL import Image, ImageFilter
import io
import time
import random
import hashlib
from collections import defaultdict

def calculate_image_hash(img):
    """Calculate a hash for image content to detect duplicates"""
    # Resize to small size for faster hashing
    small_img = img.resize((8, 8), Image.Resampling.LANCZOS)
    # Convert to grayscale
    gray_img = small_img.convert('L')
    # Get pixel values
    pixels = list(gray_img.getdata())
    # Calculate hash
    return hashlib.md5(str(pixels).encode()).hexdigest()

def check_image_quality(img):
    """Check if image is good quality (not too blurry, good contrast)"""
    try:
        # Convert to grayscale for analysis
        gray = img.convert('L')
        
        # Check image size
        if img.size[0] < 200 or img.size[1] < 200:
            return False, "Too small"
        
        # Check aspect ratio (avoid very wide or tall images)
        aspect_ratio = img.size[0] / img.size[1]
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            return False, "Bad aspect ratio"
        
        # Check for blur using variance of Laplacian
        laplacian = gray.filter(ImageFilter.FIND_EDGES)
        variance = sum(laplacian.getextrema()) / 2
        if variance < 10:  # Low variance indicates blur
            return False, "Too blurry"
        
        # Check contrast (standard deviation of pixel values)
        pixels = list(gray.getdata())
        mean_pixel = sum(pixels) / len(pixels)
        variance_pixels = sum((p - mean_pixel) ** 2 for p in pixels) / len(pixels)
        if variance_pixels < 500:  # Low contrast
            return False, "Low contrast"
        
        return True, "Good quality"
        
    except Exception as e:
        return False, f"Quality check error: {e}"

def download_image(url, filepath, existing_hashes=None):
    """Download image from URL and save to filepath"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        
        # Check if it's actually an image
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            print(f"Not an image: {content_type}")
            return False, None
        
        # Open image with PIL to verify it's valid
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')  # Convert to RGB if needed
        
        # Check for duplicates
        if existing_hashes is not None:
            img_hash = calculate_image_hash(img)
            if img_hash in existing_hashes:
                print(f"Duplicate image detected (hash: {img_hash[:8]})")
                return False, None
        
        # Check image quality
        is_good_quality, quality_reason = check_image_quality(img)
        if not is_good_quality:
            print(f"Poor quality: {quality_reason}")
            return False, None
        
        # Save the image
        img.save(filepath, 'JPEG', quality=85)
        
        # Return success and hash for future duplicate detection
        img_hash = calculate_image_hash(img)
        return True, img_hash
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False, None

def scrape_images(search_query, folder_name, num_images=30, global_hashes=None):
    """Scrape images for a specific search query"""
    print(f"🔍 Searching for '{search_query}'...")
    
    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Initialize hash set for this category
    category_hashes = set()
    if global_hashes is None:
        global_hashes = set()
    
    # Search for images
    with DDGS() as ddgs:
        try:
            results = list(ddgs.images(search_query, max_results=num_images + 30))
            
            downloaded = 0
            attempts = 0
            max_attempts = num_images + 30
            
            for i, result in enumerate(results):
                if downloaded >= num_images or attempts >= max_attempts:
                    break
                    
                attempts += 1
                image_url = result['image']
                filename = f"{os.path.basename(folder_name)}_{downloaded}.jpg"
                filepath = os.path.join(folder_name, filename)
                
                print(f"  [{attempts}/{max_attempts}] Downloading {filename}...")
                success, img_hash = download_image(image_url, filepath, global_hashes)
                
                if success and img_hash:
                    downloaded += 1
                    category_hashes.add(img_hash)
                    global_hashes.add(img_hash)
                    print(f"  ✅ Downloaded {filename} (quality: good)")
                else:
                    print(f"  ❌ Failed to download {filename}")
                
                # Small delay to be respectful
                time.sleep(random.uniform(1.0, 2.0))
            
            print(f"✅ Downloaded {downloaded}/{num_images} images for {folder_name}")
            return downloaded, global_hashes
            
        except Exception as e:
            print(f"❌ Error searching for {search_query}: {e}")
            return 0, global_hashes

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
    global_hashes = set()  # Track all image hashes to avoid duplicates across categories
    
    # Scrape boxing images
    print("\n🥊 Scraping Boxing Images...")
    for folder, query in boxing_queries.items():
        folder_path = os.path.join("images", folder)
        downloaded, global_hashes = scrape_images(query, folder_path, 30, global_hashes)
        total_downloaded += downloaded
        print(f"📁 {folder}: {downloaded} images")
        print("-" * 30)
    
    # Scrape fencing images
    print("\n🤺 Scraping Fencing Images...")
    for folder, query in fencing_queries.items():
        folder_path = os.path.join("images", folder)
        downloaded, global_hashes = scrape_images(query, folder_path, 30, global_hashes)
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
