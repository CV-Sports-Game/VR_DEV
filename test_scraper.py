import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
import io
import time

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

def test_scrape():
    """Test scraping with just one query"""
    print(" Testing image scraper...")
    
    # Create test folder
    test_folder = "test_images"
    os.makedirs(test_folder, exist_ok=True)
    
    # Search for just a few images
    with DDGS() as ddgs:
        try:
            results = list(ddgs.images("boxing punch", max_results=3))
            
            for i, result in enumerate(results):
                image_url = result['image']
                filename = f"test_{i}.jpg"
                filepath = os.path.join(test_folder, filename)
                
                print(f"Downloading {filename}...")
                if download_image(image_url, filepath):
                    print(f" Downloaded {filename}")
                else:
                    print(f" Failed to download {filename}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_scrape() 