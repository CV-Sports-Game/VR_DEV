#!/usr/bin/env python3
"""
Google Cloud Storage Upload Script
Uploads videos and project files to GCS bucket 'still_data'
"""

import os
import glob
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
import argparse

class GCSUploader:
    def __init__(self, bucket_name="still_data"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    def upload_file(self, source_file, destination_blob_name):
        """Upload a single file to GCS."""
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file)
            print(f"✅ Uploaded: {source_file} → gs://{self.bucket_name}/{destination_blob_name}")
            return True
        except GoogleCloudError as e:
            print(f"❌ Failed to upload {source_file}: {e}")
            return False
    
    def upload_directory(self, local_dir, gcs_prefix=""):
        """Upload all files from a directory to GCS."""
        if not os.path.exists(local_dir):
            print(f"❌ Directory not found: {local_dir}")
            return False
        
        success_count = 0
        total_count = 0
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # Create GCS path
                relative_path = os.path.relpath(local_path, local_dir)
                gcs_path = os.path.join(gcs_prefix, relative_path).replace("\\", "/")
                
                if self.upload_file(local_path, gcs_path):
                    success_count += 1
                total_count += 1
        
        print(f"\n📊 Upload Summary: {success_count}/{total_count} files uploaded successfully")
        return success_count == total_count
    
    def upload_videos(self, video_dir="scraped_videos", gcs_prefix="training_videos"):
        """Upload video files specifically."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []
        
        for ext in video_extensions:
            videos.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        
        if not videos:
            print(f"❌ No video files found in {video_dir}")
            return False
        
        print(f"🎬 Found {len(videos)} video files to upload...")
        
        success_count = 0
        for video in videos:
            filename = os.path.basename(video)
            gcs_path = f"{gcs_prefix}/{filename}"
            
            if self.upload_file(video, gcs_path):
                success_count += 1
        
        print(f"✅ Uploaded {success_count}/{len(videos)} videos to gs://{self.bucket_name}/{gcs_prefix}/")
        return success_count == len(videos)
    
    def upload_models(self, model_dir=".", gcs_prefix="models"):
        """Upload trained models."""
        model_files = glob.glob(os.path.join(model_dir, "*.pth"))
        
        if not model_files:
            print("❌ No model files (*.pth) found")
            return False
        
        print(f"🤖 Found {len(model_files)} model files to upload...")
        
        success_count = 0
        for model in model_files:
            filename = os.path.basename(model)
            gcs_path = f"{gcs_prefix}/{filename}"
            
            if self.upload_file(model, gcs_path):
                success_count += 1
        
        print(f"✅ Uploaded {success_count}/{len(model_files)} models to gs://{self.bucket_name}/{gcs_prefix}/")
        return success_count == len(model_files)
    
    def list_bucket_contents(self, prefix=""):
        """List all files in the bucket."""
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        files = []
        
        for blob in blobs:
            files.append(blob.name)
        
        if files:
            print(f"\n📁 Contents of gs://{self.bucket_name}/{prefix}:")
            for file in files:
                print(f"  📄 {file}")
        else:
            print(f"\n📁 No files found in gs://{self.bucket_name}/{prefix}")
        
        return files

def main():
    parser = argparse.ArgumentParser(description='Upload files to Google Cloud Storage')
    parser.add_argument('--videos', action='store_true', help='Upload video files from scraped_videos/')
    parser.add_argument('--models', action='store_true', help='Upload model files (*.pth)')
    parser.add_argument('--images', action='store_true', help='Upload image dataset from images/')
    parser.add_argument('--all', action='store_true', help='Upload everything')
    parser.add_argument('--list', action='store_true', help='List bucket contents')
    parser.add_argument('--file', type=str, help='Upload a specific file')
    parser.add_argument('--destination', type=str, help='GCS destination path for specific file')
    
    args = parser.parse_args()
    
    uploader = GCSUploader()
    
    if args.list:
        uploader.list_bucket_contents()
        return
    
    if args.file:
        if not args.destination:
            args.destination = os.path.basename(args.file)
        uploader.upload_file(args.file, args.destination)
        return
    
    if args.all or args.videos:
        print("🎬 Uploading videos...")
        uploader.upload_videos()
    
    if args.all or args.models:
        print("🤖 Uploading models...")
        uploader.upload_models()
    
    if args.all or args.images:
        print("🖼️ Uploading images...")
        uploader.upload_directory("images", "training_images")
    
    if not any([args.videos, args.models, args.images, args.all, args.file, args.list]):
        print("💡 Usage examples:")
        print("  python3 upload_to_gcs.py --videos          # Upload videos")
        print("  python3 upload_to_gcs.py --models          # Upload trained models")
        print("  python3 upload_to_gcs.py --images          # Upload image dataset")
        print("  python3 upload_to_gcs.py --all             # Upload everything")
        print("  python3 upload_to_gcs.py --list            # List bucket contents")
        print("  python3 upload_to_gcs.py --file video.mp4  # Upload specific file")

if __name__ == "__main__":
    main() 