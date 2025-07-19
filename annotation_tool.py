import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import shutil

class SportsAnnotationTool:
    """
    A tool for labeling and annotating sports images and videos
    to create training datasets and validate AI analysis accuracy.
    """
    
    def __init__(self, images_dir: str = "images", videos_dir: str = "scraped_videos"):
        self.images_dir = images_dir
        self.videos_dir = videos_dir
        self.annotations_dir = "annotations"
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Define annotation categories
        self.sport_categories = {
            "boxing": [
                "jab", "cross", "hook", "uppercut", "straight_punch", "fast_punch",
                "block", "guard", "footwork", "defense", "combination"
            ],
            "fencing": [
                "lunge", "parry", "riposte", "block", "guard", "en_garde",
                "slide", "footwork", "attack", "defense", "counter"
            ]
        }
        
        # Quality ratings
        self.quality_ratings = ["excellent", "good", "fair", "poor"]
        
        # Form aspects to rate
        self.form_aspects = [
            "stance", "balance", "technique", "power", "speed", "accuracy", "safety"
        ]
    
    def create_image_annotation_csv(self) -> str:
        """
        Create a CSV file for annotating all images in the dataset
        
        Returns:
            Path to the created CSV file
        """
        csv_path = os.path.join(self.annotations_dir, "image_annotations.csv")
        
        # Get all image files
        image_files = []
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    rel_path = os.path.relpath(os.path.join(root, file), self.images_dir)
                    image_files.append(rel_path)
        
        # Create CSV with annotation columns
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header row
            header = [
                "image_path", "sport", "technique", "quality_rating",
                "stance_rating", "balance_rating", "technique_rating", 
                "power_rating", "speed_rating", "accuracy_rating", "safety_rating",
                "notes", "ai_prediction", "ai_confidence", "ai_accuracy"
            ]
            writer.writerow(header)
            
            # Data rows
            for image_path in image_files:
                # Extract sport and technique from folder structure
                parts = image_path.split(os.sep)
                sport = parts[0].split('_')[0] if parts else "unknown"
                technique = parts[0] if parts else "unknown"
                
                row = [
                    image_path, sport, technique,
                    "",  # quality_rating (to be filled manually)
                    "", "", "", "", "", "", "",  # form ratings (to be filled manually)
                    "",  # notes (to be filled manually)
                    "",  # ai_prediction (to be filled by AI analysis)
                    "",  # ai_confidence (to be filled by AI analysis)
                    ""   # ai_accuracy (to be calculated)
                ]
                writer.writerow(row)
        
        print(f"✅ Created image annotation CSV: {csv_path}")
        print(f"📊 Total images to annotate: {len(image_files)}")
        return csv_path
    
    def create_video_annotation_csv(self) -> str:
        """
        Create a CSV file for annotating all videos in the dataset
        
        Returns:
            Path to the created CSV file
        """
        csv_path = os.path.join(self.annotations_dir, "video_annotations.csv")
        
        # Get all video files
        video_files = []
        if os.path.exists(self.videos_dir):
            for file in os.listdir(self.videos_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(file)
        
        # Create CSV with annotation columns
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header row
            header = [
                "video_path", "sport", "technique", "duration", "quality_rating",
                "movement_sequence", "key_frames", "notes", "ai_analysis"
            ]
            writer.writerow(header)
            
            # Data rows
            for video_file in video_files:
                # Extract technique from filename (assuming format from video scraper)
                technique = video_file.split('_')[0] if '_' in video_file else "unknown"
                sport = "boxing" if technique.startswith("boxing") else "fencing"
                
                row = [
                    video_file, sport, technique,
                    "",  # duration (to be filled automatically)
                    "",  # quality_rating (to be filled manually)
                    "",  # movement_sequence (to be filled manually)
                    "",  # key_frames (to be filled manually)
                    "",  # notes (to be filled manually)
                    ""   # ai_analysis (to be filled by AI analysis)
                ]
                writer.writerow(row)
        
        print(f"✅ Created video annotation CSV: {csv_path}")
        print(f"📊 Total videos to annotate: {len(video_files)}")
        return csv_path
    
    def generate_annotation_instructions(self) -> str:
        """
        Generate annotation instructions for manual labeling
        
        Returns:
            Path to the instructions file
        """
        instructions_path = os.path.join(self.annotations_dir, "annotation_instructions.md")
        
        instructions = """# Sports Annotation Instructions

## Overview
This document provides instructions for manually annotating sports images and videos for AI training and validation.

## Image Annotation

### Quality Ratings
- **excellent**: Perfect form, clear technique, high quality image
- **good**: Good form with minor issues, clear technique
- **fair**: Some form issues, technique recognizable
- **poor**: Poor form, unclear technique, low quality

### Form Aspect Ratings (1-5 scale)
- **stance**: Body positioning and balance
- **balance**: Weight distribution and stability
- **technique**: Correct execution of the movement
- **power**: Force generation and delivery
- **speed**: Movement velocity and timing
- **accuracy**: Precision and target hitting
- **safety**: Safe execution and positioning

## Video Annotation

### Movement Sequence
Describe the sequence of movements in the video:
1. Starting position
2. Key movements
3. Ending position

### Key Frames
Identify important frames (frame numbers) where:
- Technique begins
- Peak of movement
- Technique ends

## Sport-Specific Guidelines

### Boxing
- **jab**: Quick straight punch with lead hand
- **cross**: Powerful straight punch with rear hand
- **hook**: Curved punch to the side
- **uppercut**: Upward punch from below
- **block**: Defensive movement to stop incoming punches
- **guard**: Defensive stance with hands protecting face

### Fencing
- **lunge**: Forward attack with extended arm and leg
- **parry**: Defensive blade movement to block attack
- **riposte**: Counter-attack after successful parry
- **en garde**: Ready stance with weapon extended
- **slide**: Footwork movement for positioning

## AI Validation
After manual annotation, compare with AI predictions:
- **ai_prediction**: What the AI identified
- **ai_confidence**: AI's confidence score (0-1)
- **ai_accuracy**: Whether AI prediction matches manual annotation (yes/no)

## Tips
- Be consistent in your ratings
- Focus on technique execution, not image/video quality
- Use notes field for specific observations
- Take breaks to maintain accuracy
"""
        
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        print(f"✅ Created annotation instructions: {instructions_path}")
        return instructions_path
    
    def create_validation_dataset(self, ai_analysis_results: Dict[str, Any]) -> str:
        """
        Create a validation dataset by comparing AI predictions with manual annotations
        
        Args:
            ai_analysis_results: Results from AI analysis
            
        Returns:
            Path to the validation dataset
        """
        validation_path = os.path.join(self.annotations_dir, "validation_dataset.json")
        
        validation_data = {
            "dataset_info": {
                "total_samples": 0,
                "ai_accuracy": 0.0,
                "sport_distribution": {},
                "technique_distribution": {}
            },
            "samples": []
        }
        
        # This would be populated with actual comparison data
        # For now, create a template structure
        
        with open(validation_path, 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        print(f"✅ Created validation dataset template: {validation_path}")
        return validation_path
    
    def generate_annotation_summary(self) -> str:
        """
        Generate a summary of annotation progress and statistics
        
        Returns:
            Path to the summary file
        """
        summary_path = os.path.join(self.annotations_dir, "annotation_summary.md")
        
        # Count files
        image_count = sum(1 for root, dirs, files in os.walk(self.images_dir) 
                         for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
        
        video_count = 0
        if os.path.exists(self.videos_dir):
            video_count = sum(1 for file in os.listdir(self.videos_dir) 
                            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')))
        
        summary = f"""# Annotation Summary

## Dataset Overview
- **Total Images**: {image_count}
- **Total Videos**: {video_count}
- **Total Samples**: {image_count + video_count}

## Annotation Files Created
- `image_annotations.csv` - For manual image annotation
- `video_annotations.csv` - For manual video annotation
- `annotation_instructions.md` - Detailed annotation guidelines
- `validation_dataset.json` - Template for AI validation

## Next Steps
1. **Manual Annotation**: Fill in the CSV files with expert ratings
2. **AI Analysis**: Run AI analysis on all samples
3. **Validation**: Compare AI predictions with manual annotations
4. **Training**: Use validated data for model training

## Quality Metrics
- **Inter-rater Reliability**: Have multiple experts annotate same samples
- **AI Accuracy**: Compare AI predictions with ground truth
- **Confidence Correlation**: Check if AI confidence correlates with accuracy

## Usage for MVP
- Use annotated data to validate AI performance
- Show accuracy metrics to investors
- Demonstrate data quality and expert validation
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"✅ Created annotation summary: {summary_path}")
        return summary_path

def main():
    """Create annotation system for the sports dataset"""
    print("🏷️ Creating Sports Annotation System")
    print("=" * 50)
    
    # Initialize annotation tool
    annotator = SportsAnnotationTool()
    
    # Create annotation files
    print("\n📋 Creating annotation files...")
    image_csv = annotator.create_image_annotation_csv()
    video_csv = annotator.create_video_annotation_csv()
    instructions = annotator.generate_annotation_instructions()
    validation = annotator.create_validation_dataset({})
    summary = annotator.generate_annotation_summary()
    
    print(f"\n🎉 Annotation system created successfully!")
    print(f"📁 All files saved in: {annotator.annotations_dir}")
    print(f"\n📊 Files created:")
    print(f"  - {image_csv}")
    print(f"  - {video_csv}")
    print(f"  - {instructions}")
    print(f"  - {validation}")
    print(f"  - {summary}")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Manually annotate images/videos using the CSV files")
    print(f"  2. Run AI analysis on all samples")
    print(f"  3. Compare AI predictions with manual annotations")
    print(f"  4. Generate validation reports for investors")

if __name__ == "__main__":
    main() 