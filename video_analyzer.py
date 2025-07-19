import os
import cv2
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from gemini_sports_analyzer import GeminiSportsAnalyzer

class VideoAnalyzer:
    """
    A video analysis system that extracts frames from sports videos
    and analyzes them using AI for movement sequence analysis.
    """
    
    def __init__(self, gemini_api_key: str):
        """
        Initialize the video analyzer
        
        Args:
            gemini_api_key: API key for Gemini Sports Analyzer
        """
        self.gemini_analyzer = GeminiSportsAnalyzer(gemini_api_key)
        self.output_dir = "video_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_frames(self, video_path: str, frame_interval: int = 30, max_frames: int = 10) -> List[str]:
        """
        Extract frames from a video at specified intervals
        
        Args:
            video_path: Path to the video file
            frame_interval: Extract every Nth frame (default: 30 = 1 frame per second at 30fps)
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of paths to extracted frame images
        """
        print(f"🎬 Extracting frames from {os.path.basename(video_path)}...")
        
        # Create frames directory for this video
        video_name = Path(video_path).stem
        frames_dir = os.path.join(self.output_dir, f"{video_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"  📊 Video info: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s duration")
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{extracted_count:04d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                
                # Save frame
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                extracted_count += 1
                
                print(f"  📸 Extracted frame {extracted_count}/{max_frames} (frame {frame_count})")
            
            frame_count += 1
        
        cap.release()
        print(f"✅ Extracted {len(frame_paths)} frames to {frames_dir}")
        return frame_paths
    
    def analyze_video_sequence(self, video_path: str, sport_type: str = "detailed", 
                             frame_interval: int = 30, max_frames: int = 8) -> Dict[str, Any]:
        """
        Analyze a video by extracting frames and analyzing them with AI
        
        Args:
            video_path: Path to the video file
            sport_type: Type of sport for specialized analysis
            frame_interval: Frame extraction interval
            max_frames: Maximum frames to analyze
            
        Returns:
            Dictionary containing video analysis results
        """
        print(f"\n🎬 Analyzing video: {os.path.basename(video_path)}")
        print("=" * 60)
        
        # Extract frames
        frame_paths = self.extract_frames(video_path, frame_interval, max_frames)
        
        if not frame_paths:
            return {"error": "No frames extracted"}
        
        # Analyze each frame
        frame_analyses = []
        print(f"\n🤖 Analyzing {len(frame_paths)} frames with AI...")
        
        for i, frame_path in enumerate(frame_paths):
            print(f"  [{i+1}/{len(frame_paths)}] Analyzing {os.path.basename(frame_path)}...")
            
            try:
                analysis = self.gemini_analyzer.analyze_single_image(frame_path, sport_type)
                frame_analyses.append({
                    "frame_number": i,
                    "frame_path": frame_path,
                    "analysis": analysis
                })
                
                # Add delay to respect API rate limits
                time.sleep(1.5)
                
            except Exception as e:
                print(f"  ❌ Error analyzing frame {i}: {e}")
                continue
        
        # Generate sequence analysis
        if frame_analyses:
            sequence_report = self.generate_video_sequence_report(frame_analyses, video_path)
            return sequence_report
        else:
            return {"error": "No successful frame analyses"}
    
    def generate_video_sequence_report(self, frame_analyses: List[Dict], video_path: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report from video frame analyses
        
        Args:
            frame_analyses: List of frame analysis results
            video_path: Path to the original video
            
        Returns:
            Dictionary containing video sequence analysis
        """
        print(f"\n📊 Generating video sequence report...")
        
        # Extract analysis data
        movement_types = []
        confidence_scores = []
        form_qualities = []
        all_technique_feedback = []
        all_improvement_suggestions = []
        
        for frame_data in frame_analyses:
            analysis = frame_data["analysis"]
            movement_types.append(analysis.movement_type)
            confidence_scores.append(analysis.confidence_score)
            form_qualities.append(analysis.form_quality)
            all_technique_feedback.extend(analysis.technique_feedback)
            all_improvement_suggestions.extend(analysis.improvement_suggestions)
        
        # Calculate statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Most common movement type
        from collections import Counter
        movement_counter = Counter(movement_types)
        most_common_movement = movement_counter.most_common(1)[0][0] if movement_counter else "unknown"
        
        # Form quality distribution
        quality_counter = Counter(form_qualities)
        
        # Remove duplicates and get top suggestions
        unique_technique_feedback = list(set(all_technique_feedback))[:5]
        unique_improvement_suggestions = list(set(all_improvement_suggestions))[:5]
        
        # Create report
        report = {
            "video_path": video_path,
            "video_name": os.path.basename(video_path),
            "total_frames_analyzed": len(frame_analyses),
            "primary_movement_type": most_common_movement,
            "average_confidence": round(avg_confidence, 3),
            "form_quality_distribution": dict(quality_counter),
            "movement_sequence": [
                {
                    "frame": frame_data["frame_number"],
                    "movement_type": frame_data["analysis"].movement_type,
                    "confidence": frame_data["analysis"].confidence_score,
                    "form_quality": frame_data["analysis"].form_quality
                }
                for frame_data in frame_analyses
            ],
            "top_technique_feedback": unique_technique_feedback,
            "top_improvement_suggestions": unique_improvement_suggestions,
            "frame_analyses": [
                {
                    "frame_number": frame_data["frame_number"],
                    "movement_type": frame_data["analysis"].movement_type,
                    "confidence": frame_data["analysis"].confidence_score,
                    "form_quality": frame_data["analysis"].form_quality,
                    "technique_feedback": frame_data["analysis"].technique_feedback[:2],
                    "improvement_suggestions": frame_data["analysis"].improvement_suggestions[:2]
                }
                for frame_data in frame_analyses
            ]
        }
        
        return report
    
    def save_video_analysis(self, report: Dict[str, Any], output_path: str = None):
        """Save video analysis report to JSON file"""
        if output_path is None:
            video_name = Path(report["video_path"]).stem
            output_path = os.path.join(self.output_dir, f"{video_name}_analysis.json")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Video analysis saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
    
    def print_video_summary(self, report: Dict[str, Any]):
        """Print a formatted summary of video analysis results"""
        print(f"\n📊 Video Analysis Summary:")
        print(f"  Video: {report['video_name']}")
        print(f"  Frames Analyzed: {report['total_frames_analyzed']}")
        print(f"  Primary Movement: {report['primary_movement_type']}")
        print(f"  Average Confidence: {report['average_confidence']:.2f}")
        print(f"  Form Quality Distribution: {report['form_quality_distribution']}")
        
        print(f"\n🎯 Top Technique Feedback:")
        for i, feedback in enumerate(report['top_technique_feedback'][:3], 1):
            print(f"  {i}. {feedback}")
        
        print(f"\n💡 Top Improvement Suggestions:")
        for i, suggestion in enumerate(report['top_improvement_suggestions'][:3], 1):
            print(f"  {i}. {suggestion}")

def main():
    """Test the video analyzer"""
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        return
    
    # Initialize analyzer
    analyzer = VideoAnalyzer(api_key)
    
    # Test with a video if available
    test_video = "scraped_videos/sample_video.mp4"
    
    if os.path.exists(test_video):
        print(f"🧪 Testing Video Analyzer with {test_video}")
        report = analyzer.analyze_video_sequence(test_video, "detailed", frame_interval=30, max_frames=5)
        
        if "error" not in report:
            analyzer.print_video_summary(report)
            analyzer.save_video_analysis(report)
        else:
            print(f"❌ Analysis failed: {report['error']}")
    else:
        print("❌ No test video found. Please run the video scraper first or provide a test video.")

if __name__ == "__main__":
    main() 