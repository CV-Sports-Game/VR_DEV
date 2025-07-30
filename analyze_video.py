import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
from train_model import SimpleCNN, ImagePoseDataset

class SportsAnalyzer:
    def __init__(self, model_path="image_model.pth", label_mapping=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained model
        self.model = SimpleCNN(num_classes=16)  # 16 pose classes
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Label mapping (same as training)
        self.label_mapping = label_mapping or {
            'fencing_footwork': 0, 'fencing_riposte': 1, 'boxing_uppercut': 2, 
            'fencing_slide': 3, 'fencing_en_garde': 4, 'boxing_hook': 5, 
            'boxing_punch': 6, 'fencing_guard': 7, 'fencing_block': 8, 
            'boxing_straight_punch': 9, 'fencing_lunge': 10, 'boxing_block': 11, 
            'fencing_parry': 12, 'boxing_footwork': 13, 'boxing_fast_punch': 14, 
            'boxing_guard': 15
        }
        
        # Reverse mapping for readable output
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        # Coaching feedback database
        self.feedback_db = {
            'boxing_punch': {
                'good': "Great punch form! Keep your guard up.",
                'tips': "Focus on rotating your hips and extending your arm fully."
            },
            'boxing_guard': {
                'good': "Solid guard position! You're well protected.",
                'tips': "Keep your hands close to your face and elbows in."
            },
            'fencing_lunge': {
                'good': "Excellent lunge! Good extension and balance.",
                'tips': "Push off with your back leg and maintain posture."
            },
            'fencing_parry': {
                'good': "Good parry! You're defending well.",
                'tips': "Keep your blade in the correct line and move efficiently."
            }
        }
    
    def analyze_image(self, image_path):
        """Analyze a single image and provide feedback."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get pose name
            pose_name = self.reverse_mapping[predicted_class]
            
            # Generate feedback
            feedback = self.generate_feedback(pose_name, confidence)
            
            return {
                'pose': pose_name,
                'confidence': confidence,
                'feedback': feedback,
                'all_probabilities': {self.reverse_mapping[i]: prob.item() for i, prob in enumerate(probabilities[0])}
            }
            
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
    
    def analyze_video(self, video_path, sample_rate=30, show_progress=True):
        """Analyze video frames and provide feedback."""
        if not os.path.exists(video_path):
            return {'error': f"Video file not found: {video_path}"}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f"Could not open video: {video_path}"}
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📹 Video Analysis: {video_path}")
        print(f"   Duration: {duration:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
        print(f"   Sampling every {sample_rate} frames...")
        
        frame_count = 0
        analyses = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample every nth frame
            if frame_count % sample_rate == 0:
                # Convert OpenCV frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Save temporary image for analysis
                temp_path = f"temp_frame_{frame_count}.jpg"
                pil_image.save(temp_path)
                
                # Analyze the frame
                analysis = self.analyze_image(temp_path)
                analysis['frame'] = frame_count
                analysis['timestamp'] = frame_count / fps if fps > 0 else 0
                analyses.append(analysis)
                
                # Show progress
                if show_progress:
                    progress = (frame_count / total_frames) * 100
                    pose = analysis.get('pose', 'unknown')
                    confidence = analysis.get('confidence', 0)
                    print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - {pose} ({confidence:.1%})")
                
                # Clean up temp file
                os.remove(temp_path)
            
            frame_count += 1
        
        cap.release()
        
        if analyses:
            summary = self.get_summary(analyses)
            return {
                'analyses': analyses,
                'summary': summary,
                'total_frames': total_frames,
                'analyzed_frames': len(analyses),
                'duration': duration
            }
        else:
            return {'error': "No frames were analyzed"}
    
    def generate_feedback(self, pose_name, confidence):
        """Generate coaching feedback based on pose and confidence."""
        feedback = f"Detected pose: {pose_name} (Confidence: {confidence:.2%})\n\n"
        
        if confidence > 0.8:
            feedback += "🎯 HIGH CONFIDENCE - Great form!\n"
        elif confidence > 0.6:
            feedback += "✅ GOOD CONFIDENCE - Keep practicing!\n"
        else:
            feedback += "⚠️ LOW CONFIDENCE - Consider adjusting your form.\n"
        
        # Add sport-specific feedback
        if pose_name in self.feedback_db:
            if confidence > 0.7:
                feedback += f"\n💡 {self.feedback_db[pose_name]['good']}"
            else:
                feedback += f"\n💡 {self.feedback_db[pose_name]['tips']}"
        
        # Add general tips based on sport
        if 'boxing' in pose_name:
            feedback += "\n🥊 Boxing Tip: Remember to keep your guard up and move your feet!"
        elif 'fencing' in pose_name:
            feedback += "\n🤺 Fencing Tip: Focus on proper blade work and footwork!"
        
        return feedback
    
    def get_summary(self, analyses):
        """Generate a summary of video analysis."""
        if not analyses:
            return "No analysis data available."
        
        # Count poses
        pose_counts = {}
        total_confidence = 0
        
        for analysis in analyses:
            pose = analysis.get('pose', 'unknown')
            confidence = analysis.get('confidence', 0)
            
            pose_counts[pose] = pose_counts.get(pose, 0) + 1
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(analyses)
        most_common_pose = max(pose_counts.items(), key=lambda x: x[1])
        
        summary = f"""
📊 ANALYSIS SUMMARY
==================
Total frames analyzed: {len(analyses)}
Average confidence: {avg_confidence:.2%}
Most common pose: {most_common_pose[0]} ({most_common_pose[1]} times)

Pose breakdown:
"""
        for pose, count in sorted(pose_counts.items()):
            percentage = (count / len(analyses)) * 100
            summary += f"  {pose}: {count} times ({percentage:.1f}%)\n"
        
        # Add coaching recommendations
        summary += "\n🎯 COACHING RECOMMENDATIONS:\n"
        if avg_confidence < 0.6:
            summary += "  • Overall confidence is low - focus on form fundamentals\n"
        if len(pose_counts) < 3:
            summary += "  • Limited pose variety - try different techniques\n"
        if most_common_pose[1] > len(analyses) * 0.7:
            summary += "  • Very repetitive - work on technique diversity\n"
        
        return summary

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='VR Sports Analyzer - AI Coach')
    parser.add_argument('--image', type=str, help='Path to image file to analyze')
    parser.add_argument('--video', type=str, help='Path to video file to analyze')
    parser.add_argument('--sample-rate', type=int, default=30, 
                       help='Sample every N frames for video analysis (default: 30)')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demo on test images')
    
    args = parser.parse_args()
    
    analyzer = SportsAnalyzer()
    
    print("🎯 VR Sports Analyzer - AI Coach")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists("image_model.pth"):
        print("❌ Model not found! Please train the model first using train_model.py")
        return
    
    # Handle different analysis modes
    if args.image:
        print(f"\n📸 Analyzing image: {args.image}")
        result = analyzer.analyze_image(args.image)
        
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            print(f"🎯 {result['feedback']}")
    
    elif args.video:
        print(f"\n📹 Analyzing video: {args.video}")
        result = analyzer.analyze_video(args.video, args.sample_rate)
        
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            print(f"\n{result['summary']}")
    
    elif args.demo:
        # Demo with test images
        test_images = ["test_images/test_0.jpg", "test_images/test_1.jpg"]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"\n📸 Analyzing: {img_path}")
                result = analyzer.analyze_image(img_path)
                
                if 'error' in result:
                    print(f"❌ {result['error']}")
                else:
                    print(f"🎯 {result['feedback']}")
            else:
                print(f"⚠️ Test image not found: {img_path}")
    
    else:
        # Default demo
        print("\n💡 Usage examples:")
        print("   python3 analyze_video.py --image path/to/image.jpg")
        print("   python3 analyze_video.py --video path/to/video.mp4")
        print("   python3 analyze_video.py --demo")
        print("\n📸 Running demo on test images...")
        
        test_images = ["test_images/test_0.jpg", "test_images/test_1.jpg"]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"\n📸 Analyzing: {img_path}")
                result = analyzer.analyze_image(img_path)
                
                if 'error' in result:
                    print(f"❌ {result['error']}")
                else:
                    print(f"🎯 {result['feedback']}")
            else:
                print(f"⚠️ Test image not found: {img_path}")

if __name__ == "__main__":
    main() 