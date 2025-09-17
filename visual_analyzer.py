import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import argparse
from train_model import SimpleCNN, ImagePoseDataset
import mediapipe as mp

class VisualSportsAnalyzer:
    def __init__(self, model_path="image_model.pth", label_mapping=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained model
        self.model = SimpleCNN(num_classes=16)
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
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        # Initialize MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        
        # Coaching feedback database
        self.feedback_db = {
            'boxing_punch': {
                'good': "Great punch form! Keep your guard up.",
                'tips': "Focus on rotating your hips and extending your arm fully.",
                'description': "Throwing a punch with proper form"
            },
            'boxing_guard': {
                'good': "Solid guard position! You're well protected.",
                'tips': "Keep your hands close to your face and elbows in.",
                'description': "Maintaining defensive guard position"
            },
            'boxing_hook': {
                'good': "Excellent hook! Good power and technique.",
                'tips': "Pivot on your front foot and rotate your body.",
                'description': "Executing a hook punch"
            },
            'boxing_uppercut': {
                'good': "Great uppercut! Good upward motion.",
                'tips': "Drive from your legs and keep it tight.",
                'description': "Throwing an uppercut punch"
            },
            'fencing_lunge': {
                'good': "Excellent lunge! Good extension and balance.",
                'tips': "Push off with your back leg and maintain posture.",
                'description': "Performing a fencing lunge attack"
            },
            'fencing_parry': {
                'good': "Good parry! You're defending well.",
                'tips': "Keep your blade in the correct line and move efficiently.",
                'description': "Executing a defensive parry"
            },
            'fencing_guard': {
                'good': "Solid guard position! Ready for action.",
                'tips': "Keep your blade in the correct line and stay balanced.",
                'description': "Maintaining fencing guard position"
            },
            'fencing_en_garde': {
                'good': "Perfect en garde! Ready to fight.",
                'tips': "Keep your feet shoulder-width apart and blade ready.",
                'description': "In en garde position"
            }
        }
    
    def detect_pose_landmarks(self, image):
        """Detect pose landmarks using MediaPipe."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None
    
    def get_bounding_box(self, landmarks):
        """Get bounding box from pose landmarks."""
        if not landmarks:
            return None
        
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add some padding
        padding = 0.1
        width = max_x - min_x
        height = max_y - min_y
        
        min_x = max(0, min_x - padding * width)
        max_x = min(1, max_x + padding * width)
        min_y = max(0, min_y - padding * height)
        max_y = min(1, max_y + padding * height)
        
        return (min_x, min_y, max_x, max_y)
    
    def analyze_image(self, image_path):
        """Analyze a single image and provide feedback."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Get pose landmarks
            landmarks = self.detect_pose_landmarks(image_cv)
            
            # Get prediction
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
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
                'landmarks': landmarks,
                'image_size': image.size
            }
            
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
    
    def generate_feedback(self, pose_name, confidence):
        """Generate coaching feedback based on pose and confidence."""
        feedback = {
            'pose': pose_name,
            'confidence': confidence,
            'description': '',
            'coaching': '',
            'tips': ''
        }
        
        # Get description
        if pose_name in self.feedback_db:
            feedback['description'] = self.feedback_db[pose_name]['description']
            if confidence > 0.7:
                feedback['coaching'] = self.feedback_db[pose_name]['good']
            else:
                feedback['coaching'] = self.feedback_db[pose_name]['tips']
        else:
            feedback['description'] = f"Detected {pose_name.replace('_', ' ')}"
            feedback['coaching'] = "Keep practicing your technique!"
        
        # Add confidence level
        if confidence > 0.8:
            feedback['confidence_level'] = "HIGH"
        elif confidence > 0.6:
            feedback['confidence_level'] = "GOOD"
        else:
            feedback['confidence_level'] = "LOW"
        
        return feedback
    
    def draw_analysis_on_image(self, image_path, output_path=None):
        """Draw pose analysis on image with bounding boxes and text."""
        analysis = self.analyze_image(image_path)
        
        if 'error' in analysis:
            print(f"❌ {analysis['error']}")
            return None
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default if not available
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Get image dimensions
        width, height = image.size
        
        # Draw bounding box if landmarks detected
        if analysis['landmarks']:
            bbox = self.get_bounding_box(analysis['landmarks'])
            if bbox:
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        
        # Draw pose landmarks
        if analysis['landmarks']:
            for landmark in analysis['landmarks']:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                draw.ellipse([x-3, y-3, x+3, y+3], fill=(255, 0, 0))
        
        # Draw text overlay
        feedback = analysis['feedback']
        
        # Background for text
        text_bg_height = 200
        draw.rectangle([0, height - text_bg_height, width, height], fill=(0, 0, 0, 180))
        
        # Text content
        y_offset = height - text_bg_height + 10
        
        # Pose name and confidence
        pose_text = f"{feedback['pose'].replace('_', ' ').title()}"
        conf_text = f"Confidence: {feedback['confidence']:.1%} ({feedback['confidence_level']})"
        
        draw.text((10, y_offset), pose_text, fill=(255, 255, 255), font=font_large)
        draw.text((10, y_offset + 30), conf_text, fill=(255, 255, 255), font=font_medium)
        
        # Description
        draw.text((10, y_offset + 55), feedback['description'], fill=(255, 255, 255), font=font_small)
        
        # Coaching feedback
        draw.text((10, y_offset + 75), feedback['coaching'], fill=(255, 255, 255), font=font_small)
        
        # Save or return image
        if output_path:
            image.save(output_path)
            print(f"✅ Saved visual analysis: {output_path}")
        
        return image
    
    def analyze_video_with_overlay(self, video_path, output_path=None, sample_rate=30):
        """Analyze video with real-time overlay."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        analyses = []
        
        print(f"📹 Analyzing video with overlay: {video_path}")
        print(f"   Duration: {total_frames/fps:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every nth frame
            if frame_count % sample_rate == 0:
                # Convert to PIL for analysis
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Save temp image
                temp_path = f"temp_frame_{frame_count}.jpg"
                pil_image.save(temp_path)
                
                # Analyze
                analysis = self.analyze_image(temp_path)
                analysis['frame'] = frame_count
                analysis['timestamp'] = frame_count / fps
                analyses.append(analysis)
                
                # Draw overlay on frame
                if 'error' not in analysis:
                    frame = self.draw_overlay_on_frame(frame, analysis)
                
                # Clean up
                os.remove(temp_path)
            
            # Write frame
            if output_path:
                out.write(frame)
            
            # Show progress
            if frame_count % (sample_rate * 10) == 0:
                progress = (frame_count / total_frames) * 100
                pose = analysis.get('pose', 'unknown') if 'analysis' in locals() else 'unknown'
                print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - {pose}")
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
            print(f"✅ Saved video analysis: {output_path}")
        
        return analyses
    
    def draw_overlay_on_frame(self, frame, analysis):
        """Draw analysis overlay on video frame."""
        height, width = frame.shape[:2]
        
        # Draw bounding box if landmarks detected
        if analysis.get('landmarks'):
            bbox = self.get_bounding_box(analysis['landmarks'])
            if bbox:
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw pose landmarks
        if analysis.get('landmarks'):
            for landmark in analysis['landmarks']:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        # Draw text overlay
        feedback = analysis['feedback']
        
        # Background rectangle
        cv2.rectangle(frame, (0, height - 150), (width, height), (0, 0, 0), -1)
        
        # Text
        y_offset = height - 130
        
        # Pose name
        pose_text = f"{feedback['pose'].replace('_', ' ').title()}"
        cv2.putText(frame, pose_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Confidence
        conf_text = f"Confidence: {feedback['confidence']:.1%} ({feedback['confidence_level']})"
        cv2.putText(frame, conf_text, (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Description
        cv2.putText(frame, feedback['description'], (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Coaching
        cv2.putText(frame, feedback['coaching'], (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Visual VR Sports Analyzer - AI Coach')
    parser.add_argument('--image', type=str, help='Path to image file to analyze')
    parser.add_argument('--video', type=str, help='Path to video file to analyze')
    parser.add_argument('--output', type=str, help='Output path for visual analysis')
    parser.add_argument('--sample-rate', type=int, default=30, 
                       help='Sample every N frames for video analysis (default: 30)')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demo on test images')
    
    args = parser.parse_args()
    
    analyzer = VisualSportsAnalyzer()
    
    print("🎯 Visual VR Sports Analyzer - AI Coach")
    print("=" * 45)
    
    # Check if model exists
    if not os.path.exists("image_model.pth"):
        print("❌ Model not found! Please train the model first using train_model.py")
        return
    
    # Handle different analysis modes
    if args.image:
        print(f"\n📸 Analyzing image: {args.image}")
        output_path = args.output or f"visual_analysis_{os.path.basename(args.image)}"
        result = analyzer.draw_analysis_on_image(args.image, output_path)
        
        if result:
            print(f"✅ Visual analysis complete!")
    
    elif args.video:
        print(f"\n📹 Analyzing video: {args.video}")
        output_path = args.output or f"visual_analysis_{os.path.basename(args.video)}"
        result = analyzer.analyze_video_with_overlay(args.video, output_path, args.sample_rate)
        
        if result:
            print(f"✅ Video analysis complete! Analyzed {len(result)} frames")
    
    elif args.demo:
        # Demo with test images
        test_images = ["test_images/test_0.jpg", "test_images/test_1.jpg"]
        
        for i, img_path in enumerate(test_images):
            if os.path.exists(img_path):
                print(f"\n📸 Analyzing: {img_path}")
                output_path = f"demo_analysis_{i+1}.jpg"
                result = analyzer.draw_analysis_on_image(img_path, output_path)
                
                if result:
                    print(f"✅ Demo analysis saved: {output_path}")
            else:
                print(f"⚠️ Test image not found: {img_path}")
    
    else:
        print("\n💡 Usage examples:")
        print("   python3 visual_analyzer.py --image path/to/image.jpg")
        print("   python3 visual_analyzer.py --video path/to/video.mp4")
        print("   python3 visual_analyzer.py --demo")
        print("\n📸 Running demo on test images...")
        
        test_images = ["test_images/test_0.jpg", "test_images/test_1.jpg"]
        
        for i, img_path in enumerate(test_images):
            if os.path.exists(img_path):
                print(f"\n📸 Analyzing: {img_path}")
                output_path = f"demo_analysis_{i+1}.jpg"
                result = analyzer.draw_analysis_on_image(img_path, output_path)
                
                if result:
                    print(f"✅ Demo analysis saved: {output_path}")
            else:
                print(f"⚠️ Test image not found: {img_path}")

if __name__ == "__main__":
    main()
