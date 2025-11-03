import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
import json
from collections import deque
import mediapipe as mp
from train_model import SimpleCNN, ImagePoseDataset
from pose_transformer import PoseTransformer
from progress_tracker import ProgressTracker

class SportsAnalyzer:
    def __init__(self, image_model_path="image_model.pth", pose_model_path="pose_model.pth", 
                 use_sequence_model=True, sequence_len=30, label_mapping=None, 
                 enable_tracking=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_len = sequence_len
        self.use_sequence_model = use_sequence_model
        self.enable_tracking = enable_tracking
        
        # Initialize progress tracker if enabled
        self.progress_tracker = None
        if enable_tracking:
            self.progress_tracker = ProgressTracker()
        
        # Load the CNN model for image classification
        self.image_model = SimpleCNN(num_classes=16)  # 16 pose classes
        if os.path.exists(image_model_path):
            self.image_model.load_state_dict(torch.load(image_model_path, map_location=self.device))
        self.image_model.to(self.device)
        self.image_model.eval()
        
        # Load the PoseTransformer model for sequence analysis
        self.pose_model = None
        self.pose_label_map = None
        if use_sequence_model and os.path.exists(pose_model_path):
            # Try to load pose sequence model
            try:
                # Load label mapping from pose_data/labels.csv
                import pandas as pd
                pose_label_file = os.path.join("pose_data", "labels.csv")
                if os.path.exists(pose_label_file):
                    label_df = pd.read_csv(pose_label_file)
                    unique_labels = label_df['label'].unique()
                    self.pose_label_map = {label: idx for idx, label in enumerate(unique_labels)}
                    num_pose_classes = len(self.pose_label_map)
                else:
                    num_pose_classes = 7  # Default from labels.csv content
                
                self.pose_model = PoseTransformer(input_dim=99, model_dim=128, 
                                                  num_classes=num_pose_classes, seq_len=sequence_len)
                self.pose_model.load_state_dict(torch.load(pose_model_path, map_location=self.device))
                self.pose_model.to(self.device)
                self.pose_model.eval()
                print(f"✅ Loaded PoseTransformer model with {num_pose_classes} classes")
            except Exception as e:
                print(f"⚠️ Could not load PoseTransformer: {e}")
                self.pose_model = None
        
        # Initialize MediaPipe for pose landmark extraction
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Sliding window buffer for pose sequences
        self.pose_sequence_buffer = deque(maxlen=sequence_len)
        
        # Label mapping for image model (same as training)
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
        
        # Reverse mapping for pose sequence model
        self.pose_reverse_mapping = None
        if self.pose_label_map:
            self.pose_reverse_mapping = {v: k for k, v in self.pose_label_map.items()}
        
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
            },
            'roll': {
                'good': "Good rolling technique! Maintain momentum.",
                'tips': "Keep your body tight and protect your head during the roll."
            },
            'slash': {
                'good': "Powerful slash! Good arm extension.",
                'tips': "Follow through completely and maintain blade control."
            },
            'block1': {
                'good': "Solid block! Good defensive positioning.",
                'tips': "Keep your guard active and ready to counter."
            },
            'forward': {
                'good': "Good forward movement! Maintain balance.",
                'tips': "Keep your weight balanced and stay light on your feet."
            },
            'thrust': {
                'good': "Precise thrust! Good extension.",
                'tips': "Extend fully and maintain body alignment."
            },
            'jump': {
                'good': "Good jump! Maintain control in the air.",
                'tips': "Land softly and maintain balance upon landing."
            }
        }
    
    def extract_pose_landmarks_to_vector(self, frame):
        """
        Extract pose landmarks from frame and convert to 99-dimensional vector.
        MediaPipe has 33 landmarks, each with x, y, z coordinates = 99 dimensions.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)
        
        if results.pose_landmarks:
            # Extract 33 landmarks, each with x, y, z (99 dimensions total)
            pose_vector = []
            for landmark in results.pose_landmarks.landmark:
                pose_vector.extend([landmark.x, landmark.y, landmark.z])
            return np.array(pose_vector, dtype=np.float32)
        else:
            # Return zero vector if no pose detected
            return np.zeros(99, dtype=np.float32)
    
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
    
    def analyze_video(self, video_path, sample_rate=30, show_progress=True, use_both_models=True,
                     user_id='default', track_session=True):
        """
        Analyze video frames using both CNN (frame-level) and PoseTransformer (sequence-level).
        """
        if not os.path.exists(video_path):
            return {'error': f"Video file not found: {video_path}"}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f"Could not open video: {video_path}"}
        
        # Start tracking session if enabled
        session_id = None
        if track_session and self.progress_tracker:
            session_id = self.progress_tracker.start_session(
                user_id=user_id,
                sport_type='mixed',
                metadata={'video_path': video_path, 'sample_rate': sample_rate}
            )
            print(f"📊 Started tracking session: {session_id}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📹 Video Analysis: {video_path}")
        print(f"   Duration: {duration:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
        print(f"   Sampling every {sample_rate} frames...")
        if use_both_models and self.pose_model:
            print(f"   Using both CNN (frame-level) and PoseTransformer (sequence-level) models")
        else:
            print(f"   Using CNN (frame-level) model only")
        
        # Reset sequence buffer
        self.pose_sequence_buffer.clear()
        
        frame_count = 0
        analyses = []
        analyzed_frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract pose landmarks for sequence analysis
            pose_vector = self.extract_pose_landmarks_to_vector(frame)
            self.pose_sequence_buffer.append(pose_vector)
            
            # Sample every nth frame for analysis
            if frame_count % sample_rate == 0:
                analysis = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0
                }
                
                # Frame-level analysis using CNN
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                temp_path = f"temp_frame_{frame_count}.jpg"
                pil_image.save(temp_path)
                
                frame_analysis = self.analyze_image(temp_path)
                analysis.update(frame_analysis)
                
                # Sequence-level analysis using PoseTransformer
                if use_both_models and self.pose_model and len(self.pose_sequence_buffer) >= self.sequence_len:
                    sequence_analysis = self.analyze_pose_sequence()
                    analysis['sequence_analysis'] = sequence_analysis
                    
                    # Combine predictions if both models agree or disagree
                    if sequence_analysis:
                        analysis['combined_confidence'] = (
                            analysis.get('confidence', 0) + sequence_analysis.get('confidence', 0)
                        ) / 2
                
                analyses.append(analysis)
                analyzed_frame_count += 1
                
                # Track frame analysis if enabled
                if track_session and self.progress_tracker and session_id:
                    self.progress_tracker.add_frame_analysis(
                        session_id=session_id,
                        frame_number=frame_count,
                        timestamp=analysis.get('timestamp', 0),
                        pose_label=analysis.get('pose', 'unknown'),
                        confidence=analysis.get('confidence', 0),
                        model_type='cnn',
                        sequence_pose_label=sequence_analysis.get('pose') if sequence_analysis else None,
                        sequence_confidence=sequence_analysis.get('confidence') if sequence_analysis else None,
                        feedback=analysis.get('feedback', '')
                    )
                
                # Show progress
                if show_progress:
                    progress = (frame_count / total_frames) * 100
                    pose = analysis.get('pose', 'unknown')
                    confidence = analysis.get('confidence', 0)
                    seq_info = ""
                    if 'sequence_analysis' in analysis:
                        seq_pose = analysis['sequence_analysis'].get('pose', 'N/A')
                        seq_info = f" | Sequence: {seq_pose}"
                    print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - {pose} ({confidence:.1%}){seq_info}")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            frame_count += 1
        
        cap.release()
        
        # End tracking session if enabled
        if track_session and self.progress_tracker and session_id:
            session_metrics = self.progress_tracker.end_session(
                session_id=session_id,
                total_frames=total_frames,
                analyzed_frames=analyzed_frame_count
            )
            print(f"📊 Session metrics saved: {json.dumps(session_metrics, indent=2, default=str)}")
        
        if analyses:
            summary = self.get_summary(analyses)
            result = {
                'analyses': analyses,
                'summary': summary,
                'total_frames': total_frames,
                'analyzed_frames': len(analyses),
                'duration': duration,
                'models_used': {
                    'cnn': True,
                    'transformer': use_both_models and self.pose_model is not None
                }
            }
            
            if track_session and session_id:
                result['session_id'] = session_id
            
            return result
        else:
            return {'error': "No frames were analyzed"}
    
    def analyze_pose_sequence(self):
        """
        Analyze the current pose sequence buffer using PoseTransformer.
        """
        if len(self.pose_sequence_buffer) < self.sequence_len:
            return None
        
        try:
            # Convert buffer to tensor
            sequence = np.array(list(self.pose_sequence_buffer), dtype=np.float32)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.pose_model(sequence_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get pose name
            if self.pose_reverse_mapping:
                pose_name = self.pose_reverse_mapping.get(predicted_class, f"class_{predicted_class}")
            else:
                pose_name = f"class_{predicted_class}"
            
            return {
                'pose': pose_name,
                'confidence': confidence,
                'all_probabilities': {
                    self.pose_reverse_mapping.get(i, f"class_{i}"): prob.item() 
                    for i, prob in enumerate(probabilities[0])
                } if self.pose_reverse_mapping else {}
            }
        except Exception as e:
            return {'error': f"Sequence analysis failed: {str(e)}"}
    
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
        """Generate a summary of video analysis including both model results."""
        if not analyses:
            return "No analysis data available."
        
        # Count poses from CNN (frame-level)
        pose_counts = {}
        sequence_pose_counts = {}
        total_confidence = 0
        sequence_confidences = []
        
        for analysis in analyses:
            pose = analysis.get('pose', 'unknown')
            confidence = analysis.get('confidence', 0)
            
            pose_counts[pose] = pose_counts.get(pose, 0) + 1
            total_confidence += confidence
            
            # Sequence analysis results
            if 'sequence_analysis' in analysis:
                seq_analysis = analysis['sequence_analysis']
                seq_pose = seq_analysis.get('pose', 'unknown')
                seq_confidence = seq_analysis.get('confidence', 0)
                sequence_pose_counts[seq_pose] = sequence_pose_counts.get(seq_pose, 0) + 1
                sequence_confidences.append(seq_confidence)
        
        avg_confidence = total_confidence / len(analyses)
        most_common_pose = max(pose_counts.items(), key=lambda x: x[1]) if pose_counts else ('N/A', 0)
        
        summary = f"""
📊 ANALYSIS SUMMARY
==================
Total frames analyzed: {len(analyses)}
Average frame-level confidence: {avg_confidence:.2%}
Most common frame-level pose: {most_common_pose[0]} ({most_common_pose[1]} times)

Frame-level pose breakdown:
"""
        for pose, count in sorted(pose_counts.items()):
            percentage = (count / len(analyses)) * 100
            summary += f"  {pose}: {count} times ({percentage:.1f}%)\n"
        
        # Sequence-level analysis
        if sequence_pose_counts:
            avg_seq_confidence = sum(sequence_confidences) / len(sequence_confidences) if sequence_confidences else 0
            most_common_seq_pose = max(sequence_pose_counts.items(), key=lambda x: x[1])
            summary += f"\nSequence-level analysis (PoseTransformer):\n"
            summary += f"  Average sequence confidence: {avg_seq_confidence:.2%}\n"
            summary += f"  Most common sequence pose: {most_common_seq_pose[0]} ({most_common_seq_pose[1]} times)\n"
            summary += f"\nSequence pose breakdown:\n"
            for pose, count in sorted(sequence_pose_counts.items()):
                percentage = (count / len([a for a in analyses if 'sequence_analysis' in a])) * 100
                summary += f"  {pose}: {count} times ({percentage:.1f}%)\n"
        
        # Add coaching recommendations
        summary += "\n🎯 COACHING RECOMMENDATIONS:\n"
        if avg_confidence < 0.6:
            summary += "  • Overall confidence is low - focus on form fundamentals\n"
        if len(pose_counts) < 3:
            summary += "  • Limited pose variety - try different techniques\n"
        if most_common_pose[1] > len(analyses) * 0.7:
            summary += "  • Very repetitive - work on technique diversity\n"
        
        # Model agreement analysis
        if sequence_pose_counts:
            agreement_count = 0
            for analysis in analyses:
                if 'sequence_analysis' in analysis:
                    frame_pose = analysis.get('pose', '')
                    seq_pose = analysis['sequence_analysis'].get('pose', '')
                    # Check if poses are similar (same base category)
                    if frame_pose and seq_pose:
                        frame_sport = frame_pose.split('_')[0] if '_' in frame_pose else ''
                        seq_sport = seq_pose.split('_')[0] if '_' in seq_pose else ''
                        if frame_sport and seq_sport and frame_sport == seq_sport:
                            agreement_count += 1
            
            if len([a for a in analyses if 'sequence_analysis' in a]) > 0:
                agreement_rate = agreement_count / len([a for a in analyses if 'sequence_analysis' in a])
                summary += f"\n🤖 Model Agreement: {agreement_rate:.1%} ({agreement_count}/{len([a for a in analyses if 'sequence_analysis' in a])})\n"
                if agreement_rate < 0.5:
                    summary += "  • Models disagree - consider reviewing technique consistency\n"
        
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
    parser.add_argument('--image-model', type=str, default='image_model.pth',
                       help='Path to CNN image model (default: image_model.pth)')
    parser.add_argument('--pose-model', type=str, default='pose_model.pth',
                       help='Path to PoseTransformer model (default: pose_model.pth)')
    parser.add_argument('--no-sequence', action='store_true',
                       help='Disable sequence model and use only CNN')
    
    args = parser.parse_args()
    
    analyzer = SportsAnalyzer(
        image_model_path=args.image_model,
        pose_model_path=args.pose_model,
        use_sequence_model=not args.no_sequence
    )
    
    print("🎯 VR Sports Analyzer - AI Coach")
    print("=" * 40)
    
    # Check if models exist
    if not os.path.exists(args.image_model):
        print(f"⚠️ Image model not found at {args.image_model}!")
        print("   Training with mode 2 will create this model.")
        if not args.image:
            return
    elif not args.no_sequence and not os.path.exists(args.pose_model):
        print(f"⚠️ PoseTransformer model not found at {args.pose_model}!")
        print("   Training with mode 1 will create this model.")
        print("   Continuing with CNN model only...")
    
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
        result = analyzer.analyze_video(args.video, args.sample_rate, use_both_models=not args.no_sequence)
        
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            print(f"\n{result['summary']}")
            if result.get('models_used', {}).get('transformer'):
                print("\n✅ Used both CNN and PoseTransformer models for comprehensive analysis")
            else:
                print("\n✅ Used CNN model for frame-level analysis")
    
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