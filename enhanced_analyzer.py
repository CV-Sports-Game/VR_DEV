"""
Enhanced Sports Analyzer with Detailed Explanations and Visual Labels
====================================================================
This module provides:
- Detailed pose landmark analysis
- Body part-specific feedback
- Visual labeling of areas needing improvement
- Confidence scoring per body part
- Biomechanical analysis
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from analyze_video import SportsAnalyzer

@dataclass
class BodyPartAnalysis:
    """Analysis result for a specific body part."""
    name: str
    confidence: float
    position_score: float
    angle_score: float
    needs_improvement: bool
    feedback: str
    landmarks: List[Tuple[int, Tuple[float, float, float]]]  # (landmark_id, (x, y, z))
    recommended_angle: Optional[float] = None
    current_angle: Optional[float] = None

@dataclass
class DetailedPoseAnalysis:
    """Comprehensive pose analysis with body part breakdown."""
    pose_label: str
    overall_confidence: float
    body_parts: List[BodyPartAnalysis]
    key_issues: List[str]
    strengths: List[str]
    recommendations: List[str]
    visual_labels: List[Dict]  # For drawing on image

class EnhancedSportsAnalyzer(SportsAnalyzer):
    """Enhanced analyzer with detailed body part analysis and visual labels."""
    
    # MediaPipe landmark indices for body parts
    BODY_PARTS = {
        'left_shoulder': [11],
        'right_shoulder': [12],
        'left_elbow': [13],
        'right_elbow': [14],
        'left_wrist': [15],
        'right_wrist': [16],
        'left_hip': [23],
        'right_hip': [24],
        'left_knee': [25],
        'right_knee': [26],
        'left_ankle': [27],
        'right_ankle': [28],
        'head': [0],
        'neck': [1, 2],
        'spine': [23, 24, 11, 12],
        'left_arm': [11, 13, 15],
        'right_arm': [12, 14, 16],
        'left_leg': [23, 25, 27],
        'right_leg': [24, 26, 28]
    }
    
    # Pose-specific requirements (angles in degrees)
    POSE_REQUIREMENTS = {
        'boxing_punch': {
            'left_arm': {'min_angle': 160, 'max_angle': 180, 'description': 'Full extension'},
            'right_arm': {'min_angle': 45, 'max_angle': 90, 'description': 'Guard position'},
            'spine': {'min_angle': 170, 'max_angle': 180, 'description': 'Straight alignment'},
            'left_leg': {'min_angle': 160, 'max_angle': 180, 'description': 'Stable base'},
        },
        'boxing_guard': {
            'left_arm': {'min_angle': 45, 'max_angle': 90, 'description': 'Guard up'},
            'right_arm': {'min_angle': 45, 'max_angle': 90, 'description': 'Guard up'},
            'spine': {'min_angle': 170, 'max_angle': 180, 'description': 'Upright'},
            'head': {'position': 'protected', 'description': 'Behind guard'},
        },
        'fencing_lunge': {
            'right_leg': {'min_angle': 90, 'max_angle': 120, 'description': 'Bent front leg'},
            'left_leg': {'min_angle': 160, 'max_angle': 180, 'description': 'Extended back leg'},
            'right_arm': {'min_angle': 160, 'max_angle': 180, 'description': 'Extended arm'},
            'spine': {'min_angle': 150, 'max_angle': 180, 'description': 'Forward lean'},
        },
        'fencing_en_garde': {
            'right_arm': {'min_angle': 90, 'max_angle': 120, 'description': 'Blade ready'},
            'left_arm': {'min_angle': 45, 'max_angle': 90, 'description': 'Balanced'},
            'spine': {'min_angle': 170, 'max_angle': 180, 'description': 'Upright'},
            'left_leg': {'min_angle': 160, 'max_angle': 180, 'description': 'Stable base'},
        }
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points in degrees.
        point2 is the vertex.
        """
        # Convert to numpy arrays
        p1 = np.array([point1[0], point1[1]])
        p2 = np.array([point2[0], point2[1]])
        p3 = np.array([point3[0], point3[1]])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def analyze_body_part(self, landmarks, pose_label: str, body_part_name: str) -> BodyPartAnalysis:
        """Analyze a specific body part for the given pose."""
        if body_part_name not in self.BODY_PARTS:
            return None
        
        landmark_indices = self.BODY_PARTS[body_part_name]
        
        # Get landmark positions
        part_landmarks = []
        for idx in landmark_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                part_landmarks.append((idx, (lm.x, lm.y, lm.z)))
        
        if not part_landmarks:
            return None
        
        # Get pose requirements
        requirements = self.POSE_REQUIREMENTS.get(pose_label, {}).get(body_part_name, {})
        
        # Calculate angles if needed
        current_angle = None
        angle_score = 1.0
        position_score = 1.0
        
        if len(part_landmarks) >= 3 and 'min_angle' in requirements:
            # Calculate angle for 3-point joints (elbow, knee, etc.)
            p1_idx, p1_coords = part_landmarks[0]
            p2_idx, p2_coords = part_landmarks[1] if len(part_landmarks) > 1 else part_landmarks[0]
            p3_idx, p3_coords = part_landmarks[2] if len(part_landmarks) > 2 else part_landmarks[1]
            
            current_angle = self.calculate_angle(
                (p1_coords[0], p1_coords[1]),
                (p2_coords[0], p2_coords[1]),
                (p3_coords[0], p3_coords[1])
            )
            
            min_angle = requirements.get('min_angle', 0)
            max_angle = requirements.get('max_angle', 180)
            
            if min_angle <= current_angle <= max_angle:
                angle_score = 1.0
            else:
                # Calculate score based on distance from ideal range
                if current_angle < min_angle:
                    distance = min_angle - current_angle
                    ideal_range = max_angle - min_angle
                    angle_score = max(0, 1 - (distance / ideal_range))
                else:
                    distance = current_angle - max_angle
                    ideal_range = max_angle - min_angle
                    angle_score = max(0, 1 - (distance / ideal_range))
        
        # Calculate overall confidence
        confidence = (angle_score + position_score) / 2
        
        # Determine if improvement is needed
        needs_improvement = confidence < 0.7
        
        # Generate feedback
        feedback = ""
        if needs_improvement:
            if current_angle is not None:
                recommended_angle = (requirements.get('min_angle', 0) + requirements.get('max_angle', 180)) / 2
                if current_angle < recommended_angle:
                    feedback = f"Extend {body_part_name.replace('_', ' ')} more. Current: {current_angle:.1f}°, Target: ~{recommended_angle:.1f}°"
                else:
                    feedback = f"Reduce {body_part_name.replace('_', ' ')} angle. Current: {current_angle:.1f}°, Target: ~{recommended_angle:.1f}°"
            else:
                feedback = f"Adjust {body_part_name.replace('_', ' ')} position. {requirements.get('description', '')}"
        else:
            feedback = f"Good {body_part_name.replace('_', ' ')} position!"
        
        return BodyPartAnalysis(
            name=body_part_name,
            confidence=confidence,
            position_score=position_score,
            angle_score=angle_score,
            needs_improvement=needs_improvement,
            feedback=feedback,
            landmarks=part_landmarks,
            recommended_angle=requirements.get('min_angle'),
            current_angle=current_angle
        )
    
    def analyze_pose_detailed(self, image, pose_label: str, landmarks=None) -> DetailedPoseAnalysis:
        """
        Perform detailed analysis of pose with body part breakdown.
        
        Args:
            image: Input image (OpenCV format)
            pose_label: Detected pose label
            landmarks: MediaPipe landmarks (if None, will detect)
        
        Returns:
            DetailedPoseAnalysis with body part breakdown
        """
        # Detect landmarks if not provided
        if landmarks is None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(image_rgb)
            if not results.pose_landmarks:
                return None
            landmarks = results.pose_landmarks
        
        # Analyze each body part
        body_parts = []
        for body_part_name in self.BODY_PARTS.keys():
            analysis = self.analyze_body_part(landmarks, pose_label, body_part_name)
            if analysis:
                body_parts.append(analysis)
        
        # Calculate overall confidence
        if body_parts:
            overall_confidence = np.mean([bp.confidence for bp in body_parts])
        else:
            overall_confidence = 0.5
        
        # Identify key issues
        key_issues = [
            bp.feedback for bp in body_parts 
            if bp.needs_improvement and bp.confidence < 0.6
        ]
        
        # Identify strengths
        strengths = [
            f"{bp.name.replace('_', ' ').title()} is well positioned"
            for bp in body_parts 
            if not bp.needs_improvement and bp.confidence > 0.8
        ]
        
        # Generate recommendations
        recommendations = []
        if key_issues:
            recommendations.extend(key_issues[:3])  # Top 3 issues
        
        # Generate visual labels
        visual_labels = []
        for bp in body_parts:
            if bp.needs_improvement:
                for landmark_id, coords in bp.landmarks:
                    visual_labels.append({
                        'type': 'circle' if bp.confidence < 0.5 else 'warning',
                        'position': (int(coords[0] * image.shape[1]), int(coords[1] * image.shape[0])),
                        'color': (0, 0, 255) if bp.confidence < 0.5 else (0, 165, 255),
                        'label': bp.name.replace('_', ' ').title(),
                        'feedback': bp.feedback,
                        'confidence': bp.confidence
                    })
        
        return DetailedPoseAnalysis(
            pose_label=pose_label,
            overall_confidence=overall_confidence,
            body_parts=body_parts,
            key_issues=key_issues,
            strengths=strengths,
            recommendations=recommendations,
            visual_labels=visual_labels
        )
    
    def draw_analysis_labels(self, image, detailed_analysis: DetailedPoseAnalysis):
        """
        Draw visual labels on image showing areas needing improvement.
        
        Args:
            image: OpenCV image
            detailed_analysis: DetailedPoseAnalysis result
        
        Returns:
            Image with visual labels drawn
        """
        labeled_image = image.copy()
        
        # Draw pose landmarks
        image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)
        
        if results.pose_landmarks:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                labeled_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                )
            )
        
        # Draw visual labels for areas needing improvement
        for label in detailed_analysis.visual_labels:
            pos = label['position']
            color = label['color']
            label_text = label['label']
            
            # Draw circle/warning
            if label['type'] == 'circle':
                cv2.circle(labeled_image, pos, 15, color, 3)
                cv2.circle(labeled_image, pos, 5, color, -1)
            else:
                # Warning triangle
                pts = np.array([
                    [pos[0], pos[1] - 10],
                    [pos[0] - 8, pos[1] + 5],
                    [pos[0] + 8, pos[1] + 5]
                ], np.int32)
                cv2.fillPoly(labeled_image, [pts], color)
            
            # Draw text label
            cv2.putText(
                labeled_image,
                label_text,
                (pos[0] + 20, pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Draw confidence score
            conf_text = f"{label['confidence']:.0%}"
            cv2.putText(
                labeled_image,
                conf_text,
                (pos[0] + 20, pos[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        
        # Draw summary text
        summary_y = 30
        cv2.putText(
            labeled_image,
            f"Pose: {detailed_analysis.pose_label}",
            (10, summary_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            labeled_image,
            f"Confidence: {detailed_analysis.overall_confidence:.1%}",
            (10, summary_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Draw key issues
        if detailed_analysis.key_issues:
            issue_y = summary_y + 60
            cv2.putText(
                labeled_image,
                "Issues:",
                (10, issue_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
            for i, issue in enumerate(detailed_analysis.key_issues[:2]):
                cv2.putText(
                    labeled_image,
                    f"- {issue[:40]}",
                    (10, issue_y + 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1
                )
        
        return labeled_image
    
    def analyze_image_enhanced(self, image_path: str, save_labeled_image: bool = True):
        """
        Analyze image with enhanced detailed feedback and visual labels.
        
        Args:
            image_path: Path to image file
            save_labeled_image: Whether to save labeled image
        
        Returns:
            dict with detailed analysis and labeled image path
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not load image'}
        
        # Get base analysis
        base_analysis = self.analyze_image(image_path)
        if 'error' in base_analysis:
            return base_analysis
        
        pose_label = base_analysis.get('pose', 'unknown')
        confidence = base_analysis.get('confidence', 0)
        
        # Get landmarks
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)
        
        if not results.pose_landmarks:
            return {
                'error': 'No pose detected',
                'base_analysis': base_analysis
            }
        
        # Perform detailed analysis
        detailed_analysis = self.analyze_pose_detailed(
            image, pose_label, results.pose_landmarks
        )
        
        # Draw labels
        labeled_image = self.draw_analysis_labels(image, detailed_analysis)
        
        # Save labeled image if requested
        labeled_image_path = None
        if save_labeled_image:
            base_name = os.path.splitext(image_path)[0]
            labeled_image_path = f"{base_name}_labeled.jpg"
            cv2.imwrite(labeled_image_path, labeled_image)
        
        # Compile results
        result = {
            'pose_label': pose_label,
            'overall_confidence': confidence,
            'detailed_confidence': detailed_analysis.overall_confidence,
            'body_parts': [
                {
                    'name': bp.name,
                    'confidence': bp.confidence,
                    'needs_improvement': bp.needs_improvement,
                    'feedback': bp.feedback,
                    'current_angle': bp.current_angle,
                    'recommended_angle': bp.recommended_angle
                }
                for bp in detailed_analysis.body_parts
            ],
            'key_issues': detailed_analysis.key_issues,
            'strengths': detailed_analysis.strengths,
            'recommendations': detailed_analysis.recommendations,
            'labeled_image_path': labeled_image_path,
            'base_analysis': base_analysis
        }
        
        return result


def main():
    """Example usage."""
    analyzer = EnhancedSportsAnalyzer()
    
    # Analyze test image
    test_image = "test_images/test_0.jpg"
    if os.path.exists(test_image):
        result = analyzer.analyze_image_enhanced(test_image)
        
        print("📊 Enhanced Analysis Results:")
        print(f"Pose: {result['pose_label']}")
        print(f"Overall Confidence: {result['overall_confidence']:.1%}")
        print(f"\nBody Parts Analysis:")
        for bp in result['body_parts']:
            status = "⚠️" if bp['needs_improvement'] else "✅"
            print(f"{status} {bp['name']}: {bp['confidence']:.1%} - {bp['feedback']}")
        
        print(f"\nKey Issues:")
        for issue in result['key_issues']:
            print(f"  • {issue}")
        
        print(f"\nStrengths:")
        for strength in result['strengths']:
            print(f"  • {strength}")
        
        if result.get('labeled_image_path'):
            print(f"\n💾 Labeled image saved to: {result['labeled_image_path']}")


if __name__ == "__main__":
    import os
    main()

