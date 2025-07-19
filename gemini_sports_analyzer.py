import os
import json
import base64
import requests
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass
from pathlib import Path
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SportsAnalysis:
    """Data class for storing sports analysis results"""
    movement_type: str
    confidence_score: float
    form_quality: str  # "excellent", "good", "fair", "poor"
    posture_analysis: Dict[str, Any]
    technique_feedback: List[str]
    improvement_suggestions: List[str]
    safety_concerns: List[str]
    biomechanical_analysis: Dict[str, Any]
    raw_response: str

class GeminiSportsAnalyzer:
    """
    A comprehensive sports analysis system using Google's Gemini Pro API
    to analyze sports images and provide detailed feedback on form, movement, and posture.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """
        Initialize the Gemini Sports Analyzer
        
        Args:
            api_key: Your Google AI Studio API key
            model: Gemini model to use for analysis
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Sports-specific analysis prompts
        self.analysis_prompts = {
            "general": self._get_general_analysis_prompt(),
            "fencing": self._get_fencing_analysis_prompt(),
            "martial_arts": self._get_martial_arts_analysis_prompt(),
            "combat_sports": self._get_combat_sports_analysis_prompt(),
            "detailed": self._get_detailed_analysis_prompt()
        }
    
    def _get_general_analysis_prompt(self) -> str:
        """Get the general sports analysis prompt"""
        return """
        You are an expert sports biomechanics analyst and coach. Analyze the provided sports image and provide a comprehensive assessment including:

        1. **Movement Identification**: Identify the specific movement or technique being performed
        2. **Form Quality Assessment**: Rate the overall form quality (excellent/good/fair/poor)
        3. **Posture Analysis**: Analyze body alignment, balance, and positioning
        4. **Technique Feedback**: Provide specific feedback on execution
        5. **Improvement Suggestions**: Offer actionable coaching advice
        6. **Safety Assessment**: Identify any potential safety concerns
        7. **Biomechanical Analysis**: Analyze joint angles, muscle engagement, and movement efficiency

        Provide your analysis in the following JSON format:
        {
            "movement_type": "string",
            "confidence_score": 0.0-1.0,
            "form_quality": "excellent|good|fair|poor",
            "posture_analysis": {
                "body_alignment": "string",
                "balance": "string", 
                "head_position": "string",
                "spine_alignment": "string"
            },
            "technique_feedback": ["string"],
            "improvement_suggestions": ["string"],
            "safety_concerns": ["string"],
            "biomechanical_analysis": {
                "joint_angles": "string",
                "muscle_engagement": "string",
                "movement_efficiency": "string"
            }
        }
        """
    
    def _get_fencing_analysis_prompt(self) -> str:
        """Get fencing-specific analysis prompt"""
        return """
        You are an expert fencing coach and biomechanics specialist. Analyze this fencing image and provide detailed feedback on:

        1. **Technique Identification**: Identify the specific fencing technique (lunge, parry, riposte, etc.)
        2. **Form Assessment**: Evaluate proper fencing stance, guard position, and execution
        3. **Distance Management**: Analyze proper distance and timing
        4. **Weapon Control**: Assess blade position and control
        5. **Footwork**: Evaluate foot positioning and movement
        6. **Tactical Analysis**: Assess strategic positioning and readiness
        7. **Safety**: Check for proper protective gear and safe execution

        Provide analysis in the same JSON format as the general prompt, but with fencing-specific details.
        """
    
    def _get_martial_arts_analysis_prompt(self) -> str:
        """Get martial arts-specific analysis prompt"""
        return """
        You are an expert martial arts instructor and biomechanics analyst. Analyze this martial arts image and provide detailed feedback on:

        1. **Technique Identification**: Identify the specific martial arts technique
        2. **Stance Analysis**: Evaluate proper stance, balance, and grounding
        3. **Power Generation**: Assess hip rotation, weight transfer, and power delivery
        4. **Defensive Position**: Analyze guard position and readiness
        5. **Breathing**: Assess breathing pattern and relaxation
        6. **Flow and Timing**: Evaluate movement flow and timing
        7. **Traditional Form**: Assess adherence to traditional form principles

        Provide analysis in the same JSON format as the general prompt, but with martial arts-specific details.
        """
    
    def _get_combat_sports_analysis_prompt(self) -> str:
        """Get combat sports-specific analysis prompt"""
        return """
        You are an expert combat sports coach and biomechanics specialist. Analyze this combat sports image and provide detailed feedback on:

        1. **Technique Identification**: Identify the specific combat technique
        2. **Stance and Balance**: Evaluate fighting stance and balance
        3. **Power and Speed**: Assess power generation and speed of execution
        4. **Defensive Awareness**: Analyze defensive positioning and awareness
        5. **Range Management**: Evaluate proper distance and range control
        6. **Tactical Positioning**: Assess strategic positioning and angles
        7. **Safety and Control**: Check for safe execution and control

        Provide analysis in the same JSON format as the general prompt, but with combat sports-specific details.
        """
    
    def _get_detailed_analysis_prompt(self) -> str:
        """Get detailed analysis prompt for comprehensive feedback"""
        return """
        You are a world-class sports biomechanics expert with decades of coaching experience. Analyze this sports image with extreme attention to detail and provide:

        1. **Movement Identification**: Precisely identify the movement/technique with confidence level
        2. **Form Quality Assessment**: Rate form quality (excellent/good/fair/poor) with detailed reasoning
        3. **Posture Analysis**: Comprehensive analysis of body alignment, balance, and positioning
        4. **Technique Feedback**: Specific, actionable feedback on execution quality
        5. **Improvement Suggestions**: Detailed, step-by-step coaching advice
        6. **Safety Assessment**: Thorough safety analysis and risk identification
        7. **Biomechanical Analysis**: Detailed analysis of joint angles, muscle engagement, and movement efficiency
        8. **Performance Optimization**: Suggestions for maximizing performance and efficiency

        Provide your analysis in the following JSON format:
        {
            "movement_type": "string",
            "confidence_score": 0.0-1.0,
            "form_quality": "excellent|good|fair|poor",
            "posture_analysis": {
                "body_alignment": "string",
                "balance": "string", 
                "head_position": "string",
                "spine_alignment": "string",
                "limb_positions": "string"
            },
            "technique_feedback": ["string"],
            "improvement_suggestions": ["string"],
            "safety_concerns": ["string"],
            "biomechanical_analysis": {
                "joint_angles": "string",
                "muscle_engagement": "string",
                "movement_efficiency": "string",
                "power_generation": "string"
            },
            "performance_notes": "string"
        }
        """
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API transmission"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_single_image(self, image_path: str, sport_type: str = "detailed") -> SportsAnalysis:
        """
        Analyze a single sports image
        
        Args:
            image_path: Path to the image file
            sport_type: Type of sport for specialized analysis
            
        Returns:
            SportsAnalysis object with detailed results
        """
        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            
            # Get appropriate prompt
            prompt = self.analysis_prompts.get(sport_type, self.analysis_prompts["detailed"])
            
            # Prepare API request
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topK": 32,
                    "topP": 1,
                    "maxOutputTokens": 2048,
                }
            }
            
            # Make API call
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            analysis_text = result['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON from response
            analysis_data = self._parse_text_response(analysis_text)
            
            # Create SportsAnalysis object
            return SportsAnalysis(
                movement_type=analysis_data.get('movement_type', 'unknown'),
                confidence_score=analysis_data.get('confidence_score', 0.0),
                form_quality=analysis_data.get('form_quality', 'fair'),
                posture_analysis=analysis_data.get('posture_analysis', {}),
                technique_feedback=analysis_data.get('technique_feedback', []),
                improvement_suggestions=analysis_data.get('improvement_suggestions', []),
                safety_concerns=analysis_data.get('safety_concerns', []),
                biomechanical_analysis=analysis_data.get('biomechanical_analysis', {}),
                raw_response=analysis_text
            )
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            # Return default analysis on error
            return SportsAnalysis(
                movement_type="error",
                confidence_score=0.0,
                form_quality="unknown",
                posture_analysis={},
                technique_feedback=["Analysis failed"],
                improvement_suggestions=[],
                safety_concerns=[],
                biomechanical_analysis={},
                raw_response=str(e)
            )
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response from Gemini API"""
        try:
            # Try to extract JSON from the response
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {}

def main():
    """Test the Gemini Sports Analyzer"""
    # Get API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("Example: export GEMINI_API_KEY='your_api_key_here'")
        return
    
    # Initialize analyzer
    analyzer = GeminiSportsAnalyzer(api_key)
    
    # Test with a sample image if available
    test_image = "test_images/test_0.jpg"
    
    if os.path.exists(test_image):
        print(f"🧪 Testing Gemini Sports Analyzer with {test_image}")
        analysis = analyzer.analyze_single_image(test_image, "detailed")
        
        print("\n📊 Analysis Results:")
        print(f"Movement Type: {analysis.movement_type}")
        print(f"Confidence: {analysis.confidence_score}")
        print(f"Form Quality: {analysis.form_quality}")
        print(f"Technique Feedback: {analysis.technique_feedback}")
        print(f"Improvement Suggestions: {analysis.improvement_suggestions}")
    else:
        print("❌ No test image found. Please run the scraper first or provide a test image.")

if __name__ == "__main__":
    main() 