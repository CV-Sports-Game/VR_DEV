# Sports Annotation Instructions

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
