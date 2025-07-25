# 🚀 VR Sports Analysis Project Setup Guide

This guide will help you set up the VR Sports Analysis project on your local machine.

## 📋 Prerequisites

- **Python 3.8 or newer** (recommended: Python 3.9+)
- **Git** (for cloning the repository)
- **At least 4GB RAM** (8GB+ recommended for video processing)
- **GPU with CUDA support** (optional, for faster ML training)

## 🛠 Installation Steps

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd VR_DEV
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Install Additional System Dependencies

#### On macOS:
```bash
# Install Homebrew if you haven't already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg for video processing
brew install ffmpeg

# Install OpenCV dependencies
brew install opencv
```

#### On Ubuntu/Debian:
```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install -y ffmpeg libsm6 libxext6 libxrender-dev libgomp1

# Install OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv
```

#### On Windows:
- Download and install [FFmpeg](https://ffmpeg.org/download.html)
- Add FFmpeg to your system PATH
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for some Python packages

### 5. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
# Create .env file
touch .env
```

Add your API keys to the `.env` file:
```env
# Google AI Studio API Key (for Gemini Sports Analyzer)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Other API keys you might need
# OPENAI_API_KEY=your_openai_key_here
```

### 6. Download Pre-trained Models (Optional)
If you have pre-trained models, place them in the project root:
- `image_model.pth` - For image classification
- `pose_model.pth` - For pose sequence classification

## 🧪 Testing the Installation

### 1. Test Basic Imports
```bash
python -c "
import torch
import cv2
import mediapipe
import numpy as np
import pandas as pd
print('✅ All core dependencies imported successfully!')
"
```

### 2. Test Video Processing
```bash
python video.py
```

### 3. Test Pose Extraction
```bash
python extract_poses.py
```

## 📁 Project Structure

After setup, your project should have this structure:
```
VR_DEV/
├── requirements.txt          # Python dependencies
├── SETUP.md                 # This setup guide
├── README.md                # Project documentation
├── .env                     # Environment variables (create this)
├── frame/                   # Input videos (.mp4 files)
├── frames/                  # Extracted video frames
├── pose_data/              # Extracted pose data
├── images/                 # Training images
├── scraped_videos/         # Downloaded videos
├── annotations/            # Manual annotations
├── VR_SPORT_SIM/          # Unity VR project
└── *.py                    # Python scripts
```

## 🎯 Quick Start

1. **Add videos to analyze**: Place `.mp4` files in the `frame/` directory
2. **Extract frames**: Run `python video.py`
3. **Extract poses**: Run `python extract_poses.py`
4. **Create labels**: Run `python labels_script.py`
5. **Train model**: Run `python train_model.py`
6. **Analyze videos**: Run `python analyze_video.py`

## 🔧 Troubleshooting

### Common Issues:

#### 1. MediaPipe Installation Issues
```bash
# Try upgrading pip first
pip install --upgrade pip

# Install MediaPipe with specific version
pip install mediapipe==0.10.0
```

#### 2. OpenCV Issues
```bash
# Uninstall and reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 3. PyTorch Installation Issues
```bash
# For CPU-only installation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA support (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Memory Issues
- Reduce batch size in training scripts
- Process videos in smaller chunks
- Use CPU instead of GPU for inference

#### 5. API Key Issues
- Ensure your `.env` file is in the project root
- Check that API keys are valid and have sufficient credits
- Verify internet connection for API calls

## 🚀 Next Steps

1. **Add your own videos** to the `frame/` directory
2. **Customize the analysis** by modifying the scripts
3. **Train on your own data** by updating the training scripts
4. **Integrate with Unity VR** by following the Unity project setup

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure you're using Python 3.8+
4. Check that your API keys are valid

## 🔄 Updates

To update dependencies:
```bash
pip install -r requirements.txt --upgrade
```

---

**Happy coding! 🎉** 