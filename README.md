#  VR Sports Analysis - AI Coach
Project by: Vrishan, Adhya, Aditya

An AI-powered sports analysis system that provides real-time feedback on boxing and fencing techniques. Built for VR integration and automated coaching(VR aspect work-in-progress).

##  What This Project Does

### **Core Features:**
- **AI Pose Recognition**: Automatically detects boxing and fencing poses from images/videos
- **Real-Time Analysis**: Provides instant feedback on form and technique
- **Coaching Feedback**: Gives specific tips like a real coach would
- **VR Ready**: Designed to integrate with Unity for VR sports training

### **Supported Sports:**
- **Boxing**: punch, uppercut, straight_punch, fast_punch, hook, block, guard, footwork
- **Fencing**: lunge, slide, parry, block, guard, en_garde, riposte, footwork

### **Use Cases:**
- Replace human coaches with AI analysis
- Real-time feedback during training sessions
- Foundation for VR games with AI opponents
- Automated sports technique assessment

##  Installation Guide

### **Quick Install (Recommended)**
```bash
# Clone the repository
git clone https://github.com/your-username/VR_DEV.git
cd VR_DEV

# Run the installer (like npm install)
./install.sh
```

### **Manual Installation**
```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### **Requirements:**
- Python 3.8+
- 4GB+ RAM (for model training)
- Webcam (for real-time analysis)

##  How to Use

### **1. Train the Model**
```bash
# Activate environment
source .venv/bin/activate

# Train the model (choose mode when prompted)
python3 train_model.py
```

### **2. Analyze Images**
```bash
# Analyze a single image
python3 analyze_video.py --image path/to/your/image.jpg

# Run demo on test images
python3 analyze_video.py --demo
```

### **3. Analyze Videos**
```bash
# Analyze a video file
python3 analyze_video.py --video path/to/your/video.mp4

# Custom sampling rate (every 15 frames)
python3 analyze_video.py --video video.mp4 --sample-rate 15
```

##  How the Model Works

### **Architecture:**
- **CNN Model**: Simple convolutional neural network for image classification
- **Input**: 128x128 RGB images
- **Output**: 16 pose classes with confidence scores
- **Training**: Uses labeled dataset of boxing/fencing images

### **Analysis Pipeline:**
1. **Image Preprocessing**: Resize and normalize input images
2. **Feature Extraction**: CNN extracts visual features
3. **Classification**: Predicts pose class and confidence
4. **Feedback Generation**: Provides coaching tips based on pose and confidence

### **Model Performance:**
- **Accuracy**: ~85% on test dataset
- **Speed**: Real-time capable (30+ FPS)
- **Classes**: 16 different boxing/fencing poses

##  Project Structure

```
VR_DEV/
├── images/                 # Training dataset (480 images)
│   ├── boxing_punch/      # Boxing pose images
│   ├── fencing_lunge/     # Fencing pose images
│   └── ...
├── pose_data/             # Pose sequence data
├── test_images/           # Test images for demo
├── train_model.py         # Unified training script
├── analyze_video.py       # Analysis and feedback tool
├── labels_script.py       # Label generation
├── requirements.txt       # Dependencies
├── setup.py              # Installation script
└── install.sh            # Quick installer
```

##  MVP Features

### **Current Capabilities:**
-  Image and video analysis
-  Real-time pose detection
-  Coaching feedback generation
-  Command-line interface
- Model training pipeline

### **Coming Soon:**
-  Webcam real-time analysis
-  Unity VR integration
-  AI opponent system
-  Progress tracking

##  Future Vision

### **VR Integration:**
- Real-time feedback in VR headsets
- AI coaches (like Mike Tyson) in VR games
- Interactive training scenarios
- Performance tracking and improvement

### **Advanced Features:**
- Motion sequence analysis
- Biomechanical assessment
- Personalized training plans
- Multi-player VR competitions

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Troubleshooting

### **Common Issues:**

**"Module not found" errors:**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

**Model training fails:**
```bash
# Check if you have enough RAM
# Try reducing batch size in train_model.py
```

**Video analysis slow:**
```bash
# Increase sample rate for faster processing
python3 analyze_video.py --video video.mp4 --sample-rate 60
```


