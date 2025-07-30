# 🚀 Quick Setup Guide

## ⚡ Get Started in 5 Minutes

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/VR_DEV.git
cd VR_DEV
```

### **Step 2: Run the Installer**
```bash
./install.sh
```
*This will install all dependencies and set up the environment*

### **Step 3: Test the System**
```bash
# Activate the environment
source .venv/bin/activate

# Run the demo
python3 analyze_video.py --demo
```

### **Step 4: Train Your Model**
```bash
# Train the model (choose option 2 for image classification)
python3 train_model.py
```

### **Step 5: Analyze Your Own Content**
```bash
# Analyze an image
python3 analyze_video.py --image path/to/your/image.jpg

# Analyze a video
python3 analyze_video.py --video path/to/your/video.mp4
```

## 🎯 What You'll See

### **Demo Output:**
```
🎯 VR Sports Analyzer - AI Coach
==========================================
📸 Analyzing: test_images/test_0.jpg
🎯 Detected pose: boxing_punch (Confidence: 85.2%)

🎯 HIGH CONFIDENCE - Great form!

💡 Great punch form! Keep your guard up.

🥊 Boxing Tip: Remember to keep your guard up and move your feet!
```

### **Video Analysis Output:**
```
📹 Video Analysis: boxing_session.mp4
   Duration: 45.2s, FPS: 30.0, Total frames: 1356
   Sampling every 30 frames...
   Frame 30/1356 (2.2%) - boxing_punch (85.2%)
   Frame 60/1356 (4.4%) - boxing_guard (92.1%)

📊 ANALYSIS SUMMARY
==================
Total frames analyzed: 45
Average confidence: 78.5%
Most common pose: boxing_punch (15 times)

🎯 COACHING RECOMMENDATIONS:
  • Good variety of techniques
  • Focus on maintaining guard position
```

## 🆘 Common Issues

### **"Permission denied" on install.sh:**
```bash
chmod +x install.sh
./install.sh
```

### **"Module not found" errors:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### **Model training fails:**
- Make sure you have at least 4GB RAM
- Try running on a smaller dataset first

## 🎮 Next Steps

1. **Test with your own images/videos**
2. **Train the model on your data**
3. **Integrate with Unity for VR**
4. **Add webcam real-time analysis**

## 📞 Need Help?

- Check the main README.md for detailed documentation
- Create an issue on GitHub
- Review the troubleshooting section

---

**You're all set! 🎉** 