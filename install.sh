#!/bin/bash

echo "🎯 VR Sports Analysis Project - Quick Install"
echo "=============================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Run the setup script
python3 setup.py

echo ""
echo "🎉 Installation complete!"
echo ""
echo "💡 Next steps:"
echo "   1. Activate the environment: source .venv/bin/activate"
echo "   2. Test the analyzer: python3 analyze_video.py --demo"
echo "   3. Train the model: python3 train_model.py"
echo "" 