#!/usr/bin/env python3
"""
VR Sports Analysis Project Setup
Equivalent to 'npm install' for Python projects
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f" {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print(" VR Sports Analysis Project Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(" Python 3.8+ is required")
        sys.exit(1)
    
    print(f" Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists(".venv"):
        print(" Creating virtual environment...")
        if not run_command("python3 -m venv .venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print(" Virtual environment already exists")
    
    # Activate virtual environment and install dependencies
    print(" Installing dependencies...")
    
    # Install from requirements.txt
    if os.path.exists("requirements.txt"):
        install_cmd = "source .venv/bin/activate && pip install -r requirements.txt"
        if not run_command(install_cmd, "Installing dependencies from requirements.txt"):
            print(" Some dependencies may have failed to install")
    else:
        print(" requirements.txt not found")
        sys.exit(1)
    
    # Verify installation
    print("🔄 Verifying installation...")
    test_cmd = "source .venv/bin/activate && python3 -c \"import torch, cv2, PIL; print('✅ Core dependencies verified')\""
    if run_command(test_cmd, "Verifying core dependencies"):
        print(" Setup completed successfully!")
        print("\n To activate the environment:")
        print("   source .venv/bin/activate")
        print("\n To run the analyzer:")
        print("   python3 analyze_video.py --demo")
    else:
        print(" Setup verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 