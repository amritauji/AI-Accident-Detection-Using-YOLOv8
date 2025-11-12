#!/usr/bin/env python3
"""
Test script to verify the accident detection setup
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if required files exist"""
    required_files = [
        "accident_detection.py",
        "requirements.txt", 
        "data.yaml",
        "README.md"
    ]
    
    model_files = ["best.pt", "yolov8s.pt"]
    
    print("Checking required files...")
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} - Found")
        else:
            print(f"✗ {file} - Missing")
    
    print("\nChecking model files...")
    model_found = False
    for model in model_files:
        if os.path.exists(model):
            print(f"✓ {model} - Found")
            model_found = True
        else:
            print(f"✗ {model} - Not found")
    
    if not model_found:
        print("\n⚠️  Warning: No model files found!")
        print("Please ensure either 'best.pt' or 'yolov8s.pt' is in the directory")
    
    return model_found

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        "ultralytics",
        "cv2", 
        "numpy",
        "torch"
    ]
    
    print("\nChecking Python dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
                print(f"✓ opencv-python - Version {cv2.__version__}")
            elif package == "ultralytics":
                import ultralytics
                print(f"✓ ultralytics - Found")
            elif package == "numpy":
                import numpy as np
                print(f"✓ numpy - Version {np.__version__}")
            elif package == "torch":
                import torch
                print(f"✓ torch - Version {torch.__version__}")
        except ImportError:
            print(f"✗ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_model_loading():
    """Test if models can be loaded"""
    print("\nTesting model loading...")
    
    try:
        from ultralytics import YOLO
        
        # Try loading best.pt first
        if os.path.exists("best.pt"):
            try:
                model = YOLO("best.pt")
                print("✓ best.pt - Loaded successfully")
                return True
            except Exception as e:
                print(f"✗ best.pt - Error loading: {e}")
        
        # Try loading yolov8s.pt
        if os.path.exists("yolov8s.pt"):
            try:
                model = YOLO("yolov8s.pt")
                print("✓ yolov8s.pt - Loaded successfully")
                return True
            except Exception as e:
                print(f"✗ yolov8s.pt - Error loading: {e}")
        
        # Try downloading yolov8s.pt
        try:
            print("Attempting to download yolov8s.pt...")
            model = YOLO("yolov8s.pt")  # This will download if not present
            print("✓ yolov8s.pt - Downloaded and loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to download yolov8s.pt: {e}")
            return False
            
    except ImportError:
        print("✗ Cannot import ultralytics - install requirements first")
        return False

def main():
    """Main test function"""
    print("Accident Detection Setup Test")
    print("=" * 40)
    
    # Check files
    files_ok = check_files()
    
    # Check dependencies  
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Setup incomplete - install dependencies first")
        return False
    
    # Test model loading
    model_ok = test_model_loading()
    
    print("\n" + "=" * 40)
    if files_ok and deps_ok and model_ok:
        print("✅ Setup complete! You can now run the accident detection system.")
        print("\nQuick start:")
        print("  python accident_detection.py --webcam")
        print("  python accident_detection.py --source path/to/image.jpg")
        print("\nOr use the batch file:")
        print("  run_detection.bat")
    else:
        print("❌ Setup incomplete - please fix the issues above")
    
    return files_ok and deps_ok and model_ok

if __name__ == "__main__":
    main()