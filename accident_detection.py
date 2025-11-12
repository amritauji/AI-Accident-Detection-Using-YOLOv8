#!/usr/bin/env python3
"""
Accident Detection using YOLOv8
Standalone script to run accident detection with pre-trained models
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import glob
from datetime import datetime

class AccidentDetector:
    def __init__(self, model_path="best.pt"):
        """
        Initialize the accident detector
        
        Args:
            model_path (str): Path to the YOLO model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from: {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                print(f"Model file not found: {self.model_path}")
                print("Falling back to yolov8s.pt")
                self.model = YOLO("yolov8s.pt")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def detect_accidents(self, source, conf=0.25, save_all=False):
        """
        Detect accidents in images or video - saves ONLY accident images by default
        
        Args:
            source (str): Path to image, video, or directory
            conf (float): Confidence threshold
            save_all (bool): Save all images (default: False, only saves accidents)
        """
        try:
            # Run prediction without saving everything
            results = self.model.predict(
                source=source,
                conf=conf,
                save=save_all  # Only save all if explicitly requested
            )
            
            # Always save accident images to separate folder
            accident_dir = "detected_accidents"
            os.makedirs(accident_dir, exist_ok=True)
            accident_count = 0
            
            for i, result in enumerate(results):
                if result.boxes is not None and len(result.boxes) > 0:
                    # Save annotated image with accidents
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{accident_dir}/accident_{timestamp}_{i}.jpg"
                    
                    # Save the annotated result
                    annotated_img = result.plot()
                    cv2.imwrite(filename, annotated_img)
                    accident_count += 1
                    print(f"🚨 Accident detected! Saved: {filename}")
            
            if accident_count > 0:
                print(f"Total accidents detected and saved: {accident_count}")
            else:
                print("No accidents detected.")
            
            return results
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def detect_from_webcam(self, conf=0.25, save_accidents=True):
        """
        Real-time accident detection from webcam
        
        Args:
            conf (float): Confidence threshold
            save_accidents (bool): Save frames when accidents are detected
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Create output directory for accident frames
        if save_accidents:
            accident_dir = "accident_frames"
            os.makedirs(accident_dir, exist_ok=True)
            print(f"Accident frames will be saved to: {accident_dir}")
        
        print("Starting webcam detection. Press 'q' to quit.")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = self.model(frame, conf=conf)
            
            # Check if accidents detected
            accident_detected = len(results[0].boxes) > 0 if results[0].boxes is not None else False
            
            if accident_detected and save_accidents:
                # Save original frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{accident_dir}/accident_{timestamp}_frame{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                
                # Save annotated frame
                annotated_filename = f"{accident_dir}/accident_{timestamp}_annotated_frame{frame_count}.jpg"
                annotated_frame = results[0].plot()
                cv2.imwrite(annotated_filename, annotated_frame)
                
                print(f"🚨 ACCIDENT DETECTED! Saved: {filename}")
            else:
                annotated_frame = results[0].plot()
            
            # Display the frame
            cv2.imshow('Accident Detection', annotated_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def validate_model(self, data_yaml="data.yaml"):
        """
        Validate the model on validation dataset
        
        Args:
            data_yaml (str): Path to data configuration file
        """
        if os.path.exists(data_yaml):
            print(f"Validating model with data config: {data_yaml}")
            results = self.model.val(data=data_yaml)
            return results
        else:
            print(f"Data config file not found: {data_yaml}")
            return None

def create_data_yaml():
    """Create a basic data.yaml file for local use"""
    data_yaml_content = """
# Accident Detection Dataset Configuration
train: data/train/images
val: data/valid/images
test: data/test/images

nc: 1
names: ['Accident']
"""
    
    with open("data.yaml", "w") as f:
        f.write(data_yaml_content)
    print("Created data.yaml file")

def main():
    parser = argparse.ArgumentParser(description="Accident Detection using YOLOv8")
    parser.add_argument("--model", default="best.pt", help="Path to model file")
    parser.add_argument("--source", help="Path to image/video/directory for detection")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time detection")
    parser.add_argument("--validate", action="store_true", help="Validate model on dataset")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--create-yaml", action="store_true", help="Create data.yaml file")
    parser.add_argument("--save-all", action="store_true", help="Save all processed images (default: only saves accidents)")
    
    args = parser.parse_args()
    
    # Create data.yaml if requested
    if args.create_yaml:
        create_data_yaml()
        return
    
    # Initialize detector
    detector = AccidentDetector(args.model)
    
    if args.webcam:
        # Real-time detection from webcam
        detector.detect_from_webcam(conf=args.conf, save_accidents=True)
    elif args.validate:
        # Validate model
        detector.validate_model()
    elif args.source:
        # Detect from source
        results = detector.detect_accidents(args.source, conf=args.conf, save_all=args.save_all)
        if results:
            print(f"Detection completed.")
    else:
        # Default: detect from test images if available
        test_images = glob.glob("data/test/images/*.jpg") + glob.glob("data/test/images/*.png")
        if test_images:
            print(f"Found {len(test_images)} test images")
            results = detector.detect_accidents("data/test/images", conf=args.conf, save_all=args.save_all)
            if results:
                print("Detection completed on test images")
        else:
            print("No source specified and no test images found.")
            print("Use --help for usage information")

if __name__ == "__main__":
    main()