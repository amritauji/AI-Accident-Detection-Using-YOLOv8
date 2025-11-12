#!/usr/bin/env python3
"""
Quick video inference script
"""

import os
from ultralytics import YOLO
import cv2
from datetime import datetime

def run_video_inference():
    # Load model
    if os.path.exists("best.pt"):
        model = YOLO("best.pt")
        print("✅ Using best.pt model")
    else:
        model = YOLO("yolov8s.pt")
        print("✅ Using yolov8s.pt model")
    
    # Video file
    # video_path = "Accident detection demo (1).mp4"
    video_path = "final.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"🎥 Processing video: {video_path}")
    
    # Create output directory
    os.makedirs("detected_accidents", exist_ok=True)
    
    # Run inference
    results = model.predict(
        source=video_path,
        conf=0.3,
        save=False  # We'll save manually
    )
    
    accident_count = 0
    
    # Process results and save only accident frames
    for i, result in enumerate(results):
        if result.boxes is not None and len(result.boxes) > 0:
            # Save accident frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detected_accidents/video_accident_{timestamp}_frame{i}.jpg"
            
            # Save annotated frame
            annotated_img = result.plot()
            cv2.imwrite(filename, annotated_img)
            accident_count += 1
            print(f"🚨 Accident detected in frame {i}! Saved: {filename}")
    
    print(f"\n📊 RESULTS:")
    print(f"   • Total frames processed: {len(results)}")
    print(f"   • Accidents detected: {accident_count}")
    print(f"   • Accident frames saved to: detected_accidents/")
    
    if accident_count > 0:
        print(f"✅ Video processing completed! {accident_count} accident frames saved.")
    else:
        print("ℹ️  No accidents detected in the video.")

if __name__ == "__main__":
    run_video_inference()