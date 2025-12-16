#!/usr/bin/env python3
"""Test weapon detection on an image or video."""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.detector import MultiDetector


def test_weapon_detection(image_path: str):
    """Test weapon detection on an image."""
    print("Loading multi-detector...")
    
    # Configure detectors
    detector_config = {
        "coco": {
            "model": "yolo12n.pt",
            "confidence": 0.25,
            "classes": ["person"],  # Only detect people from COCO
            "enabled": True
        },
        "weapons": {
            "model": "firearm-yolov8n.pt",
            "confidence": 0.5,
            "classes": None,  # All weapon classes
            "enabled": True
        }
    }
    
    detector = MultiDetector(detector_config, device="cpu")
    
    # Load image
    print(f"Loading image: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Run detection
    print("Running detection...")
    results = detector.predict(frame, confidence=0.25)
    
    # Draw results
    print(f"\nDetected {len(results)} objects:")
    print("-" * 60)
    
    for i in range(len(results)):
        x1, y1, x2, y2 = results.boxes[i]
        score = results.scores[i]
        class_id = results.class_ids[i]
        class_name = results.class_names[class_id]
        
        print(f"  {i+1}. {class_name}: {score:.2f} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
        
        # Draw box
        color = (0, 0, 255) if "weapon" in class_name.lower() or "gun" in class_name.lower() or "pistol" in class_name.lower() or "rifle" in class_name.lower() or "knife" in class_name.lower() else (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{class_name} {score:.2f}", (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save result
    output_path = "output/weapon_detection_result.jpg"
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"\n✅ Result saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test weapon detection")
    parser.add_argument("image", help="Path to image file")
    
    args = parser.parse_args()
    
    test_weapon_detection(args.image)
