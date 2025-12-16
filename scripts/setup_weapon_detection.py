#!/usr/bin/env python3
"""Setup weapon detection for Ronai-Vision pipeline."""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.detector.weapon_models import download_weapon_model, list_weapon_models


def main():
    """Download weapon detection model and setup configuration."""
    print("=" * 70)
    print("RONAI-VISION WEAPON DETECTION SETUP")
    print("=" * 70)
    print()

    # Show available models
    print("Available weapon detection models:")
    print()
    list_weapon_models()
    print()

    # Download firearm detection model
    print("\nDownloading firearm detection model...")
    print("-" * 70)

    try:
        model_path = download_weapon_model("firearm-yolov8n", models_dir="models")
        print(f"\n✅ Weapon model ready: {model_path}")
    except Exception as e:
        print(f"\n❌ Failed to download: {e}")
        print("\nPlease install huggingface_hub:")
        print("  pip install huggingface_hub")
        return 1

    # Create example config
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    config_example = """
# Add this to your pipeline configuration:

detector_config = {
    "coco": {
        "model": "yolo12n.pt",
        "confidence": 0.25,
        "classes": ["person", "car", "truck", "motorcycle", "bicycle"],
        "enabled": True,
        "priority": 1
    },
    "weapons": {
        "model": "firearm-yolov8n.pt",
        "confidence": 0.5,
        "classes": None,  # Detect all weapon classes
        "enabled": True,
        "priority": 2  # Higher priority for weapons
    }
}

# Usage:
from services.detector import MultiDetector

detector = MultiDetector(detector_config, device="cpu")
results = detector.predict(frame, confidence=0.25)

# Results will contain both COCO objects and weapons
# Weapons have higher priority for tracking/alerts
"""

    print(config_example)

    # Save example script
    example_path = "scripts/test_weapon_detection.py"
    print(f"Creating example script: {example_path}")

    example_code = '''#!/usr/bin/env python3
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
    print(f"\\nDetected {len(results)} objects:")
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
    print(f"\\n✅ Result saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test weapon detection")
    parser.add_argument("image", help="Path to image file")
    
    args = parser.parse_args()
    
    test_weapon_detection(args.image)
'''

    os.makedirs("scripts", exist_ok=True)
    with open(example_path, "w") as f:
        f.write(example_code)

    os.chmod(example_path, 0o755)

    print(f"✅ Example script created: {example_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print("\n✅ Weapon detection model downloaded")
    print("✅ MultiDetector system ready")
    print("✅ Example script created")

    print("\n📋 Next steps:")
    print("  1. Test weapon detection:")
    print(f"     python {example_path} path/to/image.jpg")
    print("\n  2. Integrate into pipeline:")
    print("     See config example above")
    print("\n  3. Weapons are detected WITHOUT ReID tracking")
    print("     (as requested - no person IDs assigned to weapons)")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
