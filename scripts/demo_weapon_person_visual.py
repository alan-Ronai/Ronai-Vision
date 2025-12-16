#!/usr/bin/env python3
"""
Visual demo of weapon-person association with red box rendering.

Creates a synthetic frame showing:
- Person without weapon (green box)
- Person with weapon (red box)
"""

import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tracker.base_tracker import Track
from services.output.renderer import FrameRenderer
from services.tracker.metadata_manager import get_metadata_manager


def create_demo_frame():
    """Create a demo frame with armed and unarmed persons."""

    # Create blank frame
    frame = np.ones((600, 1000, 3), dtype=np.uint8) * 40  # Dark gray

    # Add title
    cv2.putText(
        frame,
        "Weapon-Person Association Demo",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    # Draw person silhouettes (simplified)
    # Person 1 (unarmed) - left side
    cv2.rectangle(frame, (100, 150), (250, 500), (100, 100, 100), -1)  # Body
    cv2.circle(frame, (175, 120), 30, (100, 100, 100), -1)  # Head

    # Person 2 (armed) - right side
    cv2.rectangle(frame, (500, 150), (650, 500), (100, 100, 100), -1)  # Body
    cv2.circle(frame, (575, 120), 30, (100, 100, 100), -1)  # Head

    # Draw weapon on person 2
    cv2.rectangle(frame, (580, 250), (630, 280), (200, 200, 200), -1)  # Gun

    # Add labels
    cv2.putText(
        frame,
        "Normal Person",
        (100, 530),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "Armed Person",
        (500, 530),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return frame


def main():
    print("=" * 70)
    print("WEAPON-PERSON ASSOCIATION VISUAL DEMO")
    print("=" * 70)

    # Create demo frame
    frame = create_demo_frame()

    # Create tracks
    print("\n1. Creating tracks...")

    # Track 1: Normal person (no weapon)
    track1 = Track(
        track_id=1, box=np.array([80, 90, 270, 520]), class_id=0, confidence=0.92
    )
    track1.metadata["tags"] = ["person"]
    track1.metadata["created_at"] = 1234567890.0
    track1.metadata["updated_at"] = 1234567890.0

    # Track 2: Armed person
    track2 = Track(
        track_id=2, box=np.array([480, 90, 670, 520]), class_id=0, confidence=0.89
    )
    track2.metadata["tags"] = ["person", "armed"]
    track2.metadata["alerts"] = [
        {
            "type": "armed_person",
            "message": "Person armed with: pistol",
            "severity": "critical",
            "timestamp": 1234567890.5,
        }
    ]
    track2.metadata["attributes"] = {
        "weapons_detected": ["pistol"],
        "weapon_detection_count": 1,
    }
    track2.metadata["created_at"] = 1234567890.0
    track2.metadata["updated_at"] = 1234567891.0

    tracks = [track1, track2]
    class_names = ["person"]

    print(f"  Created {len(tracks)} tracks")
    print(f"  - Track 1: Normal person (green box expected)")
    print(f"  - Track 2: Armed person (red box expected)")

    # Sync to metadata manager
    print("\n2. Syncing metadata...")
    manager = get_metadata_manager()
    for track in tracks:
        manager.update_track_metadata(
            track.track_id, track.class_id, track.get_metadata_summary()
        )
    print("  ✓ Metadata synced")

    # Render tracks
    print("\n3. Rendering tracks...")
    renderer = FrameRenderer()
    output = renderer.render_tracks(frame, tracks, class_names)

    # Add legend
    legend_y = 70
    cv2.rectangle(output, (800, legend_y), (840, legend_y + 30), (0, 255, 0), 2)
    cv2.putText(
        output,
        "= Normal",
        (850, legend_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    legend_y += 40
    cv2.rectangle(output, (800, legend_y), (840, legend_y + 30), (0, 0, 255), 2)
    cv2.putText(
        output,
        "= Armed",
        (850, legend_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Save output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "weapon_person_demo.png"
    cv2.imwrite(str(output_path), output)

    print(f"  ✓ Rendered frame saved to {output_path}")

    # Print metadata
    print("\n4. Track metadata:")
    for track in tracks:
        metadata = manager.get_track_metadata(track.track_id)
        print(f"\n  Track {track.track_id}:")
        print(f"    Tags: {metadata.get('tags', [])}")

        alerts = metadata.get("alerts", [])
        if alerts:
            print(f"    Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"      - [{alert['severity'].upper()}] {alert['message']}")

        weapons = metadata.get("attributes", {}).get("weapons_detected")
        if weapons:
            print(f"    Weapons: {weapons}")

    # Query armed persons
    print("\n5. Querying armed persons via API...")
    armed_tracks = manager.search_metadata(tag="armed")
    print(f"  Found {len(armed_tracks)} armed person(s)")
    for track_data in armed_tracks:
        tid = track_data["track_id"]
        print(f"    - Track {tid}")

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETED")
    print("=" * 70)
    print(f"\nVisual output: {output_path}")
    print("\nWhat you should see:")
    print("  - Left person: GREEN box (normal)")
    print("  - Right person: RED box with [ARMED] label")
    print("\nAPI Query:")
    print("  curl http://localhost:8000/api/metadata/search?tag=armed")
    print("\nSystem is ready for production use! 🎯")


if __name__ == "__main__":
    main()
