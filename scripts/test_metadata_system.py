#!/usr/bin/env python3
"""
Test script for track metadata system.

Demonstrates:
- Creating tracks with metadata
- Adding notes, tags, alerts
- Querying metadata through API
- Persistence and cleanup
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tracker.bot_sort import BoTSortTracker
from services.tracker.metadata_manager import get_metadata_manager


def test_track_metadata():
    """Test track metadata functionality."""

    print("=" * 70)
    print("TRACK METADATA SYSTEM TEST")
    print("=" * 70)

    # Initialize tracker
    print("\n1. Initializing BoT-SORT tracker...")

    # COCO class names (subset for testing)
    class_names = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    tracker = BoTSortTracker(
        max_lost=30,
        min_hits=3,
        iou_threshold=0.3,
        lambda_app=0.5,
        max_cost=0.8,
        min_confidence_history=0.3,
        class_names=class_names,  # Pass class names for auto-tagging
    )
    print("✓ Tracker initialized with auto-tagging enabled")

    # Create some detections
    print("\n2. Creating test detections...")
    boxes = np.array(
        [
            [100, 100, 200, 300],  # Person 1
            [300, 150, 400, 350],  # Person 2
            [500, 200, 700, 400],  # Car
        ]
    )
    class_ids = np.array([0, 0, 2])  # 0=person, 2=car
    confidences = np.array([0.9, 0.85, 0.95])

    # Update tracker (need multiple frames to confirm tracks)
    tracks = []
    for frame in range(5):
        tracks = tracker.update(boxes, class_ids, confidences)
        print(f"  Frame {frame}: {len(tracks)} confirmed tracks")
        time.sleep(0.1)

    print(f"✓ Created {len(tracks)} confirmed tracks")

    # Add metadata to tracks
    print("\n3. Adding metadata to tracks...")

    manager = get_metadata_manager()

    for i, track in enumerate(tracks):
        if track.class_id == 0:  # Person
            # Add note
            track.add_note(
                f"Person {i + 1} detected entering scene", author="test_operator"
            )

            # Add tags (class name is auto-tagged, add additional tags here)
            if i == 0:
                track.add_tag("suspicious")
                track.add_tag("high_priority")

            # Set attributes
            track.set_attribute("color", "blue_shirt" if i == 0 else "red_jacket")
            track.set_attribute("position", f"zone_{chr(65 + i)}")

            # Add alert for first person
            if i == 0:
                track.add_alert(
                    "suspicious_behavior",
                    "Person loitering near entrance",
                    severity="warning",
                )

            # Set behavior
            track.set_behavior("walking", confidence=0.85)

            # Add zones
            track.add_zone("entrance")

            # Sync metadata to manager
            manager.update_track_metadata(
                track.track_id, track.class_id, track.get_metadata_summary()
            )

            print(f"  Track {track.track_id}: Added notes, tags, attributes, alerts")

        elif track.class_id == 2:  # Car
            track.add_note("Vehicle entering parking lot", author="test_operator")
            # 'car' tag is auto-added, add additional descriptive tags
            track.add_tag("vehicle")  # Additional category tag
            track.set_attribute("color", "silver")
            track.set_attribute("type", "sedan")
            track.add_zone("parking_lot")

            # Sync metadata to manager
            manager.update_track_metadata(
                track.track_id, track.class_id, track.get_metadata_summary()
            )

            print(f"  Track {track.track_id}: Added vehicle metadata")

    print("✓ Metadata added to all tracks")

    # Query metadata through manager
    print("\n4. Querying metadata through MetadataManager...")

    # Get stats
    stats = manager.get_stats()
    print("\nStatistics:")
    print(f"  Total tracks: {stats['total_tracks']}")
    print(f"  Total notes: {stats['total_notes']}")
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  Tracks by class: {stats['tracks_by_class']}")
    print(f"  Recent tracks: {stats['recent_tracks']}")

    # Get metadata for each track
    print("\n5. Detailed metadata for each track:")
    for track in tracks:
        metadata = manager.get_track_metadata(track.track_id)
        if metadata:
            print(f"\n--- Track {track.track_id} ---")
            print(
                f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['created_at']))}"
            )
            print(f"  Tags: {metadata.get('tags', [])}")
            print(f"  Attributes: {metadata.get('attributes', {})}")
            print(f"  Notes: {len(metadata.get('notes', []))}")
            print(f"  Alerts: {len(metadata.get('alerts', []))}")
            print(f"  Zones: {list(metadata.get('zones_visited', []))}")

            # Show notes
            for note in metadata.get("notes", []):
                note_time = time.strftime("%H:%M:%S", time.localtime(note["timestamp"]))
                print(f"    [{note_time}] {note['author']}: {note['text']}")

            # Show alerts
            for alert in metadata.get("alerts", []):
                alert_time = time.strftime(
                    "%H:%M:%S", time.localtime(alert["timestamp"])
                )
                print(
                    f"    [{alert_time}] ALERT {alert['severity'].upper()}: {alert['message']}"
                )

    # Search by tag
    print("\n6. Searching metadata...")

    suspicious_tracks = manager.search_metadata(tag="suspicious")
    print(f"\nTracks tagged 'suspicious': {len(suspicious_tracks)}")
    for track_data in suspicious_tracks:
        tid = track_data["track_id"]
        meta = track_data["metadata"]
        print(f"  Track {tid}: {meta.get('tags', [])}")

    # Search by zone
    entrance_tracks = manager.search_metadata(zone="entrance")
    print(f"\nTracks in 'entrance' zone: {len(entrance_tracks)}")
    for track_data in entrance_tracks:
        tid = track_data["track_id"]
        print(f"  Track {tid}")

    # Get tracks by class
    person_tracks = manager.get_tracks_by_class(0)
    print(f"\nPerson tracks (class 0): {len(person_tracks)}")

    vehicle_tracks = manager.get_tracks_by_class(2)
    print(f"Vehicle tracks (class 2): {len(vehicle_tracks)}")

    # Save metadata
    print("\n7. Saving metadata to disk...")
    save_path = "output/test_metadata.json"
    manager.save_metadata(save_path)
    print(f"✓ Metadata saved to {save_path}")

    # Test cleanup
    print("\n8. Testing cleanup (no tracks should be expired yet)...")
    cleaned = manager.cleanup_expired()
    print(f"✓ Cleaned up {cleaned} expired tracks")

    print("\n" + "=" * 70)
    print("METADATA SYSTEM TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start the server: python api/server.py")
    print("2. Query metadata via API:")
    print("   - GET http://localhost:8000/api/metadata/stats")
    print("   - GET http://localhost:8000/api/metadata/all")
    print("   - GET http://localhost:8000/api/metadata/search?tag=suspicious")
    print("   - POST http://localhost:8000/api/metadata/track/{id}/note")


def test_formatted_output():
    """Test formatted metadata output."""

    print("\n" + "=" * 70)
    print("FORMATTED OUTPUT TEST")
    print("=" * 70)

    manager = get_metadata_manager()
    all_metadata = manager.get_all_metadata()

    if not all_metadata:
        print("No metadata found. Run test_track_metadata() first.")
        return

    print("\nFormatted metadata for each track:")

    from api.routes.metadata import _format_metadata

    for track_id, metadata in all_metadata.items():
        formatted = _format_metadata(track_id, metadata)
        print(f"\n{formatted}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test track metadata system")
    parser.add_argument(
        "--format", action="store_true", help="Test formatted output only"
    )

    args = parser.parse_args()

    try:
        if args.format:
            test_formatted_output()
        else:
            test_track_metadata()
            print("\nRunning formatted output test...")
            test_formatted_output()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
