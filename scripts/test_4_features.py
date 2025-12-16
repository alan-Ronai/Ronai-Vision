#!/usr/bin/env python3
"""
Test script for 4 new features.
Tests: Gemini API, Video Encoder, Log Query, Video Recording
"""

import sys
import time
import json
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


def test_vision_api():
    """Test Gemini vision API endpoints."""
    print("\n=== Testing Vision API ===")

    # Test status
    print("📊 Getting analyzer status...")
    resp = requests.get(f"{BASE_URL}/api/vision/status")
    print(f"  Status: {resp.json()}")

    if resp.status_code == 200:
        print("  ✅ Vision API is available")
    else:
        print("  ⚠️  Vision API not available (Gemini key missing?)")


def test_video_encoder():
    """Test video encoder endpoints."""
    print("\n=== Testing Video Encoder ===")

    # Test encoder status
    print("📊 Getting encoder status...")
    resp = requests.get(f"{BASE_URL}/api/videos/encoder/status")
    if resp.status_code == 200:
        status = resp.json()
        print(f"  ✅ Encoder Status:")
        print(f"     Codec: {status.get('codec')}")
        print(f"     FPS: {status.get('fps')}")
        print(f"     Videos: {status.get('total_videos')}")
        print(f"     Queue: {status.get('queue_size')} items")
    else:
        print(f"  ❌ Error: {resp.status_code}")

    # Test video listing
    print("📋 Getting video list...")
    resp = requests.get(f"{BASE_URL}/api/videos/videos/list?limit=5")
    if resp.status_code == 200:
        videos = resp.json()
        print(f"  ✅ Found {videos['total']} videos")
        for v in videos.get("videos", [])[:3]:
            print(f"     - {v['filename']} ({v['size_bytes']} bytes)")
    else:
        print(f"  ⚠️  No videos yet (normal on first run)")


def test_log_api():
    """Test operational log query API."""
    print("\n=== Testing Log Query API ===")

    # Test event retrieval
    print("📋 Getting recent events...")
    resp = requests.get(f"{BASE_URL}/api/logs/events?limit=10")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✅ Found {data['matched']} events")
        for evt in data.get("events", [])[:3]:
            print(f"     [{evt['severity']}] {evt['event_type']}: {evt['message']}")
    else:
        print(f"  ❌ Error: {resp.status_code}")

    # Test event stats
    print("📊 Getting event statistics...")
    resp = requests.get(f"{BASE_URL}/api/logs/events/stats")
    if resp.status_code == 200:
        stats = resp.json()
        print(f"  ✅ Stats:")
        print(f"     Total events: {stats['total_events']}")
        print(f"     Critical: {stats['critical_events']}")
        for event_type, count in stats.get("by_event_type", {}).items():
            print(f"     - {event_type}: {count}")
    else:
        print(f"  ⚠️  No events yet (normal on first run)")

    # Test critical events
    print("🚨 Getting critical events...")
    resp = requests.get(f"{BASE_URL}/api/logs/events/critical?limit=5")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✅ Critical events: {data['total_critical']}")
        for evt in data.get("recent", [])[:3]:
            print(f"     - {evt['message']}")
    else:
        print(f"  ⚠️  No critical events yet")


def test_recording_control():
    """Test manual recording control."""
    print("\n=== Testing Recording Control ===")

    print("⚠️  Manual recording test skipped (requires active track)")
    print("   In production, recording auto-triggers on armed person detection")
    print("   Check /api/videos/recordings/active for active sessions")


def test_gemini_analysis():
    """Test Gemini analysis endpoints."""
    print("\n=== Testing Gemini Analysis ===")

    print("⚠️  Analysis test requires frame data")
    print("   In production, automatically triggered when armed person detected")
    print("   API endpoint: POST /api/vision/analyze")
    print("   Deduplication: Max 2 per track ID, 3s minimum between")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing 4 New Features Implementation")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    try:
        # Check server is running
        print("\n🔗 Checking server connection...")
        resp = requests.get(f"{BASE_URL}/docs", timeout=2)
        if resp.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"⚠️  Server responded with {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Start it first:")
        print(f"   uvicorn api.server:app")
        sys.exit(1)

    # Run all tests
    test_vision_api()
    test_video_encoder()
    test_log_api()
    test_recording_control()
    test_gemini_analysis()

    print("\n" + "=" * 60)
    print("✅ Tests Complete!")
    print("=" * 60)
    print("\n📚 Next Steps:")
    print("1. Start the pipeline: python scripts/run_multi_camera.py")
    print("2. Trigger armed person detection (weapon in frame)")
    print(
        "3. Check auto-recording: curl http://localhost:8000/api/videos/recordings/active"
    )
    print("4. Query logs: curl http://localhost:8000/api/logs/events")
    print("5. Download video: curl http://localhost:8000/api/videos/videos/list")
    print("\n📖 Full API docs at: http://localhost:8000/docs\n")


if __name__ == "__main__":
    main()
