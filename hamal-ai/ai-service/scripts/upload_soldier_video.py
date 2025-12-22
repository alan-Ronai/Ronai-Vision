#!/usr/bin/env python3
"""
Upload Soldier Video Script

This script helps set up a soldier video file as a camera source for the HAMAL-AI system.
The video will loop infinitely and be processed as if it were a live camera feed.

Usage:
    python scripts/upload_soldier_video.py <video_path> [--camera-id <id>] [--camera-name <name>]

Example:
    python scripts/upload_soldier_video.py /path/to/soldier_demo.mp4 --camera-id cam-soldier --camera-name "Demo Soldier"
"""

import os
import sys
import argparse
import requests
import json
from pathlib import Path

# Backend URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")
AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://localhost:8000")


def validate_video_file(video_path: str) -> bool:
    """Check if the video file exists and is a valid video format."""
    path = Path(video_path)

    if not path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return False

    valid_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v'}
    if path.suffix.lower() not in valid_extensions:
        print(f"⚠️  Warning: Unusual video extension: {path.suffix}")
        print(f"   Supported formats: {', '.join(valid_extensions)}")

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"📹 Video file: {path.name} ({size_mb:.1f} MB)")

    return True


def create_camera(video_path: str, camera_id: str, camera_name: str, backend_url: str) -> bool:
    """Create a camera entry pointing to the video file."""

    # Convert to absolute path
    abs_path = str(Path(video_path).absolute())

    # Camera payload - use file:// URL for local files
    camera_data = {
        "cameraId": camera_id,
        "name": camera_name,
        "location": "Demo Video",
        "type": "file",  # Indicate this is a file source
        "rtspUrl": abs_path,  # The reader handles file:// automatically
        "status": "online",
        "aiEnabled": True,
        "order": 99,  # Put at end of list
    }

    try:
        # Create camera in backend
        print(f"\n📡 Creating camera in backend...")
        response = requests.post(
            f"{backend_url}/api/cameras",
            json=camera_data,
            timeout=10
        )

        if response.status_code == 201:
            print(f"✅ Camera created: {camera_id}")
            camera = response.json()
            print(f"   ID: {camera.get('_id', camera_id)}")
        elif response.status_code == 400:
            # Camera might already exist - try to update
            print(f"⚠️  Camera {camera_id} may already exist, attempting update...")
            response = requests.put(
                f"{backend_url}/api/cameras/{camera_id}",
                json=camera_data,
                timeout=10
            )
            if response.ok:
                print(f"✅ Camera updated: {camera_id}")
            else:
                print(f"❌ Failed to update camera: {response.text}")
                return False
        else:
            print(f"❌ Failed to create camera: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend at {backend_url}")
        print("   Make sure the backend is running: npm run dev")
        return False
    except Exception as e:
        print(f"❌ Error creating camera: {e}")
        return False

    return True


def start_detection(camera_id: str, video_path: str, ai_service_url: str) -> bool:
    """Start AI detection for the camera."""
    abs_path = str(Path(video_path).absolute())

    try:
        print(f"\n🤖 Starting AI detection...")
        response = requests.post(
            f"{ai_service_url}/detection/start/{camera_id}",
            params={"rtsp_url": abs_path},
            timeout=30
        )

        if response.ok:
            result = response.json()
            print(f"✅ Detection started for {camera_id}")
            print(f"   Status: {result.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Failed to start detection: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to AI service at {ai_service_url}")
        print("   Make sure the AI service is running: python main.py")
        return False
    except Exception as e:
        print(f"❌ Error starting detection: {e}")
        return False


def enable_demo_mode(ai_service_url: str) -> bool:
    """Enable demo mode (slower FPS for longer video duration)."""
    try:
        print(f"\n🐌 Enabling demo mode (slower processing)...")
        response = requests.post(
            f"{ai_service_url}/detection/config/demo-mode",
            params={"enabled": True},
            timeout=10
        )

        if response.ok:
            result = response.json()
            print(f"✅ Demo mode enabled")
            print(f"   Detection FPS: {result.get('detection_fps')}")
            print(f"   Stream FPS: {result.get('stream_fps')}")
            return True
        else:
            print(f"⚠️  Failed to enable demo mode: {response.text}")
            return False

    except Exception as e:
        print(f"⚠️  Could not enable demo mode: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload a soldier video file as a camera source for HAMAL-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/upload_soldier_video.py demo_videos/soldier.mp4
  python scripts/upload_soldier_video.py /absolute/path/video.mp4 --camera-id cam-demo
  python scripts/upload_soldier_video.py video.mp4 --camera-name "Armed Soldier Demo" --demo-mode
        """
    )

    parser.add_argument(
        "video_path",
        help="Path to the soldier video file (.mp4, .avi, etc.)"
    )

    parser.add_argument(
        "--camera-id", "-i",
        default="cam-soldier",
        help="Camera ID to use (default: cam-soldier)"
    )

    parser.add_argument(
        "--camera-name", "-n",
        default="חייל מזוין - הדגמה",
        help="Camera name to display (default: 'חייל מזוין - הדגמה')"
    )

    parser.add_argument(
        "--demo-mode", "-d",
        action="store_true",
        help="Enable demo mode (slower FPS for longer video)"
    )

    parser.add_argument(
        "--backend-url",
        default=BACKEND_URL,
        help=f"Backend URL (default: {BACKEND_URL})"
    )

    parser.add_argument(
        "--ai-url",
        default=AI_SERVICE_URL,
        help=f"AI Service URL (default: {AI_SERVICE_URL})"
    )

    args = parser.parse_args()

    # Use URLs from args (they already default to the module-level constants)
    backend_url = args.backend_url
    ai_service_url = args.ai_url

    print("=" * 60)
    print("🎬 HAMAL-AI Soldier Video Upload Script")
    print("=" * 60)

    # Step 1: Validate video file
    if not validate_video_file(args.video_path):
        sys.exit(1)

    # Step 2: Create camera
    if not create_camera(args.video_path, args.camera_id, args.camera_name, backend_url):
        sys.exit(1)

    # Step 3: Start detection
    if not start_detection(args.camera_id, args.video_path, ai_service_url):
        print("\n⚠️  Camera created but detection not started.")
        print("   You may need to start detection manually from the UI.")

    # Step 4: Enable demo mode if requested
    if args.demo_mode:
        enable_demo_mode(ai_service_url)

    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print(f"\nCamera ID: {args.camera_id}")
    print(f"Camera Name: {args.camera_name}")
    print(f"Video: {args.video_path}")
    print("\n📺 Open the frontend to see the video stream with AI detection:")
    print(f"   http://localhost:3001")
    print("\n💡 Tips:")
    print("   - The video will loop automatically")
    print("   - Use demo mode for slower processing (--demo-mode)")
    print("   - Check API docs at: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
