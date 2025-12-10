#!/usr/bin/env python3
"""Dev runner: simple loop using LocalCameraSimulator from services.camera.simulator

This is a lightweight runner for development that prints frame info
instead of attempting to open GUI windows (keeps it headless-friendly).
"""

import time
from services.camera.simulator import LocalCameraSimulator


def main():
    # default path is a placeholder; replace with a real file for local testing
    cam = LocalCameraSimulator(file_path="assets/sample_video.mp4")

    for i in range(20):
        frame = cam.get_frame()
        if frame is None:
            print("No frame returned from simulator")
            break
        print(f"Frame {i}: shape={frame.shape} dtype={frame.dtype}")
        time.sleep(0.05)


if __name__ == "__main__":
    main()
