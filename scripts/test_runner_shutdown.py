"""Integration test: start run_loop in a thread with a stop event and ensure it stops cleanly."""

import threading
import time

from scripts.run_multi_camera import run_loop


def test_runner_start_stop():
    stop_evt = threading.Event()

    def _target():
        try:
            run_loop(stop_event=stop_evt, max_frames=None)
        except Exception as e:
            print("runner raised:", e)

    t = threading.Thread(target=_target, daemon=True)
    t.start()

    # let it run briefly
    time.sleep(2.0)

    # request shutdown
    stop_evt.set()

    t.join(timeout=5.0)

    if t.is_alive():
        raise RuntimeError("Runner did not exit within timeout")


if __name__ == "__main__":
    test_runner_start_stop()
    print("runner start/stop test passed")
