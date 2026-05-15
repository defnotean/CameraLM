"""Isolate camera throughput from pipeline cost.

Measures raw cap.read() FPS with NO inference/rendering, at the resolution
CameraLM actually uses, and reports the properties that commonly cap a webcam
at ~15 fps (FOURCC/pixel format, driver FPS, exposure, buffer size).
"""
import time

import cv2

from cameralm.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


def _fourcc_str(v: float) -> str:
    n = int(v)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
print(f"opened={cap.isOpened()}")
print(f"native: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"set {FRAME_WIDTH}x{FRAME_HEIGHT} -> {w:.0f}x{h:.0f}")
print(f"FOURCC={_fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))}  "
      f"driver_FPS={cap.get(cv2.CAP_PROP_FPS):.1f}  "
      f"buffersize={cap.get(cv2.CAP_PROP_BUFFERSIZE):.0f}")
print(f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE)}  "
      f"auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")

# Warm up, then time 120 raw reads - no processing at all.
for _ in range(10):
    cap.read()
n = 120
t0 = time.perf_counter()
ok_count = 0
for _ in range(n):
    ok, frame = cap.read()
    if ok:
        ok_count += 1
dt = time.perf_counter() - t0
print(f"\nRAW cap.read() x{n}: {dt:.2f}s  ->  {n / dt:.1f} fps  (ok={ok_count}/{n})")

# Re-test forcing MJPG - the classic fix for USB-bandwidth-limited webcams.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
print(f"after force MJPG: FOURCC={_fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))}")
for _ in range(10):
    cap.read()
t0 = time.perf_counter()
for _ in range(n):
    cap.read()
dt = time.perf_counter() - t0
print(f"RAW cap.read() x{n} (MJPG): {dt:.2f}s  ->  {n / dt:.1f} fps")

cap.release()
