"""End-to-end smoke test: init every component, run one inference on a dummy frame."""
import numpy as np
import cv2

print("=== CameraLM smoke test ===\n")

print("[1/6] PersonDetector (YOLOv8n)...")
from cameralm.detector import PersonDetector
det = PersonDetector()
print("      OK\n")

print("[2/6] FaceEmbedder (InsightFace buffalo_l, ~300MB on first run)...")
from cameralm.face import FaceEmbedder
face = FaceEmbedder()
print("      OK\n")

print("[3/6] BodyEmbedder (boxmot OSNet, ~13MB on first run)...")
from cameralm.body import BodyEmbedder
body = BodyEmbedder()
print("      OK\n")

print("[4/7] PartialAppearanceEmbedder...")
from cameralm.partial import PartialAppearanceEmbedder
partial = PartialAppearanceEmbedder()
print("      OK\n")

print("[5/7] IdentityDB...")
from cameralm.identity_db import IdentityDB
db = IdentityDB()
print(f"      OK ({len(db.people)} known identities)\n")

print("[6/7] VLM warmup (Ollama / qwen3-vl:2b)...")
from cameralm.vlm import warmup as vlm_warmup
ok = vlm_warmup()
print("      OK" if ok else "      FAILED (Ollama unreachable or model missing)")
print()

print("[7/7] One-shot inference on dummy frame...")
frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
cv2.rectangle(frame, (400, 100), (700, 600), (100, 50, 200), -1)

tracks = det.track(frame)
print(f"      detector returned {len(tracks)} tracks (0 expected on blank frame)")

be = body.embed(frame, [400, 100, 700, 600])
print(f"      body embedding shape: {be.shape if be is not None else None}")

fe = face.embed(frame, [400, 100, 700, 600])
print(f"      face embedding: {fe.shape if fe is not None else 'None (no face in dummy frame)'}")

pa = partial.embed(frame, [400, 100, 700, 600])
print(f"      partial embedding shape: {pa.shape if pa is not None else None}")

print("\n=== All components loaded. Ready: python -m cameralm.main ===")
