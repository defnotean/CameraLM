import logging

import torch
from ultralytics import YOLO

from .config import (
    DETECT_CONF,
    POSE_DETECT_CONF,
    POSE_DETECT_EVERY,
    POSE_TRACK_ID_OFFSET,
    TRACKER_CONFIG,
    USE_POSE_PARTIAL_DETECT,
    YOLO_IMGSZ,
    YOLO_MODEL,
    YOLO_POSE_MODEL,
)

log = logging.getLogger(__name__)

# FP16 inference roughly halves YOLO latency + VRAM on the GTX 1650.
_USE_HALF = torch.cuda.is_available()
# Pin YOLO to the GPU explicitly - "make sure it's all on the GPU" should not
# depend on ultralytics' auto-pick default. 0 = first CUDA device.
_DEVICE = 0 if torch.cuda.is_available() else "cpu"


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class PersonDetector:
    """YOLOv8 person detection with ByteTrack IDs plus optional pose fallback."""

    PERSON_CLASS = 0  # COCO class index

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.pose_model = YOLO(YOLO_POSE_MODEL) if USE_POSE_PARTIAL_DETECT else None
        self._frame_count = 0
        if _DEVICE == "cpu":
            log.warning("YOLO detector on CPU - torch.cuda.is_available() is False.")
        else:
            log.info(
                "YOLO detector on GPU (cuda:%s, FP16=%s)%s.",
                _DEVICE, _USE_HALF,
                "" if self.pose_model is None else " + pose fallback",
            )

    def track(self, frame):
        """Return a list of (track_id, bbox_xyxy_ndarray, conf)."""
        self._frame_count += 1
        out = self._track_people(frame)
        # The pose pass is a second full YOLO inference - only run it every Nth
        # frame so it doesn't halve the framerate when a person is in view.
        if self.pose_model is not None and self._frame_count % POSE_DETECT_EVERY == 0:
            out.extend(self._track_pose_fallback(frame, out))
        return out

    def _track_people(self, frame):
        results = self.model.track(
            frame,
            conf=DETECT_CONF,
            classes=[self.PERSON_CLASS],
            persist=True,
            tracker=TRACKER_CONFIG,
            half=_USE_HALF,
            device=_DEVICE,
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )
        return self._results_to_tracks(results, tid_offset=0)

    def _track_pose_fallback(self, frame, existing):
        results = self.pose_model.track(
            frame,
            conf=POSE_DETECT_CONF,
            persist=True,
            tracker=TRACKER_CONFIG,
            half=_USE_HALF,
            device=_DEVICE,
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )
        candidates = self._results_to_tracks(results, tid_offset=POSE_TRACK_ID_OFFSET)
        accepted = []
        existing_boxes = [box for _, box, _ in existing]
        for tid, box, conf in candidates:
            if any(_iou(box, existing_box) > 0.45 for existing_box in existing_boxes):
                continue
            accepted.append((tid, box, conf))
            existing_boxes.append(box)
        return accepted

    def _results_to_tracks(self, results, tid_offset: int):
        out = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return out
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().tolist()
        for tid, box, conf in zip(ids, xyxy, confs):
            out.append((int(tid) + tid_offset, box, float(conf)))
        return out
