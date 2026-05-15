"""Capture + inference pipeline threads.

The hot loop used to run capture, detection, identity resolution, and rendering
serially on one thread, so the frame rate was 1/sum(stages). The pipeline now
spans three threads:

  * capture   - owns the camera, loops cap.read(), publishes the latest raw frame
  * inference - pulls the latest raw frame, runs detection + identity resolution,
                publishes the latest (frame_id, frame, frame_info)
  * main      - (in main.py) renders + handles input

The three stages overlap, so the frame rate approaches 1/max(stage) instead of
1/sum(stage). Every hand-off is latest-wins: a slow consumer drops stale frames
rather than queuing them - a live view must not lag behind reality.
"""

import logging
import threading
import time
from dataclasses import replace

import cv2
import numpy as np

from .config import (
    CAMERA_INDEX,
    CAMERA_MAX_READ_FAILURES,
    CAMERA_RECONNECT_ATTEMPTS,
    DETECT_EVERY_N,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_RESOLVES_PER_FRAME,
    STATS_LOG_INTERVAL_SECONDS,
    TRACK_REUSE_MAX_GAP_SECONDS,
    TRACK_REUSE_MIN_IOU,
    VLM_INTERVAL_SECONDS,
)
from .tracking import TrackState, bbox_iou, cache_ttl_for, resolve_track_identity

log = logging.getLogger(__name__)


def _configure_camera(cap, want_w: int, want_h: int) -> None:
    """Force MJPG and the target resolution.

    Most USB webcams default to YUY2 (uncompressed), which is USB-bandwidth
    limited - measured ~16 fps at 640x480 and ~1 fps at 1280x720 on this rig.
    MJPG is compressed on-camera, so it delivers the sensor's real frame rate.
    FOURCC must be set *before* the resolution for the DirectShow backend.
    A 1-frame buffer keeps cap.read() returning the freshest frame, not a stale
    queued one - important for a live, interactive view.
    """
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _frames_arrive(cap) -> bool:
    for _ in range(5):
        ok, frame = cap.read()
        if ok and frame is not None and frame.any():
            return True
    return False


def _open_camera(index: int, want_w: int, want_h: int):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        _configure_camera(cap, want_w, want_h)
        if _frames_arrive(cap):
            return cap
        cap.release()
        # Retry: reopen and force MJPG at the camera's own default resolution
        # before giving up on this backend entirely.
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if _frames_arrive(cap):
                return cap
        cap.release()
    return None


def crop_bbox(frame, bbox):
    """Return a copied sub-image for `bbox` (xyxy), or None if the box is empty."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


class PipelineWorker:
    """Owns the camera; runs capture (one thread) and detection + identity
    resolution (a second thread); publishes the latest (frame_id, frame,
    frame_info) for the main thread to render.

    Thread-safety contract:
      * Two latest-wins slots, each under its own lock: `_raw_*` (capture ->
        inference) and `_latest` (inference -> main). Separate locks so the
        capture thread writing a raw frame never blocks the main thread reading
        the published result.
      * The published `frame` is treated as immutable by all sides. The capture
        thread hands back a fresh array each cap.read(), the inference thread
        only reads it, and `ui.Renderer.finish()` returns a new array rather
        than compositing in place.
      * `frame_info` values are independent `dataclasses.replace()` snapshots, so
        the main thread never reads a `TrackState` the inference thread mutates
        on a later frame (the non-stale fast path mutates bbox/last_seen).
      * `db` / `descriptions` are internally locked; concurrent access from the
        inference thread, the main thread, and the admin server is serialized.
    """

    def __init__(self, detector, face, body, partial, db, descriptions, vlm_worker):
        self._detector = detector
        self._face = face
        self._body = body
        self._partial = partial
        self._db = db
        self._descriptions = descriptions
        self._vlm_worker = vlm_worker

        # Inference-thread-only state - never touched by another thread.
        self._track_cache: dict[int, TrackState] = {}
        self._last_vlm_submit: dict[int, float] = {}
        self._last_thumbnail_attempt: dict[int, float] = {}

        # capture -> inference hand-off (latest raw frame).
        self._raw_lock = threading.Lock()
        self._raw_frame: np.ndarray | None = None
        self._raw_seq = 0

        # inference -> main hand-off (latest rendered-ready result).
        self._lock = threading.Lock()
        self._latest: tuple[int, np.ndarray, dict[int, TrackState]] | None = None
        self._frame_id = 0

        self._stop = threading.Event()
        self._failed = threading.Event()
        self._capture_thread = threading.Thread(
            target=self._capture_run, daemon=True, name="PipelineCapture"
        )
        self._inference_thread = threading.Thread(
            target=self._inference_run, daemon=True, name="PipelineInference"
        )

    def start(self) -> None:
        self._capture_thread.start()
        self._inference_thread.start()

    @property
    def failed(self) -> bool:
        """True if the pipeline exited because the camera could not be opened/recovered."""
        return self._failed.is_set()

    def get_latest(self):
        """Most recently published (frame_id, frame, frame_info), or None pre-first-frame."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        self._inference_thread.join(timeout=3.0)
        self._capture_thread.join(timeout=3.0)
        for t in (self._inference_thread, self._capture_thread):
            if t.is_alive():
                log.warning("Pipeline thread %s did not stop within timeout - abandoning it.", t.name)

    def _get_raw(self):
        with self._raw_lock:
            return self._raw_seq, self._raw_frame

    def _publish(self, frame, frame_info) -> None:
        # Snapshot each TrackState: the non-stale fast path in _process_tracks
        # mutates the cached object's bbox/last_seen, so the main thread must get
        # its own copies or it could read a value mid-mutation.
        snapshot = {tid: replace(ts) for tid, ts in frame_info.items()}
        with self._lock:
            self._frame_id += 1
            self._latest = (self._frame_id, frame, snapshot)

    def _process_tracks(self, frame, tracks, now) -> dict[int, TrackState]:
        """Resolve identity for the tracks in this frame; returns {tid: TrackState}.

        Runs entirely on the inference thread, in three passes:
          1. classify  - apply the correctness guards, decide fresh vs. stale
          2. resolve    - re-embed at most MAX_RESOLVES_PER_FRAME stale tracks,
                          longest-waiting first, so the per-frame cost stays
                          bounded no matter how many people are in view
          3. upkeep     - thumbnail capture + VLM submit-gate for every track

        A stale track past the per-frame cap keeps its previous verdict (or, if
        brand-new, a pending "Unknown" placeholder) and is resolved on a coming
        frame - its old last_check keeps it near the front of the queue.
        """
        alive_ids: set[int] = set()
        frame_info: dict[int, TrackState] = {}
        pending: list[tuple[int, tuple, TrackState | None]] = []

        # --- Pass 1: classify ---
        for tid, bbox, _conf in tracks:
            alive_ids.add(tid)
            bbox_t = tuple(int(v) for v in bbox)
            cached = self._track_cache.get(tid)

            # Correctness guard: drop a stale lock pointing at a person the admin
            # UI deleted while this track was alive.
            if (
                cached is not None
                and cached.locked_pid is not None
                and not self._db.has_person(cached.locked_pid)
            ):
                cached.locked_pid = None
                cached.locked_at = 0.0
                cached.locked_source = None

            # Correctness guard: ByteTrack can reuse a track id for a different
            # person. Only flag it when the box JUMPS between truly consecutive
            # frames - a big move after a tracking gap is just the person walking.
            if (
                cached is not None
                and now - cached.last_seen < TRACK_REUSE_MAX_GAP_SECONDS
                and bbox_iou(cached.bbox, bbox_t) < TRACK_REUSE_MIN_IOU
            ):
                cached = None

            cached_pid_deleted = (
                cached is not None
                and cached.pid is not None
                and not self._db.has_person(cached.pid)
            )
            ttl = cache_ttl_for(cached.source if cached else None)
            stale = cached is None or cached_pid_deleted or (now - cached.last_check > ttl)

            if stale:
                pending.append((tid, bbox_t, cached))
            else:
                cached.bbox = bbox_t
                cached.last_seen = now
                frame_info[tid] = cached

        # --- Pass 2: resolve, longest-waiting first, capped per frame ---
        # A brand-new track (cached is None -> last_check 0.0) sorts to the front,
        # so new people are identified fast; stale-but-known tracks wait their turn.
        pending.sort(key=lambda item: item[2].last_check if item[2] is not None else 0.0)
        for i, (tid, bbox_t, cached) in enumerate(pending):
            if i < MAX_RESOLVES_PER_FRAME:
                cached = resolve_track_identity(
                    bbox_t, frame, cached, now, self._face, self._body, self._partial, self._db
                )
                self._track_cache[tid] = cached
            elif cached is not None:
                # Keep the previous verdict; it stays near the front of the queue
                # next frame because its last_check did not advance.
                cached.bbox = bbox_t
                cached.last_seen = now
            else:
                # Brand-new track that did not get a resolve slot this frame -
                # show a pending "Unknown" until its (imminent) turn.
                cached = TrackState(bbox=bbox_t, last_seen=now)
                self._track_cache[tid] = cached
            frame_info[tid] = cached

        # --- Pass 3: per-track upkeep for every visible track ---
        for tid, cached in frame_info.items():
            pid = cached.pid
            if pid is not None and self._db.has_person(pid) and not self._db.has_thumbnail(pid):
                if now - self._last_thumbnail_attempt.get(pid, 0.0) > 1.0:
                    self._last_thumbnail_attempt[pid] = now
                    crop = crop_bbox(frame, cached.bbox)
                    if crop is not None and self._db.save_thumbnail(pid, crop):
                        log.info("Saved thumbnail for person %d.", pid)

            # Submit-gate: only ask the VLM about a track with no current
            # description. Inert while USE_VLM is False (vlm_worker is None).
            if self._vlm_worker is not None and self._descriptions.get(tid) is None:
                if now - self._last_vlm_submit.get(tid, 0.0) > VLM_INTERVAL_SECONDS:
                    crop = crop_bbox(frame, cached.bbox)
                    if crop is not None and self._vlm_worker.submit(tid, crop):
                        self._last_vlm_submit[tid] = now

        # Prune state for tracks gone for > 5s.
        for tid in list(self._track_cache.keys()):
            if now - self._track_cache[tid].last_seen > 5.0:
                pruned = self._track_cache.pop(tid)
                self._last_vlm_submit.pop(tid, None)
                if pruned.pid is not None:
                    self._last_thumbnail_attempt.pop(pruned.pid, None)
        self._descriptions.prune(alive_ids)

        return frame_info

    def _capture_run(self) -> None:
        """Camera thread: open the camera, loop cap.read(), publish the raw frame."""
        cap = _open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
        if cap is None:
            log.error("Could not open webcam at index %d.", CAMERA_INDEX)
            self._failed.set()
            return
        log.info(
            "Camera opened at %dx%d.",
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        camera_fail_count = 0
        frames = 0
        last_stats = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()

                if not ok or frame is None:
                    camera_fail_count += 1
                    if camera_fail_count >= CAMERA_MAX_READ_FAILURES:
                        cap.release()
                        cap = None
                        for attempt in range(1, CAMERA_RECONNECT_ATTEMPTS + 1):
                            if self._stop.is_set():
                                return
                            log.warning(
                                "Camera lost - reconnect attempt %d/%d...",
                                attempt, CAMERA_RECONNECT_ATTEMPTS,
                            )
                            cap = _open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
                            if cap is not None:
                                log.info("Camera reconnected.")
                                break
                            time.sleep(min(2.0 * attempt, 10.0))  # bounded backoff
                        if cap is None:
                            log.error(
                                "Camera could not be reconnected after %d attempts - "
                                "stopping pipeline.", CAMERA_RECONNECT_ATTEMPTS,
                            )
                            self._failed.set()
                            return
                        camera_fail_count = 0
                    else:
                        time.sleep(0.05)
                    continue

                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    # A reconnected/odd camera can deliver grayscale or non-uint8
                    # frames, which would crash the overlay compositor downstream.
                    log.warning(
                        "Camera delivered an unexpected frame format (%s, %s) - skipping.",
                        frame.shape, frame.dtype,
                    )
                    time.sleep(0.02)
                    continue
                camera_fail_count = 0

                with self._raw_lock:
                    self._raw_frame = frame
                    self._raw_seq += 1

                frames += 1
                elapsed = time.monotonic() - last_stats
                if elapsed > STATS_LOG_INTERVAL_SECONDS:
                    log.info("capture: %.1f fps (camera read rate)", frames / elapsed)
                    frames = 0
                    last_stats = time.monotonic()
        finally:
            if cap is not None:
                cap.release()

    def _inference_run(self) -> None:
        """Inference thread: pull the latest raw frame, run detection (every Nth
        frame) + identity resolution, publish the result for the main thread."""
        consecutive_errors = 0
        frames = 0
        detect_frames = 0
        t_acc = {"detect": 0.0, "resolve": 0.0}
        last_stats = time.monotonic()
        last_seq = -1
        last_tracks: list = []
        since_detect = 0

        while not self._stop.is_set() and not self._failed.is_set():
            seq, frame = self._get_raw()
            if frame is None or seq == last_seq:
                time.sleep(0.001)  # no new raw frame yet - light poll
                continue
            last_seq = seq

            try:
                now = time.monotonic()

                # Detection cadence: run YOLO every Nth frame, reuse boxes between.
                if since_detect % DETECT_EVERY_N == 0:
                    t1 = time.perf_counter()
                    last_tracks = self._detector.track(frame)
                    t_acc["detect"] += time.perf_counter() - t1
                    detect_frames += 1
                since_detect += 1

                t2 = time.perf_counter()
                frame_info = self._process_tracks(frame, last_tracks, now)
                t_acc["resolve"] += time.perf_counter() - t2

                self._publish(frame, frame_info)

                frames += 1
                elapsed = now - last_stats
                if elapsed > STATS_LOG_INTERVAL_SECONDS:
                    log.info(
                        "inference: %.1f fps | detect=%.1fms (1/%d frames) resolve=%.1fms",
                        frames / elapsed,
                        1000.0 * t_acc["detect"] / max(detect_frames, 1),
                        DETECT_EVERY_N,
                        1000.0 * t_acc["resolve"] / max(frames, 1),
                    )
                    for k in t_acc:
                        t_acc[k] = 0.0
                    frames = 0
                    detect_frames = 0
                    last_stats = now

                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                # Log the first few then every 100th - a persistent per-frame
                # bug stays clearly visible without burying the log.
                if consecutive_errors <= 3 or consecutive_errors % 100 == 0:
                    log.exception(
                        "Pipeline inference error (#%d) - skipping frame.", consecutive_errors
                    )
                # Degrade to slow-but-alive instead of pegging a core.
                time.sleep(0.1)
