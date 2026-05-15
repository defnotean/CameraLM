"""Local VLM via Ollama. Decoupled from the hot loop by a background worker."""

import base64
import logging
import queue
import threading
import time
from typing import Callable

import cv2
import numpy as np
import requests

from .config import (
    OLLAMA_HOST,
    VLM_IMAGE_MAX_PX,
    VLM_KEEP_ALIVE,
    VLM_MAX_DESCRIPTION_CHARS,
    VLM_MODEL,
    VLM_NUM_CTX,
    VLM_NUM_PREDICT,
    VLM_NUM_THREADS,
    VLM_PROMPT,
    VLM_TIMEOUT_SECONDS,
    VLM_USE_GPU,
    VLM_WARMUP_TIMEOUT_SECONDS,
)

log = logging.getLogger(__name__)

# One pooled HTTP session reused for every call - avoids a TCP handshake per request.
_session = requests.Session()


def _resize_for_vlm(image_bgr: np.ndarray, max_px: int = VLM_IMAGE_MAX_PX) -> np.ndarray:
    """Downscale so the longer side is <= max_px. Cuts vision-token count and CPU
    inference time; a 10-word physical description needs no more detail than this."""
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_px:
        return image_bgr
    scale = max_px / longest
    return cv2.resize(
        image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )


def _encode_jpeg(image_bgr: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", _resize_for_vlm(image_bgr), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def call_vlm(
    image_bgr: np.ndarray,
    prompt: str = VLM_PROMPT,
    timeout: float = VLM_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> str:
    """Synchronous call to Ollama's /api/generate with one image. Returns the (capped) description.

    `session` lets each caller pass its own connection pool - `requests.Session`
    is not thread-safe, so the warmup call and the worker thread must not share one.
    """
    options = {
        "temperature": 0.2,
        "num_predict": VLM_NUM_PREDICT,
        "num_ctx": VLM_NUM_CTX,
    }
    # num_gpu=0 forces Ollama to run the model fully on CPU. On a 4 GB card the
    # VLM otherwise resident in VRAM starves the YOLO/InsightFace/OSNet models
    # and craters the camera framerate - see VLM_USE_GPU in config.py.
    if not VLM_USE_GPU:
        options["num_gpu"] = 0
    # On CPU the VLM competes with the camera loop for cores; cap its threads so
    # the live pipeline keeps a guaranteed CPU floor - see VLM_NUM_THREADS.
    if VLM_NUM_THREADS:
        options["num_thread"] = VLM_NUM_THREADS
    payload = {
        "model": VLM_MODEL,
        "prompt": prompt,
        "images": [_encode_jpeg(image_bgr)],
        "stream": False,
        "keep_alive": VLM_KEEP_ALIVE,
        "options": options,
    }
    # (connect, read) timeout: a fast connect failure means Ollama is down;
    # the longer read budget covers a cold model load.
    http = session if session is not None else _session
    r = http.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=(5.0, timeout))
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Ollama error: {data.get('error')!r}")
    text = str(data.get("response", "")).strip()
    return text[:VLM_MAX_DESCRIPTION_CHARS]


def warmup() -> bool:
    """Send a tiny dummy image so Ollama loads the model into VRAM before the first real call."""
    try:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        call_vlm(dummy, prompt="Say 'ready' and nothing else.", timeout=VLM_WARMUP_TIMEOUT_SECONDS)
        return True
    except requests.RequestException as exc:
        log.warning("VLM warmup failed (%s) - is Ollama running with the model pulled?", exc)
        return False
    except Exception:
        log.exception("VLM warmup hit an unexpected error")
        return False


class VLMWorker:
    """Background thread that pulls (track_id, crop) jobs from a queue, calls the VLM, fires a callback.

    submit() drops jobs (and counts the drop) when the queue is full so the hot loop never blocks.
    """

    def __init__(self, on_result: Callable[[int, str], None], queue_size: int = 8):
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._on_result = on_result
        self._stop = threading.Event()
        self._dropped = 0
        # The worker's own HTTP session - never shared with the warmup() call.
        self._session = requests.Session()
        self._thread = threading.Thread(target=self._run, daemon=True, name="VLMWorker")
        self._thread.start()

    def submit(self, track_id: int, image: np.ndarray, prompt: str = VLM_PROMPT) -> bool:
        try:
            self._q.put_nowait((track_id, image, prompt))
            return True
        except queue.Full:
            self._dropped += 1
            if self._dropped % 30 == 1:
                log.warning("VLM queue full - dropped %d crops so far (Ollama can't keep up).", self._dropped)
            return False

    @property
    def dropped(self) -> int:
        """Total crops dropped because the queue was full - surfaced in the stats log."""
        return self._dropped

    @property
    def pending(self) -> int:
        """Crops currently queued and not yet processed - surfaced in the stats log.
        With the main-loop submit-gate working, this should sit at 0 once everyone
        on screen has a description."""
        return self._q.qsize()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            log.info("VLM worker still finishing an in-flight request - abandoning it (daemon thread).")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                tid, image, prompt = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                desc = call_vlm(image, prompt, session=self._session)
                if desc:
                    self._on_result(tid, desc)
            except requests.RequestException as exc:
                log.warning("VLM call failed for track %d: %s", tid, exc)
            except Exception:
                log.exception("VLM worker error for track %d", tid)


class DescriptionStore:
    """Thread-safe latest-description-per-track cache with TTL (monotonic clock)."""

    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._data: dict[int, tuple[float, str]] = {}

    def put(self, track_id: int, text: str) -> None:
        with self._lock:
            self._data[track_id] = (time.monotonic(), text)

    def get(self, track_id: int) -> str | None:
        with self._lock:
            entry = self._data.get(track_id)
        if entry is None:
            return None
        ts, text = entry
        if time.monotonic() - ts > self._ttl:
            return None
        return text

    def prune(self, alive_ids: set[int]) -> None:
        with self._lock:
            for tid in list(self._data.keys()):
                if tid not in alive_ids:
                    self._data.pop(tid, None)
