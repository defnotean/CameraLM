import logging
import threading
import time

import cv2
import torch

from .admin import start_admin_server
from .body import BodyEmbedder
from .config import (
    ADMIN_ENABLED,
    ADMIN_PORT,
    AUTOSAVE_SECONDS,
    STATS_LOG_INTERVAL_SECONDS,
    USE_BODY_REID,
    USE_PARTIAL_REID,
    USE_VLM,
    VLM_DESC_TTL_SECONDS,
)
from .detector import PersonDetector
from .face import FaceEmbedder
from .identity_db import IdentityDB
from .logging_setup import configure_logging
from .naming import ClassEntry, NameEntry
from .partial import PartialAppearanceEmbedder
from .pipeline import PipelineWorker, crop_bbox
from .tracking import point_in_bbox
from .ui import (
    Renderer,
    draw_class_modal,
    draw_help,
    draw_naming_modal,
    draw_recording_indicator,
    draw_status_hud,
    draw_track,
)
from .vlm import DescriptionStore, VLMWorker, warmup as vlm_warmup

log = logging.getLogger(__name__)

WINDOW_NAME = "CameraLM"


def _bootstrap_partial_from_thumbnails(db: IdentityDB, partial: PartialAppearanceEmbedder | None) -> int:
    if partial is None:
        return 0
    seeded = 0
    for person in db.snapshot()["people"]:
        if person.get("n_partial", 0) > 0:
            continue
        pid = person["pid"]
        path = db.thumbnail_path(pid)
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            continue
        h, w = image.shape[:2]
        emb = partial.embed(image, [0, 0, w, h])
        if emb is not None:
            db.add_partial(pid, emb)
            seeded += 1
    if seeded:
        db.save()
    return seeded


def _window_closed() -> bool:
    """True once the user clicks the OS window-close button."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    configure_logging()
    if torch.cuda.is_available():
        log.info("CUDA available - GPU: %s. Pipeline models load onto the GPU.", torch.cuda.get_device_name(0))
    else:
        log.warning("CUDA NOT available - every model will run on CPU and the live view will be slow.")
    log.info("Loading models...")
    detector = PersonDetector()
    face = FaceEmbedder()
    body = BodyEmbedder() if USE_BODY_REID else None
    partial = PartialAppearanceEmbedder() if USE_PARTIAL_REID else None
    db = IdentityDB()
    seeded_partial = _bootstrap_partial_from_thumbnails(db, partial)
    if seeded_partial:
        log.info("Seeded %d partial appearance signatures from thumbnails.", seeded_partial)
    log.info("Loaded. %d known identities.", db.count_people())

    descriptions = DescriptionStore(ttl_seconds=VLM_DESC_TTL_SECONDS)
    vlm_worker = None
    if USE_VLM:
        log.info("Warming up VLM (Ollama)...")
        if vlm_warmup():
            vlm_worker = VLMWorker(on_result=descriptions.put)
            log.info("VLM ready.")
        else:
            log.warning("VLM unavailable - continuing without descriptions. Start Ollama and pull the VLM model.")

    if ADMIN_ENABLED:
        start_admin_server(db, port=ADMIN_PORT)
        log.info("Admin UI: http://localhost:%d", ADMIN_PORT)

    # Capture + detection + identity resolution run on the pipeline worker thread;
    # this thread only renders + handles input. The two halves overlap, so the
    # frame rate approaches 1/max(stage) instead of 1/sum(stage).
    pipeline = PipelineWorker(detector, face, body, partial, db, descriptions, vlm_worker)
    pipeline.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    log.info("Running. Press 'q' to quit.")

    mouse = {"x": -1, "y": -1, "click": None}
    mouse_lock = threading.Lock()  # the callback runs on the UI thread; guard the shared dict

    def on_mouse(event, x, y, flags, param):
        with mouse_lock:
            mouse["x"] = x
            mouse["y"] = y
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse["click"] = (x, y)

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    naming: NameEntry | ClassEntry | None = None
    last_save = time.monotonic()
    last_fps_t = time.monotonic()
    last_stats_log = time.monotonic()
    ema_dt = 0.0          # smoothed frame interval; fps is its reciprocal
    fps_ema = 0.0
    consecutive_errors = 0
    last_rendered_id = -1
    # Per-section wall-time accumulated between stats logs - render + display
    # only; the pipeline worker logs its own capture/detect/resolve line.
    t_acc = {"render": 0.0, "show": 0.0}
    stats_frames = 0

    try:
        while True:
            if _window_closed():
                break
            if pipeline.failed:
                log.error("Camera pipeline stopped (camera unavailable) - exiting.")
                break

            latest = pipeline.get_latest()
            if latest is None:
                # Worker hasn't produced its first frame yet - keep the window
                # alive and watch for an early quit.
                if cv2.waitKey(15) & 0xFF == ord("q"):
                    break
                continue

            frame_id, frame, frame_info = latest
            is_new = frame_id != last_rendered_id

            try:
                now = time.monotonic()
                alive_ids = frame_info.keys()

                # --- Mouse snapshot (the callback runs on this UI thread) ---
                with mouse_lock:
                    click = mouse["click"]
                    mouse["click"] = None
                    mx, my = mouse["x"], mouse["y"]

                # --- Click: start naming an unknown track ---
                if naming is None and click is not None:
                    cx, cy = click
                    for tid, info in frame_info.items():
                        if info.pid is None and point_in_bbox(cx, cy, info.bbox):
                            naming = NameEntry(
                                tid=tid,
                                face_emb=info.face_emb,
                                body_emb=info.body_emb,
                                partial_emb=info.partial_emb,
                                crop=crop_bbox(frame, info.bbox),
                            )
                            break

                # --- Auto-cancel the NAME stage if the subject left the frame.
                # Only NameEntry is tied to a live track; ClassEntry is exempt. ---
                if isinstance(naming, NameEntry):
                    if naming.tid not in alive_ids:
                        if naming.lost_since is None:
                            naming.lost_since = now
                        if now - naming.lost_since > 3.0:
                            log.info("Naming cancelled - subject left the frame.")
                            naming = None
                    else:
                        naming.lost_since = None

                # --- Hover ---
                hovered_tid = None
                if naming is None and mx >= 0:
                    for tid, info in frame_info.items():
                        if point_in_bbox(mx, my, info.bbox):
                            hovered_tid = tid
                            break

                # --- Render (only a NEW frame; Renderer.finish() returns a fresh
                # array, so re-rendering an unchanged frame would be wasted work) ---
                if is_new:
                    dt = now - last_fps_t
                    last_fps_t = now

                    t0 = time.perf_counter()
                    r = Renderer(frame)
                    for tid, info in frame_info.items():
                        is_unknown = info.pid is None or not db.has_person(info.pid)
                        label = "Unknown" if is_unknown else db.get_name(info.pid)
                        classes = db.classes_of(info.pid) if not is_unknown else None
                        draw_track(
                            r,
                            info.bbox,
                            label,
                            source=info.source,
                            conf=info.sim,
                            description=descriptions.get(tid),
                            hovered=(tid == hovered_tid),
                            is_unknown=is_unknown,
                            classes=classes,
                        )

                    # EMA the frame *interval* then invert - averaging
                    # instantaneous fps directly biases high under jitter
                    # (mean(1/dt) > 1/mean(dt)), inflating the HUD/stats number.
                    ema_dt = dt if ema_dt == 0 else 0.9 * ema_dt + 0.1 * dt
                    fps_ema = 1.0 / max(ema_dt, 1e-6)

                    # vlm_active is always False while USE_VLM is off; re-enabling
                    # the VLM would wire this through the pipeline worker.
                    draw_status_hud(r, fps_ema, n_people=db.count_people(), vlm_active=False)
                    draw_help(r)
                    draw_recording_indicator(r, len(frame_info))

                    if isinstance(naming, NameEntry):
                        draw_naming_modal(r, naming.buffer)
                    elif isinstance(naming, ClassEntry):
                        draw_class_modal(
                            r,
                            person_name=db.get_name(naming.pid),
                            buffer=naming.buffer,
                            assigned=naming.assigned,
                            suggestions=db.class_names(),
                        )

                    rendered = r.finish()
                    t_acc["render"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    cv2.imshow(WINDOW_NAME, rendered)
                    t_acc["show"] += time.perf_counter() - t0

                    stats_frames += 1
                    last_rendered_id = frame_id

                # --- waitKey: event pump + key source, every iteration. A slightly
                # longer wait when idle-spinning on an already-rendered frame. ---
                key = cv2.waitKey(1 if is_new else 5) & 0xFF

                # --- Key handling ---
                if naming is not None:
                    if key == 27:                       # Esc - exit naming entirely
                        naming = None
                    elif key in (13, 10):               # Enter
                        if isinstance(naming, NameEntry):
                            name = naming.buffer.strip()
                            if name:
                                pid = db.find_person_by_name(name)
                                created = pid is None
                                if pid is None:
                                    pid = db.create_person(name)
                                if naming.face_emb is not None:
                                    db.add_face(pid, naming.face_emb)
                                if naming.body_emb is not None:
                                    db.add_body(pid, naming.body_emb)
                                if naming.partial_emb is not None:
                                    db.add_partial(pid, naming.partial_emb)
                                if naming.crop is not None and (created or not db.has_thumbnail(pid)):
                                    db.save_thumbnail(pid, naming.crop)
                                db.save()
                                action = "Added" if created else "Updated"
                                log.info("%s '%s' as person %d.", action, db.get_name(pid), pid)
                                naming = ClassEntry(pid=pid)
                        else:                           # ClassEntry
                            name = naming.buffer.strip()
                            if not name:
                                naming = None
                            else:
                                db.add_to_class(naming.pid, name)
                                if name not in naming.assigned:
                                    naming.assigned.append(name)
                                naming.buffer = ""
                                db.save()
                    elif key == 8:                      # Backspace
                        naming.buffer = naming.buffer[:-1]
                    elif 32 <= key <= 126:              # Printable ASCII
                        if len(naming.buffer) < 40:
                            naming.buffer += chr(key)
                else:
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        db.save()
                        log.info("Identity DB saved.")

                if now - last_save > AUTOSAVE_SECONDS:
                    db.save()
                    last_save = now

                # --- Display stats (the worker logs its own capture line) ---
                if now - last_stats_log > STATS_LOG_INTERVAL_SECONDS:
                    n = max(stats_frames, 1)
                    log.info(
                        "display: %.1f fps | %d tracked | %d known | render=%.1fms show=%.1fms",
                        fps_ema,
                        len(frame_info),
                        db.count_people(),
                        1000.0 * t_acc["render"] / n,
                        1000.0 * t_acc["show"] / n,
                    )
                    for name in t_acc:
                        t_acc[name] = 0.0
                    stats_frames = 0
                    last_stats_log = now

                consecutive_errors = 0   # this iteration succeeded
            except Exception:
                consecutive_errors += 1
                # Log the first few, then only every 100th, so a persistent
                # per-frame bug doesn't bury the log in thousands of identical
                # tracebacks - but is still clearly visible.
                if consecutive_errors <= 3 or consecutive_errors % 100 == 0:
                    log.exception("Error in display loop (#%d) - skipping it.", consecutive_errors)
                # Defense-in-depth: if something throws every frame, sleep so the
                # loop degrades to slow-but-responsive instead of pegging the CPU.
                time.sleep(0.1)
                continue
    finally:
        pipeline.stop()
        if vlm_worker is not None:
            vlm_worker.stop()
        cv2.destroyAllWindows()
        try:
            db.save()
            log.info("Saved on exit.")
        except Exception:
            log.exception("Failed to save identity DB on exit.")


if __name__ == "__main__":
    main()
