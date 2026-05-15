from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Webcam ---
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# Consecutive failed frame reads before the loop tries to reconnect the camera.
CAMERA_MAX_READ_FAILURES = 10
# How many times to retry opening the camera (with backoff) before giving up.
CAMERA_RECONNECT_ATTEMPTS = 5

# --- Detection + tracking ---
YOLO_MODEL = "yolov8n.pt"
DETECT_CONF = 0.5
TRACKER_CONFIG = "bytetrack.yaml"
# YOLO inference size. The camera feed is 640x480 and people are at
# conversational distance (large in frame), so 512 trims inference cost ~0.6x
# vs the 640 default with negligible accuracy loss at this range. The pose
# fallback backstops any small/partial detections this might miss.
YOLO_IMGSZ = 512
# Run YOLO detection on every Nth inference frame; reuse the previous boxes in
# between. ByteTrack ids stay stable (it is framerate-agnostic) - the box just
# lags one inference frame (~30ms, a few px).
# Defaults to 1 (detect every frame): with the 3-thread pipeline the inference
# stage (detect ~21ms + resolve ~9ms ≈ 30ms) already keeps up with the 30 fps
# MJPG camera, so skipping detection buys no fps - it would only add box lag.
# Raise to 2+ only if detection/resolution grow past the camera frame interval
# (heavier models, a faster camera, or a higher resolution).
DETECT_EVERY_N = 1

# Pose fallback catches some partial-person views that standard person detection misses.
USE_POSE_PARTIAL_DETECT = True
YOLO_POSE_MODEL = "yolov8n-pose.pt"
POSE_DETECT_CONF = 0.35
POSE_TRACK_ID_OFFSET = 1_000_000
# The pose pass is a full second YOLO inference - only run it every Nth frame.
# Every 10th (vs 5th) halves its amortized cost; still ~1.5 passes/sec at 15fps,
# enough to catch partial-person views the main detector misses.
POSE_DETECT_EVERY = 10

# --- Face recognition (InsightFace / ArcFace) ---
FACE_MODEL_PACK = "buffalo_l"
FACE_DET_SIZE = (640, 640)
FACE_DIM = 512
# Cosine-similarity bar to DISPLAY a face match. ArcFace (buffalo_l) puts
# genuine same-person pairs around 0.5-0.8 and different-person pairs usually
# below ~0.4 - but a low-res webcam crop, a hard angle, or two similar-looking
# people (a classroom of kids) can push a cross-person pair to ~0.45. 0.44 was
# inside that danger zone and caused steady false positives; 0.52 clears it
# while still comfortably accepting the real person.
FACE_MATCH_THRESH = 0.52
# The match must also beat the next-best DIFFERENT person by this margin - stops
# "this face is sort of like two enrolled people" from resolving to either.
FACE_MATCH_MARGIN = 0.08
FACE_WEAK_MATCH_THRESH = 0.35
# Cosine-similarity bar to LEARN (permanently enroll) a face into a person.
# Deliberately higher than FACE_MATCH_THRESH: it is fine to *show* a 0.52 match,
# but only a near-certain match may widen someone's stored identity forever.
# Decoupling "confident enough to display" from "confident enough to enroll" is
# what stops one false match from snowballing into a poisoned identity
# (FOLLOWUPS #6.3).
FACE_LEARN_THRESH = 0.62
FACE_UPPER_CROP_RATIO = 0.62
FACE_PROFILE_FLIP_FALLBACK = True

# --- High-precision identity policy ---
# In classroom/doorway use, students may wear similar clothing and overlap.
# New tracks should not be identified from body, side-profile, or partial
# appearance alone; those signals are used only after a face has already locked
# that same track.
REQUIRE_FACE_FOR_NEW_TRACK = True

# --- Body ReID (boxmot / OSNet) ---
USE_BODY_REID = True
REID_WEIGHTS = "osnet_x0_25_msmt17.pt"   # boxmot auto-downloads from its model zoo
REID_DIM = 512
REID_MATCH_THRESH = 0.82
REID_MATCH_MARGIN = 0.045
REID_WEAK_MATCH_THRESH = 0.72
# Stricter body-margin floor for the multi-signal acceptance of a brand-new
# (unlocked) track - the classroom false-positive path, where similar uniforms
# make a low margin dangerous. Previously an inline 0.070 literal in matcher.py.
REID_NEWTRACK_MATCH_MARGIN = 0.070

# --- Partial appearance matching ---
# Weak fallback for occluded views where face/full-body ReID is unavailable.
# Uses visible-region color/texture signatures, so it is useful for heads, arms,
# torsos, backpacks, and clothing fragments, but gated by a high uniqueness margin.
USE_PARTIAL_REID = True
PARTIAL_DIM = 192
PARTIAL_MATCH_THRESH = 0.88
PARTIAL_MATCH_MARGIN = 0.070
PARTIAL_WEAK_MATCH_THRESH = 0.84
PARTIAL_WEAK_MATCH_MARGIN = 0.060
PARTIAL_MIN_PIXELS = 1200
PARTIAL_CONFIRM_HITS = 2
SIDE_CONFIRM_HITS = 3

# --- Track continuity ---
# Keeps a strongly identified person stable while they turn away or briefly lose
# face/body evidence. It no longer auto-LEARNS from a held track - a held lock
# that happens to be wrong would enroll the wrong person's body/clothing for up
# to 5 seconds. Learning is now gated on a high-confidence face only
# (FACE_LEARN_THRESH; see resolve_track_identity).
TRACK_MEMORY_SECONDS = 5.0
TRACK_MEMORY_SIM = 0.82
# Minimum similarity for a held lock to survive. Raised from 0.48 - that bar was
# low enough that a different person in similar clothing could inherit a lock.
TRACK_MEMORY_BODY_MIN_SIM = 0.62
TRACK_MEMORY_PARTIAL_MIN_SIM = 0.72
# Below this IoU between a track's last box and its new box, assume ByteTrack
# reused the id for a different person and discard the cached identity/lock.
TRACK_REUSE_MIN_IOU = 0.15
# ...but only when the two boxes are from near-consecutive frames. A big jump
# after a longer gap is just the person moving while briefly untracked, not reuse.
TRACK_REUSE_MAX_GAP_SECONDS = 0.4

# --- VLM (Ollama-hosted Qwen3-VL) ---
# OFF. On a 4 GB / single-GPU box the VLM (Ollama) cannot run alongside the four
# CV models without crippling the live framerate: on the GPU it exhausts VRAM,
# on the CPU it eats the cores the camera loop needs (measured ~2.6x the app's
# own CPU). Disabling it frees ~2.7 GB VRAM and ~6 CPU cores for the
# detection/recognition pipeline. vlm.py and the VLM_* settings below are kept
# dormant - flip this back to True to re-enable on an 8 GB+ GPU.
USE_VLM = False
OLLAMA_HOST = "http://localhost:11434"
VLM_MODEL = "qwen3-vl:2b"
VLM_PROMPT = (
    "Describe this person in 10 words or fewer. "
    "Mention clothing, posture, and any object they are holding. "
    "Reply with just the description, no preamble."
)
# Re-submit debounce, NOT a target rate: the loop only submits a crop for a
# track that has no current description (see main.py), so this just stops it
# re-submitting into the window before the first (slow) CPU call comes back.
VLM_INTERVAL_SECONDS = 12.0
# CPU inference is far slower than GPU - a real call is 15-25s on this box, so
# the old 12s budget timed out every call and no description ever landed.
VLM_TIMEOUT_SECONDS = 60.0
VLM_WARMUP_TIMEOUT_SECONDS = 45.0  # cold CPU model load; generous enough not to
                                   # flake startup, still bounded.
# A person's physical description is stable for as long as they're on screen,
# so hold it far longer than one inference takes - otherwise it expires before
# the next call can refresh it and the overlay is always blank.
VLM_DESC_TTL_SECONDS = 45.0
VLM_KEEP_ALIVE = "10m"       # keep the model resident between sparse calls (avoids reload stalls)
VLM_NUM_CTX = 768            # small context window - one image + a short prompt
VLM_NUM_PREDICT = 60         # cap generated tokens for short descriptions
VLM_MAX_DESCRIPTION_CHARS = 200  # hard cap on stored/rendered description length
# Keep the VLM OFF the GPU. On a 4 GB card, qwen3-vl:2b (~3.2 GB) crowds out the
# YOLO / InsightFace / OSNet models and craters the camera framerate. The VLM
# worker is async + throttled + drops under load, so a slower CPU-only model is
# an easy trade for keeping the live view fast. Set True only on an 8 GB+ GPU.
VLM_USE_GPU = False
# CPU-only, so it competes with the camera loop for cores. Cap Ollama's thread
# count at roughly half the box (12 logical cores here) so the live pipeline
# always keeps a CPU floor even while the VLM is mid-inference. None = no cap.
VLM_NUM_THREADS = 6
# Downscale person crops so the longer side is <= this before JPEG-encoding for
# the VLM. Fewer vision tokens => faster CPU inference and smaller payloads,
# with no real loss for a 10-word physical description.
VLM_IMAGE_MAX_PX = 448

# --- Identity DB ---
IDENTITY_FILE = DATA_DIR / "identities.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"
MAX_EMBEDDINGS_PER_PERSON = 20
# Partial = HSV colour/texture histograms (essentially clothing colour). 40
# stored signatures per person was a very wide net - any passer-by in similar
# colours could weak-match. 12 keeps it as a short-range tie-breaker, not a
# promiscuous identity signal. (FOLLOWUPS #5: partial/body are clothing-dependent
# and shouldn't carry identity weight on their own.)
MAX_PARTIAL_EMBEDDINGS_PER_PERSON = 12
FACE_DUPLICATE_SIM = 0.985
BODY_DUPLICATE_SIM = 0.92
PARTIAL_DUPLICATE_SIM = 0.96

# --- Identity cache (perf optimization) ---
# How long an identity result stays valid per track before we re-embed.
# A face-locked track barely changes, so it can cache long; an unknown needs
# frequent re-checks so it gets identified (and named) quickly. Differentiating
# these is a major part of the person-present framerate fix.
CACHE_TTL_UNKNOWN = 0.5
CACHE_TTL_BODY = 2.0
CACHE_TTL_FACE = 3.0
# Cap on how many stale tracks are re-embedded (face + body + partial) in a
# single inference frame. Each resolve is a per-person GPU+CPU cost, so
# resolving every stale track at once made the framerate fall as more people
# entered view. With a cap the per-frame cost is bounded no matter the crowd
# size; tracks past the cap keep their last identity and resolve on a following
# frame (longest-waiting first, so nobody starves).
# 1 = the framerate is completely independent of how many people are on screen
# (a lone new person is still identified on the very next frame; in a crowd each
# person just waits ~1 extra frame to be identified). Raise to 2-3 for snappier
# crowd identification, at the cost of a small framerate step as people enter.
MAX_RESOLVES_PER_FRAME = 1

# --- Admin server ---
ADMIN_ENABLED = True
ADMIN_HOST = "127.0.0.1"   # loopback only - never bind 0.0.0.0 for a biometric admin UI
ADMIN_PORT = 8765

# --- UI ---
AUTOSAVE_SECONDS = 30

# --- Diagnostics ---
# Periodic INFO log of fps / tracked people / VLM submit-drop count. Makes a
# framerate regression measurable from the log instead of guesswork.
STATS_LOG_INTERVAL_SECONDS = 10.0

# --- Schedule ---
# Shape of the school-style rotation the admin UI uses for assigning classes
# to time slots. 6 days x 4 blocks (A-D) = 24 cells in one cycle. Changing
# these reshapes the grid in the admin UI and the persisted schedule; cells
# that fall outside the new dimensions are dropped on the next load.
SCHEDULE_DAYS = 6
SCHEDULE_BLOCKS = ("A", "B", "C", "D")

# --- Privacy / retention ---
# How many days of inactivity before a person's biometric vectors are dropped
# by the retention sweep. The person's name + class memberships survive (so the
# audit trail still reads sensibly), only their face/body/partial embeddings go.
# 0 disables the sweep entirely (good for dev; production deployments should
# pick a real number aligned with the local retention policy).
RETENTION_DAYS = 0
# If True, run a retention sweep once on startup. Idempotent - safe to leave on
# even when RETENTION_DAYS=0 (it becomes a no-op).
RETENTION_PURGE_ON_STARTUP = True
# If True, a person whose consent status is not "granted" is shown as "Unknown"
# in the live overlay even when the pipeline has identified them. Their stored
# vectors still drive the match, so flipping consent to granted in the admin
# surfaces them immediately - the gate is purely on display, not on processing.
# Default False so the dev workflow (enroll then keep using) is unchanged;
# real classroom deployments should set this True and use the admin UI to
# attest consent for each person before recognition surfaces.
REQUIRE_CONSENT_FOR_RECOGNITION = False


# --- Invariant checks (P5) ---
# These constants are flat (every module imports them by name), but the values
# are not independent - a weak threshold above its strong threshold, a negative
# interval, an out-of-range port, etc. would cause subtly wrong behaviour far
# from here. `_validate_config()` runs at import and fails fast with a precise
# message instead, so a bad edit is caught at startup, not in production.


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"config.py: {message}")


def _validate_config() -> None:
    # Webcam
    _require(FRAME_WIDTH > 0 and FRAME_HEIGHT > 0, "FRAME_WIDTH / FRAME_HEIGHT must be positive")
    _require(CAMERA_MAX_READ_FAILURES >= 1, "CAMERA_MAX_READ_FAILURES must be >= 1")
    _require(CAMERA_RECONNECT_ATTEMPTS >= 1, "CAMERA_RECONNECT_ATTEMPTS must be >= 1")

    # Detection + tracking
    _require(0.0 < DETECT_CONF <= 1.0, "DETECT_CONF must be in (0, 1]")
    _require(0.0 < POSE_DETECT_CONF <= 1.0, "POSE_DETECT_CONF must be in (0, 1]")
    _require(YOLO_IMGSZ > 0, "YOLO_IMGSZ must be positive")
    _require(DETECT_EVERY_N >= 1, "DETECT_EVERY_N must be >= 1")
    _require(POSE_DETECT_EVERY >= 1, "POSE_DETECT_EVERY must be >= 1")

    # Face recognition
    _require(FACE_DIM > 0, "FACE_DIM must be positive")
    _require(0.0 < FACE_MATCH_THRESH <= 1.0, "FACE_MATCH_THRESH must be in (0, 1]")
    _require(0.0 <= FACE_MATCH_MARGIN <= 1.0, "FACE_MATCH_MARGIN must be in [0, 1]")
    _require(0.0 < FACE_WEAK_MATCH_THRESH <= 1.0, "FACE_WEAK_MATCH_THRESH must be in (0, 1]")
    _require(
        FACE_WEAK_MATCH_THRESH < FACE_MATCH_THRESH,
        "FACE_WEAK_MATCH_THRESH must be < FACE_MATCH_THRESH (weak signal can't outrank strong)",
    )
    _require(0.0 < FACE_LEARN_THRESH <= 1.0, "FACE_LEARN_THRESH must be in (0, 1]")
    _require(
        FACE_MATCH_THRESH <= FACE_LEARN_THRESH,
        "FACE_LEARN_THRESH must be >= FACE_MATCH_THRESH (don't enroll from matches too weak to display)",
    )
    _require(0.0 < FACE_UPPER_CROP_RATIO <= 1.0, "FACE_UPPER_CROP_RATIO must be in (0, 1]")

    # Body ReID
    _require(REID_DIM > 0, "REID_DIM must be positive")
    _require(0.0 < REID_MATCH_THRESH <= 1.0, "REID_MATCH_THRESH must be in (0, 1]")
    _require(0.0 <= REID_MATCH_MARGIN <= 1.0, "REID_MATCH_MARGIN must be in [0, 1]")
    _require(0.0 < REID_WEAK_MATCH_THRESH <= 1.0, "REID_WEAK_MATCH_THRESH must be in (0, 1]")
    _require(
        REID_WEAK_MATCH_THRESH < REID_MATCH_THRESH,
        "REID_WEAK_MATCH_THRESH must be < REID_MATCH_THRESH",
    )
    _require(0.0 <= REID_NEWTRACK_MATCH_MARGIN <= 1.0, "REID_NEWTRACK_MATCH_MARGIN must be in [0, 1]")

    # Partial appearance
    _require(PARTIAL_DIM > 0, "PARTIAL_DIM must be positive")
    _require(0.0 < PARTIAL_MATCH_THRESH <= 1.0, "PARTIAL_MATCH_THRESH must be in (0, 1]")
    _require(0.0 <= PARTIAL_MATCH_MARGIN <= 1.0, "PARTIAL_MATCH_MARGIN must be in [0, 1]")
    _require(0.0 < PARTIAL_WEAK_MATCH_THRESH <= 1.0, "PARTIAL_WEAK_MATCH_THRESH must be in (0, 1]")
    _require(
        PARTIAL_WEAK_MATCH_THRESH < PARTIAL_MATCH_THRESH,
        "PARTIAL_WEAK_MATCH_THRESH must be < PARTIAL_MATCH_THRESH",
    )
    _require(0.0 <= PARTIAL_WEAK_MATCH_MARGIN <= 1.0, "PARTIAL_WEAK_MATCH_MARGIN must be in [0, 1]")
    _require(PARTIAL_MIN_PIXELS > 0, "PARTIAL_MIN_PIXELS must be positive")
    _require(PARTIAL_CONFIRM_HITS >= 1, "PARTIAL_CONFIRM_HITS must be >= 1")
    _require(SIDE_CONFIRM_HITS >= 1, "SIDE_CONFIRM_HITS must be >= 1")

    # Track continuity
    _require(TRACK_MEMORY_SECONDS > 0, "TRACK_MEMORY_SECONDS must be positive")
    _require(0.0 < TRACK_MEMORY_SIM <= 1.0, "TRACK_MEMORY_SIM must be in (0, 1]")
    _require(0.0 < TRACK_MEMORY_BODY_MIN_SIM <= 1.0, "TRACK_MEMORY_BODY_MIN_SIM must be in (0, 1]")
    _require(0.0 < TRACK_MEMORY_PARTIAL_MIN_SIM <= 1.0, "TRACK_MEMORY_PARTIAL_MIN_SIM must be in (0, 1]")
    _require(0.0 <= TRACK_REUSE_MIN_IOU <= 1.0, "TRACK_REUSE_MIN_IOU must be in [0, 1]")
    _require(TRACK_REUSE_MAX_GAP_SECONDS > 0, "TRACK_REUSE_MAX_GAP_SECONDS must be positive")

    # VLM
    _require(VLM_INTERVAL_SECONDS > 0, "VLM_INTERVAL_SECONDS must be positive")
    _require(VLM_TIMEOUT_SECONDS > 0, "VLM_TIMEOUT_SECONDS must be positive")
    _require(VLM_WARMUP_TIMEOUT_SECONDS > 0, "VLM_WARMUP_TIMEOUT_SECONDS must be positive")
    _require(VLM_DESC_TTL_SECONDS > 0, "VLM_DESC_TTL_SECONDS must be positive")
    _require(VLM_NUM_CTX > 0, "VLM_NUM_CTX must be positive")
    _require(VLM_NUM_PREDICT > 0, "VLM_NUM_PREDICT must be positive")
    _require(VLM_MAX_DESCRIPTION_CHARS > 0, "VLM_MAX_DESCRIPTION_CHARS must be positive")
    _require(VLM_NUM_THREADS is None or VLM_NUM_THREADS >= 1, "VLM_NUM_THREADS must be None or >= 1")
    _require(VLM_IMAGE_MAX_PX > 0, "VLM_IMAGE_MAX_PX must be positive")

    # Identity DB
    _require(MAX_EMBEDDINGS_PER_PERSON >= 1, "MAX_EMBEDDINGS_PER_PERSON must be >= 1")
    _require(MAX_PARTIAL_EMBEDDINGS_PER_PERSON >= 1, "MAX_PARTIAL_EMBEDDINGS_PER_PERSON must be >= 1")
    _require(0.0 < FACE_DUPLICATE_SIM <= 1.0, "FACE_DUPLICATE_SIM must be in (0, 1]")
    _require(0.0 < BODY_DUPLICATE_SIM <= 1.0, "BODY_DUPLICATE_SIM must be in (0, 1]")
    _require(0.0 < PARTIAL_DUPLICATE_SIM <= 1.0, "PARTIAL_DUPLICATE_SIM must be in (0, 1]")

    # Identity cache
    _require(CACHE_TTL_UNKNOWN > 0, "CACHE_TTL_UNKNOWN must be positive")
    _require(CACHE_TTL_BODY > 0, "CACHE_TTL_BODY must be positive")
    _require(CACHE_TTL_FACE > 0, "CACHE_TTL_FACE must be positive")
    _require(MAX_RESOLVES_PER_FRAME >= 1, "MAX_RESOLVES_PER_FRAME must be >= 1")

    # Admin server / UI / diagnostics
    _require(1 <= ADMIN_PORT <= 65535, "ADMIN_PORT must be in [1, 65535]")
    _require(AUTOSAVE_SECONDS > 0, "AUTOSAVE_SECONDS must be positive")
    _require(STATS_LOG_INTERVAL_SECONDS > 0, "STATS_LOG_INTERVAL_SECONDS must be positive")

    # Schedule
    _require(SCHEDULE_DAYS >= 1, "SCHEDULE_DAYS must be >= 1")
    _require(len(SCHEDULE_BLOCKS) >= 1, "SCHEDULE_BLOCKS must be non-empty")
    _require(
        all(isinstance(b, str) and b.strip() for b in SCHEDULE_BLOCKS),
        "every entry in SCHEDULE_BLOCKS must be a non-empty string",
    )

    # Privacy / retention
    _require(RETENTION_DAYS >= 0, "RETENTION_DAYS must be >= 0 (0 disables the sweep)")


_validate_config()
