"""Per-track identity state and the identity-resolution engine.

This is the system's core decision logic. It used to live inline in main.py's
hot loop as an untyped ~16-key dict threaded through ~35 call sites. `TrackState`
makes the shape explicit and statically checkable; `resolve_track_identity()` is
the resolver, now importable and unit-tested in tests/test_tracking.py.
"""

from dataclasses import dataclass

import numpy as np

from .config import (
    CACHE_TTL_BODY,
    CACHE_TTL_FACE,
    CACHE_TTL_UNKNOWN,
    FACE_LEARN_THRESH,
    PARTIAL_CONFIRM_HITS,
    SIDE_CONFIRM_HITS,
    TRACK_MEMORY_BODY_MIN_SIM,
    TRACK_MEMORY_PARTIAL_MIN_SIM,
    TRACK_MEMORY_SECONDS,
    TRACK_MEMORY_SIM,
)
from .identity_db import IdentityDB
from .matcher import decide_identity
from .types import IdentitySource


@dataclass
class TrackState:
    """Everything the loop remembers about one ByteTrack track between frames."""

    bbox: tuple[int, int, int, int]
    # Embeddings from the most recent resolve (carried so the naming flow can reuse them).
    face_emb: np.ndarray | None = None
    body_emb: np.ndarray | None = None
    partial_emb: np.ndarray | None = None
    # Current identity verdict.
    pid: int | None = None
    source: IdentitySource | None = None
    sim: float = 0.0
    # Weak-signal (partial / side) multi-hit confirmation.
    weak_candidate_pid: int | None = None
    weak_candidate_source: str | None = None
    weak_candidate_count: int = 0
    # Track-memory lock - keeps a confirmed identity stable while the person
    # turns away and loses face+body evidence.
    locked_pid: int | None = None
    locked_at: float = 0.0
    locked_source: str | None = None
    # Timestamps (monotonic clock).
    last_check: float = 0.0
    last_seen: float = 0.0


def bbox_iou(a, b) -> float:
    """Intersection-over-union of two xyxy boxes. Used to detect ByteTrack id reuse."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_bbox(x, y, bbox) -> bool:
    """True if (x, y) lies within the xyxy box (inclusive edges). Used for click/hover hit-testing."""
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def cache_ttl_for(source: IdentitySource | None) -> float:
    """How long an identity result stays valid for a given track source."""
    if source == IdentitySource.FACE:
        return CACHE_TTL_FACE
    if source in (IdentitySource.BODY, IdentitySource.SIDE, IdentitySource.TRACK):
        return CACHE_TTL_BODY
    return CACHE_TTL_UNKNOWN


def is_strong_identity_source(source: IdentitySource | None) -> bool:
    return source in (IdentitySource.FACE, IdentitySource.BODY)


def track_memory_identity(state: TrackState | None, now: float):
    """If `state` holds a still-valid track-memory lock, return (pid, 'track', sim)."""
    if state is None:
        return None, None, 0.0
    if state.locked_pid is None or now - state.locked_at > TRACK_MEMORY_SECONDS:
        return None, None, 0.0
    return state.locked_pid, IdentitySource.TRACK, TRACK_MEMORY_SIM


def track_memory_consistent(db: IdentityDB, pid: int, body_emb, partial_emb) -> bool:
    """Reject a held track lock when the current appearance no longer resembles it.

    When both body and partial signals are available, BOTH must agree - a single
    weak signal is not enough to keep a 5-second identity lock alive (a different
    person in similar clothing could otherwise inherit the lock).
    """
    checks = []
    body_sim = db.similarity_to_pid("body", pid, body_emb) if body_emb is not None else None
    partial_sim = db.similarity_to_pid("partial", pid, partial_emb) if partial_emb is not None else None
    if body_sim is not None:
        checks.append(body_sim >= TRACK_MEMORY_BODY_MIN_SIM)
    if partial_sim is not None:
        checks.append(partial_sim >= TRACK_MEMORY_PARTIAL_MIN_SIM)
    # No signal at all (person fully turned away) keeps the lock - that's what the
    # lock is for. But when signals exist, every one of them must agree.
    return True if not checks else all(checks)


def resolve_track_identity(
    bbox_t, frame, prev: TrackState | None, now: float, face, body, partial, db: IdentityDB
) -> TrackState:
    """Run the embedders + identity fusion for one stale track; return a fresh TrackState.

    `prev` is the track's previous state (or None for a brand-new track). The
    returned state carries the new identity verdict, refreshed embeddings, and an
    updated track-memory lock. Side effect: a sighting confirmed by a
    high-confidence FACE match is written back into `db` so the system keeps
    learning new views of known people - see the auto-learning policy below.
    """
    fe = face.embed(frame, bbox_t)
    be = body.embed(frame, bbox_t) if body else None
    pe = partial.embed(frame, bbox_t) if partial else None

    locked_pid = prev.locked_pid if prev else None
    locked_at = prev.locked_at if prev else 0.0
    locked_source = prev.locked_source if prev else None
    expected_pid = locked_pid if locked_pid is not None and now - locked_at <= TRACK_MEMORY_SECONDS else None

    pid, source, sim = decide_identity(fe, be, pe, db, expected_pid=expected_pid)

    weak_candidate_pid = None
    weak_candidate_source = None
    weak_candidate_count = 0
    if source in (IdentitySource.PARTIAL, IdentitySource.SIDE):
        weak_candidate_pid = pid
        weak_candidate_source = source
        if (
            prev is not None
            and prev.weak_candidate_pid == pid
            and prev.weak_candidate_source == source
        ):
            weak_candidate_count = prev.weak_candidate_count + 1
        else:
            weak_candidate_count = 1
        needed_hits = SIDE_CONFIRM_HITS if source == IdentitySource.SIDE else PARTIAL_CONFIRM_HITS
        if weak_candidate_count < needed_hits:
            pid, source, sim = None, None, 0.0

    if is_strong_identity_source(source):
        locked_pid = pid
        locked_at = now
        locked_source = source
    elif pid is None or source == IdentitySource.PARTIAL:
        held_pid, held_source, held_sim = track_memory_identity(prev, now)
        if held_pid is not None and track_memory_consistent(db, held_pid, be, pe):
            pid, source, sim = held_pid, held_source, held_sim
        elif held_pid is not None:
            locked_pid = None
            locked_at = 0.0
            locked_source = None

    # Correctness guard: a pid that was deleted (via the admin UI) between the
    # FAISS search and now must never be displayed, cached, or learned into.
    if pid is not None and not db.has_person(pid):
        pid, source, sim = None, None, 0.0
        locked_pid = None
        locked_at = 0.0
        locked_source = None

    state = TrackState(
        bbox=bbox_t,
        face_emb=fe,
        body_emb=be,
        partial_emb=pe,
        pid=pid,
        source=source,
        sim=sim,
        weak_candidate_pid=weak_candidate_pid,
        weak_candidate_source=weak_candidate_source,
        weak_candidate_count=weak_candidate_count,
        locked_pid=locked_pid,
        locked_at=locked_at,
        locked_source=locked_source,
        last_check=now,
        last_seen=now,
    )

    # --- Auto-learning policy ---
    # Only enroll new appearance data when THIS frame has a high-confidence FACE
    # match. A face match well above the display threshold is the one signal
    # trustworthy enough to permanently widen a person's stored identity.
    # Learning from body / side-fusion / track-memory is what let a single false
    # match snowball into a poisoned identity (FOLLOWUPS #6.3) - those paths are
    # deliberately gone. When a strong face DOES confirm the frame, the body and
    # partial vectors from that *same* frame are confirmed-this-person too, so
    # they are safe to learn alongside it.
    if (
        pid is not None
        and source == IdentitySource.FACE
        and sim >= FACE_LEARN_THRESH
        and db.has_person(pid)
    ):
        if fe is not None:
            db.add_face(pid, fe)
        if be is not None:
            db.add_body(pid, be)
        if pe is not None:
            db.add_partial(pid, pe)
    return state
