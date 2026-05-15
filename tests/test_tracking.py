"""Tests for cameralm/tracking.py - the TrackState dataclass and the identity
resolution engine (formerly an untyped ~16-key dict inline in main.py's loop).

These import only cameralm.tracking + cameralm.identity_db (no torch / GPU
models), so they run in lightweight CI.
"""

import numpy as np
import pytest

from cameralm.config import (
    CACHE_TTL_BODY,
    CACHE_TTL_FACE,
    CACHE_TTL_UNKNOWN,
    TRACK_MEMORY_SIM,
)
from cameralm.tracking import (
    TrackState,
    bbox_iou,
    cache_ttl_for,
    is_strong_identity_source,
    point_in_bbox,
    resolve_track_identity,
    track_memory_consistent,
)
from helpers import body_vec, face_vec, partial_vec


class _FakeEmbedder:
    """Stand-in for the face/body/partial embedders: returns a fixed vector (or None)."""

    def __init__(self, vec):
        self._vec = vec

    def embed(self, frame, bbox):
        return self._vec


_DUMMY_FRAME = np.zeros((64, 64, 3), dtype=np.uint8)
_BBOX = (10, 10, 50, 90)


# --- point_in_bbox ---

def test_point_in_bbox_inside_outside_edge():
    box = (0, 0, 100, 100)
    assert point_in_bbox(50, 50, box) is True
    assert point_in_bbox(150, 50, box) is False
    assert point_in_bbox(-1, 50, box) is False
    assert point_in_bbox(0, 0, box) is True        # corner is inclusive
    assert point_in_bbox(100, 100, box) is True    # opposite corner inclusive


# --- bbox_iou ---

def test_bbox_iou_identical_is_one():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_bbox_iou_disjoint_is_zero():
    assert bbox_iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_bbox_iou_partial_overlap():
    # Two 10x10 boxes overlapping in a 5x5 square: inter=25, union=175.
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_bbox_iou_zero_area_box_no_divide_by_zero():
    assert bbox_iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


# --- cache_ttl_for ---

def test_cache_ttl_for_each_source():
    assert cache_ttl_for("face") == CACHE_TTL_FACE
    assert cache_ttl_for("body") == CACHE_TTL_BODY
    assert cache_ttl_for("side") == CACHE_TTL_BODY
    assert cache_ttl_for("track") == CACHE_TTL_BODY
    assert cache_ttl_for("partial") == CACHE_TTL_UNKNOWN
    assert cache_ttl_for(None) == CACHE_TTL_UNKNOWN
    assert cache_ttl_for("anything-else") == CACHE_TTL_UNKNOWN


# --- is_strong_identity_source ---

def test_is_strong_identity_source():
    assert is_strong_identity_source("face") is True
    assert is_strong_identity_source("body") is True
    for weak in ("side", "partial", "track", None, ""):
        assert is_strong_identity_source(weak) is False


# --- TrackState ---

def test_track_state_defaults():
    s = TrackState(bbox=(0, 0, 10, 10))
    assert s.bbox == (0, 0, 10, 10)
    assert s.pid is None
    assert s.source is None
    assert s.sim == 0.0
    assert s.locked_pid is None
    assert s.locked_at == 0.0
    assert s.face_emb is None and s.body_emb is None and s.partial_emb is None


def test_track_state_is_mutable():
    s = TrackState(bbox=(0, 0, 10, 10))
    s.bbox = (1, 1, 2, 2)
    s.pid = 7
    s.last_seen = 123.0
    assert s.bbox == (1, 1, 2, 2)
    assert s.pid == 7
    assert s.last_seen == 123.0


# --- track_memory_consistent ---

def test_track_memory_consistent_no_signal_keeps_lock(db):
    pid = db.create_person("Locked")
    db.add_body(pid, body_vec(seed=1))
    # No body/partial embeddings to check against -> the lock is kept.
    assert track_memory_consistent(db, pid, None, None) is True


def test_track_memory_consistent_far_body_breaks_lock(db):
    pid = db.create_person("Locked")
    db.add_body(pid, body_vec(seed=1))
    # A body embedding nothing like the enrolled one -> lock is not consistent.
    assert track_memory_consistent(db, pid, body_vec(seed=999), None) is False


def test_track_memory_consistent_matching_body_keeps_lock(db):
    pid = db.create_person("Locked")
    bv = body_vec(seed=1)
    db.add_body(pid, bv)
    assert track_memory_consistent(db, pid, bv, None) is True


# --- resolve_track_identity (the core engine) ---

def test_resolve_new_track_strong_face_identifies_and_locks(db):
    fv = face_vec(seed=3)
    pid = db.create_person("Ada")
    db.add_face(pid, fv)

    state = resolve_track_identity(
        _BBOX, _DUMMY_FRAME, None, 100.0,
        face=_FakeEmbedder(fv), body=_FakeEmbedder(None), partial=_FakeEmbedder(None), db=db,
    )
    assert state.pid == pid
    assert state.source == "face"
    assert state.locked_pid == pid          # a strong source sets the track-memory lock
    assert state.locked_at == 100.0
    assert state.face_emb is fv             # embeddings are carried on the state
    assert state.last_check == 100.0 and state.last_seen == 100.0


def test_resolve_new_track_body_only_does_not_identify(db):
    bv = body_vec(seed=4)
    pid = db.create_person("Bob")
    db.add_body(pid, bv)

    state = resolve_track_identity(
        _BBOX, _DUMMY_FRAME, None, 50.0,
        face=_FakeEmbedder(None), body=_FakeEmbedder(bv), partial=_FakeEmbedder(None), db=db,
    )
    # REQUIRE_FACE_FOR_NEW_TRACK: a brand-new track can't be identified from body alone.
    assert state.pid is None
    assert state.source is None
    assert state.body_emb is bv             # still carried, just not trusted as identity


def test_resolve_holds_identity_via_track_memory(db):
    fv = face_vec(seed=5)
    pid = db.create_person("Cleo")
    db.add_face(pid, fv)

    # prev: this track was confidently locked to `pid` half a second ago.
    prev = TrackState(bbox=_BBOX, locked_pid=pid, locked_at=99.5, last_check=99.5, last_seen=99.5)
    # This frame: no signal at all (person turned away).
    state = resolve_track_identity(
        _BBOX, _DUMMY_FRAME, prev, 100.0,
        face=_FakeEmbedder(None), body=_FakeEmbedder(None), partial=_FakeEmbedder(None), db=db,
    )
    assert state.pid == pid
    assert state.source == "track"
    assert state.sim == TRACK_MEMORY_SIM


def test_resolve_expired_lock_is_not_held(db):
    fv = face_vec(seed=6)
    pid = db.create_person("Dave")
    db.add_face(pid, fv)

    # prev: the lock is older than TRACK_MEMORY_SECONDS.
    prev = TrackState(bbox=_BBOX, locked_pid=pid, locked_at=90.0, last_check=90.0, last_seen=90.0)
    state = resolve_track_identity(
        _BBOX, _DUMMY_FRAME, prev, 100.0,
        face=_FakeEmbedder(None), body=_FakeEmbedder(None), partial=_FakeEmbedder(None), db=db,
    )
    assert state.pid is None        # an expired lock is not held


def test_resolve_unknown_when_no_signal_and_no_prev(db):
    state = resolve_track_identity(
        _BBOX, _DUMMY_FRAME, None, 10.0,
        face=_FakeEmbedder(None), body=_FakeEmbedder(None), partial=_FakeEmbedder(None), db=db,
    )
    assert state.pid is None
    assert state.source is None
    assert state.last_check == 10.0


# --- auto-learning policy (only a high-confidence FACE match may enroll) ---

def test_resolve_strong_face_learns_body_and_partial_from_same_frame(db):
    """A high-confidence FACE match enrolls the body + partial vectors from the
    SAME frame - they are confirmed-this-person too."""
    fv, bv, pv = face_vec(seed=3), body_vec(seed=4), partial_vec(seed=5)
    pid = db.create_person("Ada")
    db.add_face(pid, fv)                      # enrolled with face only
    assert db.body.count_for(pid) == 0 and db.partial.count_for(pid) == 0

    resolve_track_identity(
        _BBOX, _DUMMY_FRAME, None, 100.0,
        face=_FakeEmbedder(fv), body=_FakeEmbedder(bv), partial=_FakeEmbedder(pv), db=db,
    )
    # The exact face vector dedups (no-op); the new body + partial are learned.
    assert db.face.count_for(pid) == 1
    assert db.body.count_for(pid) == 1
    assert db.partial.count_for(pid) == 1


def test_resolve_track_source_does_not_auto_learn(db):
    """A track-memory (TRACK source) sighting must NEVER enroll new embeddings -
    a held lock that happens to be wrong would otherwise poison the identity.
    Regression guard for the misidentification feedback loop (FOLLOWUPS #6.3)."""
    fv, bv, pv = face_vec(seed=6), body_vec(seed=7), partial_vec(seed=8)
    pid = db.create_person("Cleo")
    db.add_face(pid, fv)                      # face only - 0 body, 0 partial

    # prev: confidently locked half a second ago. This frame: no face, but body
    # + partial embeddings ARE produced - the old code would have learned them.
    prev = TrackState(bbox=_BBOX, locked_pid=pid, locked_at=99.5, last_check=99.5, last_seen=99.5)
    state = resolve_track_identity(
        _BBOX, _DUMMY_FRAME, prev, 100.0,
        face=_FakeEmbedder(None), body=_FakeEmbedder(bv), partial=_FakeEmbedder(pv), db=db,
    )
    assert state.source == "track"            # identity held via track memory
    assert db.body.count_for(pid) == 0        # ...but nothing was enrolled
    assert db.partial.count_for(pid) == 0
    assert db.face.count_for(pid) == 1
