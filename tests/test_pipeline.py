"""Tests for the capture/inference pipeline worker's hardware-free surface.

`PipelineWorker.__init__` does not start the thread, so the publish/snapshot
slot and `crop_bbox` can be exercised directly without a camera or torch. The
camera loop itself (`_run`) needs hardware and is covered by the live runtime
stats log instead.
"""

import numpy as np

from cameralm.config import MAX_RESOLVES_PER_FRAME
from cameralm.pipeline import PipelineWorker, crop_bbox
from cameralm.tracking import TrackState
from cameralm.vlm import DescriptionStore


def _worker():
    # __init__ only stores deps + builds (unstarted) primitives - safe with None deps.
    return PipelineWorker(None, None, None, None, None, None, None)


def test_get_latest_is_none_before_first_publish():
    assert _worker().get_latest() is None
    assert _worker().failed is False


def test_publish_assigns_monotonic_frame_ids():
    w = _worker()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    w._publish(frame, {1: TrackState(bbox=(0, 0, 2, 2))})
    fid1, f1, info1 = w.get_latest()
    w._publish(frame, {1: TrackState(bbox=(0, 0, 2, 2))})
    fid2, _, _ = w.get_latest()
    assert (fid1, fid2) == (1, 2)
    assert f1 is frame
    assert set(info1) == {1}


def test_published_frame_info_is_an_isolated_snapshot():
    """The main thread must never read a TrackState the worker later mutates.

    The non-stale fast path in `_process_tracks` mutates the cached object's
    bbox/last_seen in place; the published snapshot must not move with it.
    """
    w = _worker()
    cached = TrackState(bbox=(0, 0, 2, 2))
    w._publish(np.zeros((4, 4, 3), dtype=np.uint8), {7: cached})
    _, _, snapshot = w.get_latest()

    cached.bbox = (9, 9, 9, 9)          # worker mutates its own cached copy
    cached.last_seen = 123.0
    assert snapshot[7].bbox == (0, 0, 2, 2)
    assert snapshot[7].last_seen == 0.0
    assert snapshot[7] is not cached


def test_crop_bbox_returns_copy_for_valid_box():
    frame = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
    crop = crop_bbox(frame, (2, 3, 6, 8))
    assert crop.shape == (5, 4, 3)
    crop[:] = 0                          # a copy - must not touch the source frame
    assert frame.any()


def test_crop_bbox_returns_none_for_empty_or_inverted_box():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert crop_bbox(frame, (5, 5, 5, 5)) is None      # zero-area
    assert crop_bbox(frame, (8, 8, 2, 2)) is None      # inverted
    assert crop_bbox(frame, (-5, -5, 0, 0)) is None    # fully off-frame


# --- _process_tracks: per-frame resolve cap (FPS must not scale with crowd) ---

def _process_worker(db):
    """A worker wired enough to run _process_tracks: real db + description store,
    no detector/embedders (resolve_track_identity is monkeypatched in the tests)."""
    return PipelineWorker(None, None, None, None, db, DescriptionStore(ttl_seconds=1.0), None)


def test_process_tracks_caps_resolves_per_frame(db, monkeypatch):
    """Inference cost must not scale with crowd size: _process_tracks re-embeds at
    most MAX_RESOLVES_PER_FRAME stale tracks per frame and defers the rest."""
    calls = []

    def _counting_resolve(bbox_t, frame, prev, now, *_a, **_k):
        calls.append(bbox_t)
        return TrackState(bbox=bbox_t, last_check=now, last_seen=now)

    monkeypatch.setattr("cameralm.pipeline.resolve_track_identity", _counting_resolve)
    worker = _process_worker(db)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    # Six brand-new tracks - all stale (no cache entry yet).
    tracks = [(tid, (tid, tid, tid + 20, tid + 40), 0.9) for tid in range(1, 7)]

    frame_info = worker._process_tracks(frame, tracks, now=100.0)

    assert len(calls) == MAX_RESOLVES_PER_FRAME      # only the cap got re-embedded
    assert len(frame_info) == 6                      # ...but every track is still shown
    # The deferred tracks carry a pending placeholder (last_check still 0.0).
    deferred = [ts for ts in frame_info.values() if ts.last_check == 0.0]
    assert len(deferred) == 6 - MAX_RESOLVES_PER_FRAME


def test_process_tracks_resolves_longest_waiting_first(db, monkeypatch):
    """When capped, the stale tracks that have waited longest (oldest last_check)
    are the ones re-embedded - so nothing starves."""
    calls = []

    def _counting_resolve(bbox_t, frame, prev, now, *_a, **_k):
        calls.append(prev.last_check if prev is not None else None)
        return TrackState(bbox=bbox_t, last_check=now, last_seen=now)

    monkeypatch.setattr("cameralm.pipeline.resolve_track_identity", _counting_resolve)
    worker = _process_worker(db)
    # Three stale cached tracks, different last_check ages (all well past the TTL).
    worker._track_cache = {
        1: TrackState(bbox=(0, 0, 10, 10), last_check=90.0, last_seen=99.9),
        2: TrackState(bbox=(0, 0, 10, 10), last_check=50.0, last_seen=99.9),
        3: TrackState(bbox=(0, 0, 10, 10), last_check=80.0, last_seen=99.9),
    }
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    tracks = [(tid, (0, 0, 10, 10), 0.9) for tid in (1, 2, 3)]

    worker._process_tracks(frame, tracks, now=100.0)

    # The MAX_RESOLVES_PER_FRAME oldest last_check values go first.
    oldest_first = sorted([90.0, 50.0, 80.0])
    assert sorted(calls) == oldest_first[:MAX_RESOLVES_PER_FRAME]
