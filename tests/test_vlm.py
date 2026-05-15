"""Tests for ``cameralm.vlm`` - only the parts exercisable without a live Ollama.

Every network path is mocked: ``cameralm.vlm._session.post`` and the module-level
``cameralm.vlm.call_vlm`` are monkeypatched so no test ever touches a real server.
The monotonic clock used by ``DescriptionStore`` is monkeypatched via
``cameralm.vlm.time.monotonic`` to make TTL expiry deterministic.
"""

import threading
import time

import numpy as np
import pytest

import cameralm.vlm as vlm
from cameralm.config import VLM_MAX_DESCRIPTION_CHARS


def dummy_bgr():
    """A tiny dummy BGR frame - content is irrelevant since JPEG encode is real but cheap."""
    return np.zeros((32, 32, 3), dtype=np.uint8)


class _FakeResponse:
    """Stand-in for a ``requests`` Response: configurable ``.json()``, no-op ``raise_for_status``."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


# --------------------------------------------------------------------------
# DescriptionStore
# --------------------------------------------------------------------------


def test_description_store_put_then_get_returns_text():
    store = vlm.DescriptionStore(ttl_seconds=100.0)
    store.put(7, "person in red jacket")
    assert store.get(7) == "person in red jacket"


def test_description_store_get_missing_returns_none():
    store = vlm.DescriptionStore(ttl_seconds=100.0)
    assert store.get(999) is None


def test_description_store_get_expires_after_ttl(monkeypatch):
    """After more than ``ttl_seconds`` of monotonic time has passed, ``get`` returns None."""
    fake_now = {"t": 1000.0}
    monkeypatch.setattr(vlm.time, "monotonic", lambda: fake_now["t"])

    store = vlm.DescriptionStore(ttl_seconds=0.5)
    store.put(1, "still here")
    # Within the TTL window the entry is still served.
    fake_now["t"] = 1000.4
    assert store.get(1) == "still here"
    # Push the clock past the TTL - the entry is now considered stale.
    fake_now["t"] = 1001.0
    assert store.get(1) is None


def test_description_store_get_expires_after_real_sleep():
    """Same expiry behaviour, but driven by a real (tiny) sleep instead of a fake clock."""
    store = vlm.DescriptionStore(ttl_seconds=0.05)
    store.put(2, "fleeting")
    assert store.get(2) == "fleeting"  # served immediately
    time.sleep(0.12)
    assert store.get(2) is None


def test_description_store_prune_keeps_alive_drops_others():
    store = vlm.DescriptionStore(ttl_seconds=100.0)
    store.put(1, "alpha")
    store.put(2, "bravo")
    store.put(3, "charlie")

    store.prune({1, 3})

    assert store.get(1) == "alpha"
    assert store.get(3) == "charlie"
    assert store.get(2) is None


def test_description_store_prune_empty_set_drops_all():
    store = vlm.DescriptionStore(ttl_seconds=100.0)
    store.put(1, "alpha")
    store.put(2, "bravo")

    store.prune(set())

    assert store.get(1) is None
    assert store.get(2) is None


# --------------------------------------------------------------------------
# call_vlm
# --------------------------------------------------------------------------


def test_call_vlm_caps_response_length(monkeypatch):
    """A pathologically long ``response`` is truncated to VLM_MAX_DESCRIPTION_CHARS."""
    long_payload = {"response": "x" * 1000}
    monkeypatch.setattr(vlm._session, "post", lambda *a, **k: _FakeResponse(long_payload))

    result = vlm.call_vlm(dummy_bgr())

    assert isinstance(result, str)
    assert len(result) <= VLM_MAX_DESCRIPTION_CHARS
    # The cap is exactly VLM_MAX_DESCRIPTION_CHARS here since the source is 1000 'x's.
    assert result == "x" * VLM_MAX_DESCRIPTION_CHARS


def test_call_vlm_strips_and_returns_short_response(monkeypatch):
    """A normal short response comes back stripped and unmodified."""
    monkeypatch.setattr(
        vlm._session, "post", lambda *a, **k: _FakeResponse({"response": "  a calm description  "})
    )

    result = vlm.call_vlm(dummy_bgr())

    assert result == "a calm description"


def test_call_vlm_error_body_raises_runtimeerror(monkeypatch):
    """An Ollama error body ``{"error": ...}`` is surfaced as a RuntimeError."""
    monkeypatch.setattr(vlm._session, "post", lambda *a, **k: _FakeResponse({"error": "boom"}))

    with pytest.raises(RuntimeError):
        vlm.call_vlm(dummy_bgr())


def test_call_vlm_missing_response_key_returns_empty(monkeypatch):
    """A success body without a ``response`` key yields an empty string, not an error."""
    monkeypatch.setattr(vlm._session, "post", lambda *a, **k: _FakeResponse({}))

    assert vlm.call_vlm(dummy_bgr()) == ""


# --------------------------------------------------------------------------
# VLMWorker
# --------------------------------------------------------------------------


def test_vlm_worker_submit_returns_false_when_queue_full(monkeypatch):
    """When the worker thread is wedged inside call_vlm, a size-1 queue saturates and
    submit() returns False at least once.

    call_vlm is replaced with a function that blocks on an Event we never set until
    teardown, so the single worker thread cannot drain the queue. We then submit more
    items than the queue can hold; at least one submit must be rejected.
    """
    release = threading.Event()
    entered = threading.Event()

    def blocking_call_vlm(image, prompt=vlm.VLM_PROMPT):
        # Signal that the worker has pulled a job and is now stuck here.
        entered.set()
        release.wait(timeout=5.0)
        return "done"

    monkeypatch.setattr(vlm, "call_vlm", blocking_call_vlm)

    results = []
    worker = vlm.VLMWorker(on_result=lambda tid, desc: results.append((tid, desc)), queue_size=1)
    try:
        img = dummy_bgr()
        # First submit may be accepted, then immediately picked up by the worker
        # (which then blocks). Submit a generous number of jobs so that - regardless
        # of the exact interleaving - the size-1 queue is saturated and at least one
        # submit is rejected.
        outcomes = [worker.submit(i, img) for i in range(20)]
        assert any(outcome is False for outcome in outcomes), (
            "expected submit() to return False when the queue is saturated"
        )
    finally:
        # Unblock the in-flight call_vlm and shut the worker down cleanly.
        release.set()
        worker.stop()


def test_vlm_worker_stop_is_safe_to_call(monkeypatch):
    """A worker with no work submitted stops without hanging."""
    monkeypatch.setattr(vlm, "call_vlm", lambda image, prompt=vlm.VLM_PROMPT: "unused")
    worker = vlm.VLMWorker(on_result=lambda tid, desc: None, queue_size=4)
    try:
        pass
    finally:
        worker.stop()
    # join() inside stop() has a 2s timeout; if we got here the daemon thread exited.
    assert not worker._thread.is_alive()


def test_vlm_worker_processes_job_and_fires_callback(monkeypatch):
    """A submitted job flows through the worker and the on_result callback fires with the
    track id and the (mocked) description."""
    # The worker calls call_vlm(image, prompt, session=...) - the mock must
    # accept the session kwarg.
    monkeypatch.setattr(
        vlm,
        "call_vlm",
        lambda image, prompt=vlm.VLM_PROMPT, session=None: "mocked description",
    )

    got = []
    done = threading.Event()

    def on_result(tid, desc):
        got.append((tid, desc))
        done.set()

    worker = vlm.VLMWorker(on_result=on_result, queue_size=4)
    try:
        assert worker.submit(42, dummy_bgr()) is True
        # The worker runs on its own thread; give it a bounded window to deliver.
        assert done.wait(timeout=3.0), "worker did not invoke on_result in time"
        assert got == [(42, "mocked description")]
    finally:
        worker.stop()
