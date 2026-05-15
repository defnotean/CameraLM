"""Tests for the typed naming-FSM dataclasses (cameralm/naming.py).

These exist because an untyped stage-keyed dict previously caused a per-frame
KeyError that froze the app: the "name" stage carried a `tid` key, the "class"
stage did not, and a code path read `naming["tid"]` unconditionally. The
dataclasses turn that into a structural impossibility.
"""

import numpy as np

from cameralm.naming import ClassEntry, NameEntry, NamingState


def test_name_entry_defaults():
    e = NameEntry(tid=7, face_emb=None, body_emb=None, partial_emb=None, crop=None)
    assert e.tid == 7
    assert e.buffer == ""
    assert e.lost_since is None


def test_name_entry_carries_embeddings_and_buffer():
    fe = np.zeros(4, dtype=np.float32)
    e = NameEntry(tid=1, face_emb=fe, body_emb=None, partial_emb=None, crop=None, buffer="Al")
    assert e.face_emb is fe
    assert e.buffer == "Al"


def test_class_entry_defaults():
    e = ClassEntry(pid=3)
    assert e.pid == 3
    assert e.buffer == ""
    assert e.assigned == []


def test_class_entry_assigned_lists_are_independent():
    """field(default_factory=list): each ClassEntry must get its own list."""
    a = ClassEntry(pid=1)
    b = ClassEntry(pid=2)
    a.assigned.append("Grade 8A")
    assert b.assigned == []          # not a shared mutable default


def test_class_entry_has_no_tid():
    """Regression guard for the exact bug that froze the app.

    The original code did `naming["tid"]` on the class stage. With dataclasses,
    a ClassEntry simply has no `tid` attribute - reading it is an AttributeError
    at that line (and a static type error), not a per-frame crash in a hot loop.
    """
    e = ClassEntry(pid=5)
    assert not hasattr(e, "tid")


def test_isinstance_discriminates_the_two_stages():
    name_stage: NamingState = NameEntry(
        tid=1, face_emb=None, body_emb=None, partial_emb=None, crop=None
    )
    class_stage: NamingState = ClassEntry(pid=1)
    assert isinstance(name_stage, NameEntry)
    assert not isinstance(name_stage, ClassEntry)
    assert isinstance(class_stage, ClassEntry)
    assert not isinstance(class_stage, NameEntry)
