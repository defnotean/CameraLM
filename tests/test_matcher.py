"""Behavior tests for cameralm.matcher.decide_identity.

These tests assert the *contract* of decide_identity, not its internal
constants or line numbers. A separate change may be extracting the magic
numbers in matcher.py into named constants; that does not change behavior,
so these tests should keep passing.

decide_identity(face_emb, body_emb, partial_emb, db, expected_pid) returns
a 3-tuple (pid_or_None, source_str_or_None, similarity_float).

Identity policy under test (see cameralm/config.py):
  * REQUIRE_FACE_FOR_NEW_TRACK is True by default -> a brand-new track
    (expected_pid is None) can only be identified by a face match. Body /
    partial / side-agreement evidence is only usable once a track is
    already locked (expected_pid is set).
  * Face evidence wins when it clears FACE_MATCH_THRESH and FACE_MATCH_MARGIN.
  * For a locked track, body/partial evidence is only accepted if it points
    at the *expected* pid.
"""

import numpy as np
import pytest

from cameralm import config
from cameralm.matcher import decide_identity

from helpers import face_vec, body_vec, partial_vec


# --------------------------------------------------------------------------
# 1. Empty DB, no embeddings -> definitive no-match.
# --------------------------------------------------------------------------

def test_empty_db_no_embeddings_returns_no_match(db):
    """No evidence and nothing enrolled -> (None, None, 0.0)."""
    result = decide_identity(None, None, None, db, None)
    assert result == (None, None, 0.0)


# --------------------------------------------------------------------------
# 5. None embeddings against a populated DB -> no crash, no match.
# --------------------------------------------------------------------------

def test_none_embeddings_with_populated_db_no_crash(populated_db):
    """All embeddings None must be handled gracefully even when people exist.

    Every evidence branch in decide_identity is guarded by `if emb is not
    None`, so with no embeddings supplied nothing can match regardless of
    what is enrolled.
    """
    assert populated_db.count_people() > 0  # sanity: db really is populated

    # New-track form.
    assert decide_identity(None, None, None, populated_db, None) == (None, None, 0.0)

    # Locked-track form: still nothing to compare against, so still no match.
    some_pid = sorted(populated_db.people.keys())[0]
    assert decide_identity(None, None, None, populated_db, some_pid) == (None, None, 0.0)


def test_none_embeddings_with_enrolled_person_no_match(db):
    """Same guarantee using the empty-db fixture plus a manually enrolled person."""
    pid = db.create_person("Solo")
    db.add_face(pid, face_vec(seed=1))
    db.add_body(pid, body_vec(seed=1))
    db.add_partial(pid, partial_vec(seed=1))

    assert decide_identity(None, None, None, db, None) == (None, None, 0.0)
    # Even for a locked track, None embeddings cannot confirm anything.
    assert decide_identity(None, None, None, db, pid) == (None, None, 0.0)


# --------------------------------------------------------------------------
# 2. New track + REQUIRE_FACE_FOR_NEW_TRACK: body-only / partial-only is
#    insufficient to identify a brand-new track.
# --------------------------------------------------------------------------

def test_new_track_body_only_does_not_identify(db):
    """A strong body match alone cannot name a new track when REQUIRE_FACE_FOR_NEW_TRACK.

    We enroll a person, then ask decide_identity to identify a NEW track
    (expected_pid=None) given that person's exact body embedding. Because
    REQUIRE_FACE_FOR_NEW_TRACK is True, body evidence is not allowed to
    identify a new track, so the result must be a no-match -- and in
    particular must not return that person's pid.
    """
    if not config.REQUIRE_FACE_FOR_NEW_TRACK:
        pytest.skip("Test assumes REQUIRE_FACE_FOR_NEW_TRACK is True (the default)")

    body = body_vec(seed=3)
    pid = db.create_person("BodyOnly")
    db.add_body(pid, body)

    # Sanity: the DB itself *can* match this body strongly...
    matched_pid, matched_sim = db.search_body(body)
    assert matched_pid == pid
    assert matched_sim >= config.REID_MATCH_THRESH

    # ...but the matcher must refuse to identify a brand-new track from it.
    result_pid, source, sim = decide_identity(None, body, None, db, None)
    assert result_pid is None
    assert source is None
    assert sim == 0.0


def test_new_track_partial_only_does_not_identify(db):
    """A strong partial-appearance match alone cannot name a new track either."""
    if not config.REQUIRE_FACE_FOR_NEW_TRACK:
        pytest.skip("Test assumes REQUIRE_FACE_FOR_NEW_TRACK is True (the default)")

    partial = partial_vec(seed=4)
    pid = db.create_person("PartialOnly")
    db.add_partial(pid, partial)

    # Sanity: the partial index can match this strongly on its own.
    matched_pid, matched_sim, _margin = db.search_partial(partial)
    assert matched_pid == pid
    assert matched_sim >= config.PARTIAL_MATCH_THRESH

    # The matcher must still refuse to identify a brand-new track from it.
    assert decide_identity(None, None, partial, db, None) == (None, None, 0.0)


def test_new_track_body_and_partial_together_still_do_not_identify(db):
    """Body + partial agreeing is still not enough for a NEW track.

    The body/partial side-agreement fallback in decide_identity is gated on
    `not REQUIRE_FACE_FOR_NEW_TRACK`, so with the default config even two
    weak signals agreeing cannot identify a track that was never face-locked.
    """
    if not config.REQUIRE_FACE_FOR_NEW_TRACK:
        pytest.skip("Test assumes REQUIRE_FACE_FOR_NEW_TRACK is True (the default)")

    pid = db.create_person("BodyAndPartial")
    body = body_vec(seed=5)
    partial = partial_vec(seed=5)
    db.add_body(pid, body)
    db.add_partial(pid, partial)

    assert decide_identity(None, body, partial, db, None) == (None, None, 0.0)


# --------------------------------------------------------------------------
# 3. A strong face match identifies the person on a new track.
# --------------------------------------------------------------------------

def test_strong_face_match_identifies_person(db):
    """An exact face embedding for an enrolled person returns ("face", high sim).

    Face is the one signal allowed to identify a brand-new track. With the
    person's exact enrolled vector the cosine similarity is ~1.0, which
    clears FACE_MATCH_THRESH, and with only one person enrolled the
    uniqueness margin clears FACE_MATCH_MARGIN.
    """
    face = face_vec(seed=7)
    pid = db.create_person("FaceMatch")
    db.add_face(pid, face)

    result_pid, source, sim = decide_identity(face, None, None, db, None)
    assert result_pid == pid
    assert source == "face"
    assert sim >= config.FACE_MATCH_THRESH
    # Exact-vector match -> essentially perfect similarity.
    assert sim == pytest.approx(1.0, abs=1e-3)


def test_strong_face_match_wins_over_body(db):
    """When both face and body evidence are present, face evidence is returned."""
    face = face_vec(seed=8)
    body = body_vec(seed=8)
    pid = db.create_person("FaceAndBody")
    db.add_face(pid, face)
    db.add_body(pid, body)

    result_pid, source, sim = decide_identity(face, body, None, db, None)
    assert result_pid == pid
    assert source == "face"
    assert sim >= config.FACE_MATCH_THRESH


def test_unrelated_face_does_not_match(db):
    """A face embedding unrelated to the (single) enrolled person yields no match.

    A random high-dimensional unit vector is ~orthogonal to the enrolled
    face vector, so its similarity falls well below FACE_MATCH_THRESH.
    """
    enrolled = face_vec(seed=10)
    pid = db.create_person("Enrolled")
    db.add_face(pid, enrolled)

    stranger = face_vec(seed=999)
    result_pid, source, sim = decide_identity(stranger, None, None, db, None)
    assert result_pid is None
    assert source is None
    assert sim == 0.0


# --------------------------------------------------------------------------
# 4. expected_pid (locked track): body evidence is accepted only for the
#    expected person, not for some other enrolled person.
# --------------------------------------------------------------------------

def test_locked_track_body_matching_expected_pid_is_accepted(db):
    """For a locked track, a strong body match on the expected pid is accepted.

    locked_track is True, body_allowed becomes `body_pid == expected_pid`,
    so the expected person's own body embedding identifies the track via
    the "body" source.
    """
    pid_a = db.create_person("Locked-A")
    pid_b = db.create_person("Other-B")
    body_a = body_vec(seed=11)
    body_b = body_vec(seed=12)
    db.add_body(pid_a, body_a)
    db.add_body(pid_b, body_b)

    result_pid, source, sim = decide_identity(None, body_a, None, db, expected_pid=pid_a)
    assert result_pid == pid_a
    assert source == "body"
    assert sim >= config.REID_MATCH_THRESH


def test_locked_track_body_matching_different_pid_is_not_accepted_as_that_pid(db):
    """A body that matches a DIFFERENT enrolled person is not accepted as that person.

    The track is locked to pid_a, but we feed pid_b's body embedding. Since
    body_allowed requires `body_pid == expected_pid`, the matcher will not
    return pid_b. (It also will not silently relabel the track as pid_a,
    because pid_b's body vector does not match pid_a.) The documented,
    code-accurate outcome here is a clean no-match.
    """
    pid_a = db.create_person("Locked-A")
    pid_b = db.create_person("Other-B")
    body_a = body_vec(seed=13)
    body_b = body_vec(seed=14)
    db.add_body(pid_a, body_a)
    db.add_body(pid_b, body_b)

    # Sanity: pid_b's body really does match pid_b in the DB.
    matched_pid, matched_sim = db.search_body(body_b)
    assert matched_pid == pid_b
    assert matched_sim >= config.REID_MATCH_THRESH

    result_pid, source, sim = decide_identity(None, body_b, None, db, expected_pid=pid_a)
    # Must not be accepted as the other person...
    assert result_pid != pid_b
    # ...and with the default config this is a no-match.
    assert result_pid is None
    assert source is None
    assert sim == 0.0


def test_locked_track_partial_matching_expected_pid_is_accepted(db):
    """For a locked track, a strong partial match on the expected pid is accepted."""
    pid_a = db.create_person("Locked-A")
    pid_b = db.create_person("Other-B")
    partial_a = partial_vec(seed=15)
    partial_b = partial_vec(seed=16)
    db.add_partial(pid_a, partial_a)
    db.add_partial(pid_b, partial_b)

    result_pid, source, sim = decide_identity(None, None, partial_a, db, expected_pid=pid_a)
    assert result_pid == pid_a
    assert source == "partial"
    assert sim >= config.PARTIAL_MATCH_THRESH


def test_locked_track_partial_matching_different_pid_is_not_accepted_as_that_pid(db):
    """A partial that matches a different enrolled person is not accepted as that person."""
    pid_a = db.create_person("Locked-A")
    pid_b = db.create_person("Other-B")
    partial_a = partial_vec(seed=17)
    partial_b = partial_vec(seed=18)
    db.add_partial(pid_a, partial_a)
    db.add_partial(pid_b, partial_b)

    result_pid, source, sim = decide_identity(None, None, partial_b, db, expected_pid=pid_a)
    assert result_pid != pid_b
    assert result_pid is None
    assert source is None
    assert sim == 0.0


def test_locked_track_face_still_wins_for_expected_pid(db):
    """A locked track still gets a direct face identification for its own face."""
    pid_a = db.create_person("Locked-A")
    pid_b = db.create_person("Other-B")
    face_a = face_vec(seed=19)
    db.add_face(pid_a, face_a)
    db.add_face(pid_b, face_vec(seed=20))

    result_pid, source, sim = decide_identity(face_a, None, None, db, expected_pid=pid_a)
    assert result_pid == pid_a
    assert source == "face"
    assert sim >= config.FACE_MATCH_THRESH


# --------------------------------------------------------------------------
# Cross-check against the shared populated_db fixture.
# --------------------------------------------------------------------------

def test_populated_db_new_track_body_only_does_not_identify(populated_db):
    """Using the shared 3-person fixture: body-only evidence cannot name a new track.

    We pick an enrolled person, pull a body embedding that the DB matches
    to them, and confirm decide_identity still refuses to identify a
    brand-new track from body evidence alone.
    """
    if not config.REQUIRE_FACE_FOR_NEW_TRACK:
        pytest.skip("Test assumes REQUIRE_FACE_FOR_NEW_TRACK is True (the default)")

    # Find a person in the fixture that actually has a body embedding.
    target_pid = None
    target_body = None
    for pid in sorted(populated_db.people.keys()):
        idxs = [i for i, p in enumerate(populated_db.body_pids) if p == pid]
        if idxs:
            target_pid = pid
            target_body = np.array(populated_db.body_embeddings[idxs[0]], dtype=np.float32)
            break
    assert target_pid is not None, "populated_db fixture should enroll body embeddings"

    # The DB matches this embedding strongly to its owner...
    matched_pid, matched_sim = populated_db.search_body(target_body)
    assert matched_pid == target_pid

    # ...but a brand-new track cannot be identified from body alone.
    result_pid, source, sim = decide_identity(None, target_body, None, populated_db, None)
    assert result_pid is None
    assert source is None
    assert sim == 0.0


def test_populated_db_locked_track_body_identifies_expected_pid(populated_db):
    """Using the shared fixture: a locked track is confirmed by its own body embedding."""
    target_pid = None
    target_body = None
    for pid in sorted(populated_db.people.keys()):
        idxs = [i for i, p in enumerate(populated_db.body_pids) if p == pid]
        if idxs:
            target_pid = pid
            target_body = np.array(populated_db.body_embeddings[idxs[0]], dtype=np.float32)
            break
    assert target_pid is not None, "populated_db fixture should enroll body embeddings"

    matched_pid, _sim, margin = populated_db.search_body_detailed(target_body)
    assert matched_pid == target_pid

    result_pid, source, sim = decide_identity(
        None, target_body, None, populated_db, expected_pid=target_pid
    )
    if margin >= config.REID_MATCH_MARGIN:
        # Clean, unique body match for the expected person -> accepted as "body".
        assert result_pid == target_pid
        assert source == "body"
        assert sim >= config.REID_MATCH_THRESH
    else:
        # If the fixture's people are too close in body space to clear the
        # uniqueness margin, the matcher conservatively returns no match --
        # but it must never return the *wrong* person.
        assert result_pid in (None, target_pid)
