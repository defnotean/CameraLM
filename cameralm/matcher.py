from .config import (
    FACE_MATCH_MARGIN,
    FACE_MATCH_THRESH,
    FACE_WEAK_MATCH_THRESH,
    PARTIAL_MATCH_MARGIN,
    PARTIAL_MATCH_THRESH,
    PARTIAL_WEAK_MATCH_MARGIN,
    PARTIAL_WEAK_MATCH_THRESH,
    REID_MATCH_MARGIN,
    REID_MATCH_THRESH,
    REID_NEWTRACK_MATCH_MARGIN,
    REID_WEAK_MATCH_THRESH,
    REQUIRE_FACE_FOR_NEW_TRACK,
    USE_BODY_REID,
    USE_PARTIAL_REID,
)
from .types import IdentitySource


def decide_identity(face_emb, body_emb, partial_emb, db, expected_pid=None):
    """Return (pid_or_None, source, similarity).

    Face evidence wins when available. Body ReID is the next strongest signal.
    Partial appearance is deliberately last and requires both a high score and
    a clear margin over the next person, because clothing/arms are not identity
    proof by themselves.
    """
    locked_track = expected_pid is not None

    face_pid = None
    face_sim = 0.0
    face_margin = 0.0
    if face_emb is not None:
        face_pid, face_sim, face_margin = db.search_face_detailed(face_emb)
        if face_pid is not None and face_sim >= FACE_MATCH_THRESH and face_margin >= FACE_MATCH_MARGIN:
            return face_pid, IdentitySource.FACE, face_sim

    body_pid = None
    body_sim = 0.0
    body_margin = 0.0
    if USE_BODY_REID and body_emb is not None:
        body_pid, body_sim, body_margin = db.search_body_detailed(body_emb)
        body_allowed = body_pid == expected_pid if locked_track else not REQUIRE_FACE_FOR_NEW_TRACK
        if (
            body_pid is not None
            and body_sim >= REID_MATCH_THRESH
            and body_margin >= REID_MATCH_MARGIN
            and body_allowed
        ):
            return body_pid, IdentitySource.BODY, body_sim

    partial_pid = None
    partial_sim = 0.0
    partial_margin = 0.0
    if USE_PARTIAL_REID and partial_emb is not None:
        partial_pid, partial_sim, partial_margin = db.search_partial(partial_emb)
        partial_allowed = partial_pid == expected_pid if locked_track else not REQUIRE_FACE_FOR_NEW_TRACK
        if (
            partial_pid is not None
            and partial_sim >= PARTIAL_MATCH_THRESH
            and partial_margin >= PARTIAL_MATCH_MARGIN
            and partial_allowed
        ):
            return partial_pid, IdentitySource.PARTIAL, partial_sim

    # Side-profile views often do not produce a strong frontal ArcFace match.
    # Accept them only when independent weak signals agree on the same person.
    # If a track was already confirmed, allow agreement specifically with that
    # expected person; otherwise use stricter new-track evidence to avoid false
    # positives from similar clothing/body shape.
    if expected_pid is not None:
        expected_hits = 0
        expected_score = 0.0
        if face_pid == expected_pid and face_sim >= FACE_WEAK_MATCH_THRESH:
            expected_hits += 1
            expected_score = max(expected_score, face_sim)
        if body_pid == expected_pid and body_sim >= REID_WEAK_MATCH_THRESH and body_margin >= REID_MATCH_MARGIN:
            expected_hits += 1
            expected_score = max(expected_score, body_sim)
        if (
            partial_pid == expected_pid
            and partial_sim >= PARTIAL_WEAK_MATCH_THRESH
            and partial_margin >= PARTIAL_WEAK_MATCH_MARGIN
        ):
            expected_hits += 1
            expected_score = max(expected_score, partial_sim)
        if expected_hits >= 2:
            return expected_pid, IdentitySource.SIDE, expected_score

    if (
        not locked_track
        and not REQUIRE_FACE_FOR_NEW_TRACK
        and
        face_pid is not None
        and body_pid == face_pid
        and face_sim >= FACE_WEAK_MATCH_THRESH
        and body_sim >= REID_WEAK_MATCH_THRESH
        and body_margin >= max(REID_MATCH_MARGIN, REID_NEWTRACK_MATCH_MARGIN)
    ):
        return face_pid, IdentitySource.SIDE, max(face_sim, body_sim)

    if (
        not locked_track
        and not REQUIRE_FACE_FOR_NEW_TRACK
        and
        body_pid is not None
        and partial_pid == body_pid
        and body_sim >= REID_WEAK_MATCH_THRESH
        and partial_sim >= PARTIAL_WEAK_MATCH_THRESH
        and body_margin >= max(REID_MATCH_MARGIN, REID_NEWTRACK_MATCH_MARGIN)
        and partial_margin >= PARTIAL_WEAK_MATCH_MARGIN
    ):
        return body_pid, IdentitySource.SIDE, max(body_sim, partial_sim)

    if (
        not locked_track
        and not REQUIRE_FACE_FOR_NEW_TRACK
        and
        face_pid is not None
        and partial_pid == face_pid
        and face_sim >= FACE_WEAK_MATCH_THRESH
        and partial_sim >= PARTIAL_WEAK_MATCH_THRESH
        and partial_margin >= PARTIAL_WEAK_MATCH_MARGIN
    ):
        return face_pid, IdentitySource.SIDE, max(face_sim, partial_sim)

    return None, None, 0.0
