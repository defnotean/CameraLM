"""Typed state for the two-stage 'name an unknown person' UI flow.

This replaces an untyped, stage-keyed dict. That dict was the source of a
per-frame ``KeyError`` that froze the whole app: the "name" stage carried a
``tid`` key, the "class" stage did not, and one code path read ``naming["tid"]``
unconditionally. With two distinct dataclasses, reading a stage-specific field
on the wrong stage is an ``AttributeError`` at that line (and a static type
error) - not a runtime crash buried in a hot loop.

Callers discriminate the stage with ``isinstance(...)``, never a string key.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NameEntry:
    """Stage 1 - the operator is typing a name for a tracked, still-unknown person.

    Tied to a live ByteTrack id (``tid``); the embeddings/crop are snapshotted at
    click time so they stay valid even if the person moves while the name is typed.
    """

    tid: int
    face_emb: np.ndarray | None
    body_emb: np.ndarray | None
    partial_emb: np.ndarray | None
    crop: np.ndarray | None
    buffer: str = ""
    lost_since: float | None = None   # monotonic time the subject left frame, or None


@dataclass
class ClassEntry:
    """Stage 2 - the person now exists in the DB; the operator is assigning classes.

    Not tied to any live track: the subject may leave the frame freely here.
    """

    pid: int
    buffer: str = ""
    assigned: list[str] = field(default_factory=list)


# A naming session is in exactly one of the two stages (or absent).
NamingState = NameEntry | ClassEntry
