"""Shared enums - the canonical vocabulary for cross-module values.

Replacing bare string literals ("face"/"body"/...) with an enum gives one
definition site and lets a typo be caught instead of silently falling through a
comparison. `StrEnum` members compare equal to their string value
(`IdentitySource.FACE == "face"`), so the enum is a drop-in for the old literals
and the migration is safe even half-applied.
"""

from enum import StrEnum


class IdentitySource(StrEnum):
    """How a track's identity was determined on the current frame."""

    FACE = "face"        # strong - a confident face (ArcFace) match
    BODY = "body"        # strong - a confident body (OSNet) match
    SIDE = "side"        # weak fusion - independent weak signals agree on one person
    PARTIAL = "partial"  # weak - a partial-appearance match (multi-hit confirmed)
    TRACK = "track"      # held - track-memory keeps a recently-confirmed identity alive
