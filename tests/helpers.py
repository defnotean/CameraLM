"""Plain importable test helpers - no pytest, no fixtures.

These produce deterministic, L2-normalized float32 embedding vectors so test
files across the suite can build identity-DB fixtures without touching real
models. Import as ``from helpers import face_vec, ...`` (pytest puts the
conftest dir on sys.path).
"""

import numpy as np

from cameralm.config import FACE_DIM, PARTIAL_DIM, REID_DIM


def unit_vec(dim, seed=0):
    """Return an L2-normalized float32 numpy array of length ``dim``.

    Deterministic for a given ``seed`` via ``np.random.default_rng``.
    """
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        # Astronomically unlikely for these dims, but keep it total.
        vec = np.ones(dim, dtype=np.float32)
        norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32)


def face_vec(seed=0):
    """A unit face embedding of length ``cameralm.config.FACE_DIM``."""
    return unit_vec(FACE_DIM, seed=seed)


def body_vec(seed=0):
    """A unit body ReID embedding of length ``cameralm.config.REID_DIM``."""
    return unit_vec(REID_DIM, seed=seed)


def partial_vec(seed=0):
    """A unit partial-appearance embedding of length ``cameralm.config.PARTIAL_DIM``."""
    return unit_vec(PARTIAL_DIM, seed=seed)


def near_vec(base, jitter=0.005, seed=0):
    """Return a unit vector very close to ``base``.

    Adds tiny gaussian noise scaled by ``jitter`` and renormalizes, so the
    result has high cosine similarity to ``base`` - useful for dedup tests.
    """
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(base.shape[0]).astype(np.float32) * jitter
    vec = base + noise
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return base.astype(np.float32)
    return (vec / norm).astype(np.float32)
