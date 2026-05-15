"""Tests for cameralm.partial.PartialAppearanceEmbedder.

The partial embedder builds a normalized HSV + grayscale histogram signature
from a person crop. It uses only OpenCV + numpy (no GPU, no models), so it is
directly testable with synthetic numpy frames.

No pytest fixtures - plain functions, runnable under pytest. Synthetic frames
are built with numpy so the tests are fully deterministic.
"""

import numpy as np

from cameralm.config import PARTIAL_DIM, PARTIAL_MIN_PIXELS
from cameralm.partial import PartialAppearanceEmbedder


# --- helpers ---------------------------------------------------------------

def _solid_frame(h, w, bgr):
    """Return an (h, w, 3) uint8 BGR frame filled with a solid color."""
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _cosine(a, b):
    """Cosine similarity of two 1-D vectors (embeddings are already unit norm)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# --- 1. valid full-frame bbox ---------------------------------------------

def test_embed_valid_full_frame_bbox_shape_and_norm():
    """A valid full-frame bbox yields a float32 (PARTIAL_DIM,) unit vector."""
    embedder = PartialAppearanceEmbedder()
    # Use textured noise so every histogram region has real variation.
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(200, 120, 3), dtype=np.uint8)

    emb = embedder.embed(frame, [0, 0, 120, 200])

    assert emb is not None
    assert isinstance(emb, np.ndarray)
    assert emb.dtype == np.float32
    assert emb.shape == (PARTIAL_DIM,)
    # L2-normalized inside embed().
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5


def test_embed_valid_solid_frame_is_unit_norm():
    """Even a flat solid crop produces a finite unit-norm embedding."""
    embedder = PartialAppearanceEmbedder()
    frame = _solid_frame(200, 120, (40, 160, 90))

    emb = embedder.embed(frame, [0, 0, 120, 200])

    assert emb is not None
    assert emb.shape == (PARTIAL_DIM,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5


# --- 2. crop below PARTIAL_MIN_PIXELS -------------------------------------

def test_embed_returns_none_below_min_pixels():
    """A crop with fewer than PARTIAL_MIN_PIXELS pixels returns None."""
    embedder = PartialAppearanceEmbedder()
    frame = _solid_frame(200, 120, (200, 50, 50))

    # 10 x 10 = 100 px, far below PARTIAL_MIN_PIXELS (1200).
    side = 10
    assert side * side < PARTIAL_MIN_PIXELS
    emb = embedder.embed(frame, [5, 5, 5 + side, 5 + side])

    assert emb is None


def test_embed_returns_none_just_below_min_pixels_boundary():
    """A crop just under the PARTIAL_MIN_PIXELS threshold returns None."""
    embedder = PartialAppearanceEmbedder()
    frame = _solid_frame(200, 200, (10, 220, 120))

    # 34 x 34 = 1156 px < 1200; 35 x 35 = 1225 px >= 1200.
    small = 34
    assert small * small < PARTIAL_MIN_PIXELS
    emb = embedder.embed(frame, [0, 0, small, small])

    assert emb is None


# --- 3. zero-area / inverted bbox -----------------------------------------

def test_embed_returns_none_for_zero_area_bbox():
    """A zero-area bbox (x1 == x2, y1 == y2) returns None without raising."""
    embedder = PartialAppearanceEmbedder()
    frame = _solid_frame(200, 120, (123, 45, 200))

    emb = embedder.embed(frame, [50, 50, 50, 50])

    assert emb is None


def test_embed_returns_none_for_inverted_bbox():
    """An inverted bbox (x2 < x1, y2 < y1) returns None without raising."""
    embedder = PartialAppearanceEmbedder()
    frame = _solid_frame(200, 120, (60, 60, 60))

    emb = embedder.embed(frame, [80, 80, 10, 10])

    assert emb is None


def test_embed_returns_none_for_zero_width_bbox():
    """A degenerate bbox with zero width but nonzero height returns None."""
    embedder = PartialAppearanceEmbedder()
    frame = _solid_frame(200, 120, (15, 15, 240))

    emb = embedder.embed(frame, [30, 10, 30, 180])

    assert emb is None


# --- 4. discriminative power + repeatability ------------------------------

def test_embed_different_colors_are_well_separated():
    """A solid red crop and a solid blue crop give well-separated embeddings."""
    embedder = PartialAppearanceEmbedder()
    # BGR: red is (0, 0, 255), blue is (255, 0, 0).
    red_frame = _solid_frame(200, 120, (0, 0, 255))
    blue_frame = _solid_frame(200, 120, (255, 0, 0))

    red_emb = embedder.embed(red_frame, [0, 0, 120, 200])
    blue_emb = embedder.embed(blue_frame, [0, 0, 120, 200])

    assert red_emb is not None
    assert blue_emb is not None
    sim = _cosine(red_emb, blue_emb)
    # Distinct colors must not collapse to the same signature.
    assert sim < 0.9


def test_embed_same_crop_twice_is_near_identical():
    """The same crop embedded twice yields near-identical embeddings."""
    embedder = PartialAppearanceEmbedder()
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(200, 120, 3), dtype=np.uint8)

    emb_a = embedder.embed(frame, [0, 0, 120, 200])
    emb_b = embedder.embed(frame.copy(), [0, 0, 120, 200])

    assert emb_a is not None
    assert emb_b is not None
    sim = _cosine(emb_a, emb_b)
    assert sim > 0.999
    # Deterministic pipeline: should be bit-for-bit equal too.
    assert np.allclose(emb_a, emb_b, atol=1e-6)


def test_embed_same_color_high_similarity_across_frames():
    """Two independently built frames of the same color match closely."""
    embedder = PartialAppearanceEmbedder()
    frame_a = _solid_frame(200, 120, (30, 200, 75))
    frame_b = _solid_frame(180, 140, (30, 200, 75))

    emb_a = embedder.embed(frame_a, [0, 0, 120, 200])
    emb_b = embedder.embed(frame_b, [0, 0, 140, 180])

    assert emb_a is not None
    assert emb_b is not None
    assert _cosine(emb_a, emb_b) > 0.95


# --- 5. no NaN / inf -------------------------------------------------------

def test_embed_output_is_finite_textured():
    """A textured crop produces an embedding with no NaN or inf."""
    embedder = PartialAppearanceEmbedder()
    rng = np.random.default_rng(2024)
    frame = rng.integers(0, 256, size=(200, 120, 3), dtype=np.uint8)

    emb = embedder.embed(frame, [0, 0, 120, 200])

    assert emb is not None
    assert np.isfinite(emb).all()


def test_embed_output_is_finite_solid_black():
    """A solid black crop still yields a finite embedding (or None), never NaN."""
    embedder = PartialAppearanceEmbedder()
    frame = _solid_frame(200, 120, (0, 0, 0))

    emb = embedder.embed(frame, [0, 0, 120, 200])

    # Black is uniform but still has nonzero histogram mass, so a vector is
    # returned; whatever comes back must be finite.
    if emb is not None:
        assert np.isfinite(emb).all()
        assert emb.shape == (PARTIAL_DIM,)


# --- 6. bbox partially outside the frame ----------------------------------

def test_embed_bbox_partially_outside_is_clamped():
    """A bbox extending past the frame edges is clamped and stays valid."""
    embedder = PartialAppearanceEmbedder()
    rng = np.random.default_rng(99)
    frame = rng.integers(0, 256, size=(200, 120, 3), dtype=np.uint8)

    # Starts at negative coords and runs well past width/height.
    emb = embedder.embed(frame, [-40, -30, 300, 400])

    # After clamping this covers the whole 120x200 frame - large and valid.
    assert emb is not None
    assert emb.shape == (PARTIAL_DIM,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5
    assert np.isfinite(emb).all()


def test_embed_bbox_fully_outside_returns_none():
    """A bbox entirely off-frame clamps to empty and returns None, no raise."""
    embedder = PartialAppearanceEmbedder()
    frame = _solid_frame(200, 120, (77, 88, 99))

    # Entirely to the right of the frame (x1 >= width).
    emb = embedder.embed(frame, [300, 10, 360, 190])

    assert emb is None


def test_embed_bbox_partially_outside_small_overlap_no_raise():
    """A bbox mostly off-frame with a tiny in-frame sliver does not raise."""
    embedder = PartialAppearanceEmbedder()
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(200, 120, 3), dtype=np.uint8)

    # Overlap region after clamping is x:[110,120], y:[-..]->[0,200] -> 10x200.
    emb = embedder.embed(frame, [110, -50, 500, 500])

    # Either a valid embedding or None is acceptable; the contract is "no raise".
    if emb is not None:
        assert emb.shape == (PARTIAL_DIM,)
        assert np.isfinite(emb).all()


def test_embed_accepts_float_bbox_coordinates():
    """Float bbox coords are floored via int() and still produce a valid result."""
    embedder = PartialAppearanceEmbedder()
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, size=(200, 120, 3), dtype=np.uint8)

    emb = embedder.embed(frame, [0.0, 0.7, 119.9, 199.4])

    assert emb is not None
    assert emb.shape == (PARTIAL_DIM,)
    assert np.isfinite(emb).all()
