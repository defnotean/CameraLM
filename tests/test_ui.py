"""Tests for cameralm/ui.py.

Focus areas:
  * ``Renderer`` transparent-overlay compositing in ``finish()`` (no-op fast
    path, opaque draw, semi-transparent blend, shape/dtype preservation).
  * ``_text_size`` memoization via ``_TEXT_SIZE_CACHE``.

``ui.py`` depends only on PIL + numpy. ``Renderer`` operates purely on numpy
arrays, so these tests need no display, camera, or GPU.
"""

import numpy as np

import cameralm.ui as ui
from cameralm.ui import Renderer


def _synthetic_frame(value=90):
    """A flat synthetic BGR frame (uint8), distinct from typical draw colors."""
    return np.full((120, 160, 3), value, dtype=np.uint8)


# --------------------------------------------------------------------------
# 1. No-op composite: nothing drawn -> frame unchanged.
# --------------------------------------------------------------------------
def test_finish_noop_returns_identical_frame():
    frame = _synthetic_frame(90)
    original = frame.copy()

    out = Renderer(frame).finish()

    # Nothing was drawn: the fast path returns the frame untouched.
    assert np.array_equal(out, original)
    # And the in-place buffer itself was not mutated.
    assert np.array_equal(frame, original)


# --------------------------------------------------------------------------
# 2. Opaque draw composites correctly: inside moves to draw color,
#    outside stays byte-identical.
# --------------------------------------------------------------------------
def test_finish_opaque_rectangle_composites():
    frame = _synthetic_frame(90)
    original = frame.copy()

    r = Renderer(frame)
    # Fully opaque rectangle. fill is RGBA; finish() converts RGB->BGR, so
    # RGBA (255, 0, 0) lands as BGR (0, 0, 255) in the composite. We only
    # assert "moved toward the drawn color", which is conversion-agnostic.
    rx1, ry1, rx2, ry2 = 40, 30, 100, 80
    r.draw.rectangle([rx1, ry1, rx2, ry2], fill=(255, 0, 0, 255))
    out = r.finish()

    inside = out[ry1 + 1:ry2 - 1, rx1 + 1:rx2 - 1]
    # Inside the rect every pixel equals the composited (fully opaque) color
    # and differs from the original flat gray.
    assert np.all(inside == inside[0, 0])
    assert not np.array_equal(inside, np.full_like(inside, 90))
    # Inside moved toward the drawn BGR color (0, 0, 255): blue channel up,
    # green/red channels down from the original 90.
    drawn_bgr = inside[0, 0]
    assert drawn_bgr[0] < 90 and drawn_bgr[1] < 90 and drawn_bgr[2] > 90

    # Outside the rect: byte-identical to the input frame.
    mask = np.ones(out.shape[:2], dtype=bool)
    mask[ry1:ry2 + 1, rx1:rx2 + 1] = False
    assert np.array_equal(out[mask], original[mask])


# --------------------------------------------------------------------------
# 3. Shape and dtype preserved by finish().
# --------------------------------------------------------------------------
def test_finish_preserves_shape_and_dtype():
    # No-op path.
    frame = _synthetic_frame(90)
    out = Renderer(frame).finish()
    assert out.shape == (120, 160, 3)
    assert out.dtype == np.uint8

    # Composited path.
    frame2 = _synthetic_frame(90)
    r = Renderer(frame2)
    r.draw.rectangle([10, 10, 50, 50], fill=(0, 200, 0, 255))
    out2 = r.finish()
    assert out2.shape == (120, 160, 3)
    assert out2.dtype == np.uint8


# --------------------------------------------------------------------------
# 4. Semi-transparent blend: result strictly between original and draw color.
# --------------------------------------------------------------------------
def test_finish_semi_transparent_blend():
    frame = _synthetic_frame(90)

    r = Renderer(frame)
    rx1, ry1, rx2, ry2 = 30, 20, 110, 90
    # RGBA white at ~50% alpha. finish() blends:
    #   base * (1 - a) + over * a  with a ~= 128/255.
    # base is 90 on every channel; over (white) is 255 on every channel; so
    # each blended channel must land strictly between 90 and 255.
    r.draw.rectangle([rx1, ry1, rx2, ry2], fill=(255, 255, 255, 128))
    out = r.finish()

    inside = out[ry1 + 1:ry2 - 1, rx1 + 1:rx2 - 1].astype(np.int32)
    assert np.all(inside > 90), "blended pixels must exceed the original value"
    assert np.all(inside < 255), "blended pixels must stay below the draw color"

    # Outside is still untouched.
    assert np.all(out[0, 0] == 90)


# --------------------------------------------------------------------------
# 5. _text_size memoization.
# --------------------------------------------------------------------------
def test_text_size_memoization():
    # Use the theme's configured font path rather than a hardcoded one; _font()
    # falls back to PIL's default if it's missing, so this works cross-platform.
    from cameralm import theme

    font = ui._font(theme.FONT_REGULAR, 17)
    text = "memoization probe - unique sample"

    first = ui._text_size(text, font)
    size_after_first = len(ui._TEXT_SIZE_CACHE)

    second = ui._text_size(text, font)
    size_after_second = len(ui._TEXT_SIZE_CACHE)

    # Identical args -> identical result.
    assert first == second
    # The second call was a pure cache hit: the cache did not grow.
    assert size_after_second == size_after_first
    # The measured text was actually cached on the first call.
    key = (text, getattr(font, "path", None), getattr(font, "size", None))
    assert key in ui._TEXT_SIZE_CACHE

    # The returned tuple is two positive ints.
    assert isinstance(first, tuple) and len(first) == 2
    w, h = first
    assert isinstance(w, int) and isinstance(h, int)
    assert w > 0 and h > 0
