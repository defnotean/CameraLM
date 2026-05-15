"""Rendering for CameraLM. PIL handles antialiased text on top of an OpenCV BGR frame."""

import logging
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import theme as T
from .types import IdentitySource

log = logging.getLogger(__name__)

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
_FONT_WARNED = False


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, size)
        except OSError:
            global _FONT_WARNED
            if not _FONT_WARNED:
                log.warning(
                    "Font %s unavailable - using PIL default; overlay layout may look off.", path
                )
                _FONT_WARNED = True
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


_TEXT_SIZE_CACHE: dict[tuple, tuple[int, int]] = {}
_MEASURE_DRAW = ImageDraw.Draw(Image.new("RGBA", (1, 1)))


def _text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Memoized text measurement - labels/classes/descriptions repeat every frame."""
    key = (text, getattr(font, "path", None), getattr(font, "size", None))
    cached = _TEXT_SIZE_CACHE.get(key)
    if cached is None:
        bbox = _MEASURE_DRAW.textbbox((0, 0), text, font=font)
        cached = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        if len(_TEXT_SIZE_CACHE) < 8192:
            _TEXT_SIZE_CACHE[key] = cached
    return cached


def _clamp(value: int, low: int, high: int) -> int:
    if high < low:
        return low
    return max(low, min(value, high))


class Renderer:
    """Wrap a BGR frame in a PIL drawing surface. Call `finish()` to get BGR back."""

    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        # Draw onto a transparent RGBA overlay - the camera frame is never
        # round-tripped through PIL. Compositing happens once, in numpy, in
        # finish(). This removes ~5 full-frame buffer copies + 2 colorspace
        # conversions per frame (the main person-present framerate cliff).
        self.pil = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.pil, "RGBA")

    def finish(self) -> np.ndarray:
        """Composite the overlay onto the frame and return the result.

        Non-mutating: the input frame is never written in place. The pipeline
        worker publishes one frame that the renderer and the click handler both
        read, so compositing must produce a new array, not edit the shared one.
        """
        overlay = np.asarray(self.pil)                        # (H, W, 4) RGBA uint8
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        if not alpha.any():
            return self.frame                                 # nothing drawn - fast path
        over_bgr = overlay[:, :, 2::-1].astype(np.float32)     # RGB -> BGR
        base = self.frame.astype(np.float32)
        return (base * (1.0 - alpha) + over_bgr * alpha).astype(np.uint8)

    def rounded_rect(self, xy, radius, fill=None, outline=None, width=1):
        x1, y1, x2, y2 = (int(v) for v in xy)
        if x2 <= x1 + 1:
            x2 = x1 + 2
        if y2 <= y1 + 1:
            y2 = y1 + 2
        self.draw.rounded_rectangle(
            [x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width,
        )

    def text(self, xy, text, font, fill):
        self.draw.text((int(xy[0]), int(xy[1])), text, font=font, fill=fill)

    def text_size(self, text, font) -> tuple[int, int]:
        return _text_size(text, font)

    def dim_full(self, alpha: int = 118):
        self.draw.rectangle([0, 0, self.w, self.h], fill=(0, 0, 0, alpha))


def _ellipsize(r: Renderer, text: str, font, max_w: int) -> str:
    if r.text_size(text, font)[0] <= max_w:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if r.text_size(text[:mid] + suffix, font)[0] <= max_w:
            lo = mid + 1
        else:
            hi = mid
    return text[:max(0, lo - 1)].rstrip() + suffix


def _wrap(r: Renderer, text: str, font, max_w: int, max_lines: int = 2) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for word in words:
        candidate = (cur + " " + word).strip() if cur else word
        cw, _ = r.text_size(candidate, font)
        if cw > max_w and cur:
            lines.append(cur)
            cur = word
            if len(lines) >= max_lines:
                cur = ""
                break
        else:
            cur = candidate
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines


def _draw_keycap(r: Renderer, x: int, y: int, text: str, font) -> int:
    tw, th = r.text_size(text, font)
    w = tw + 12
    r.rounded_rect(
        [x, y, x + w, y + th + 8],
        radius=6,
        fill=(*T.CARD_BG_SOFT, 235),
        outline=(*T.BORDER, 230),
        width=1,
    )
    r.text((x + 6, y + 3), text, font, fill=(*T.TEXT_PRIMARY, 255))
    return w


def draw_track(
    r: Renderer,
    bbox,
    label: str,
    source: str | None = None,
    conf: float = 0.0,
    description: str | None = None,
    hovered: bool = False,
    is_unknown: bool = False,
    classes: list[str] | None = None,
):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(r.w - 1, x2)
    y2 = min(r.h - 1, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return

    color = T.source_color(source)
    line_rgba = (*color, 255)

    if hovered:
        r.rounded_rect([x1 - 3, y1 - 3, x2 + 3, y2 + 3], radius=13, outline=(*color, 90), width=5)
    r.rounded_rect([x1, y1, x2, y2], radius=11, outline=line_rgba, width=3 if hovered else 2)

    corner = min(34, max(14, (x2 - x1) // 7))
    for ax, ay, bx, by in (
        (x1, y1, x1 + corner, y1),
        (x1, y1, x1, y1 + corner),
        (x2, y1, x2 - corner, y1),
        (x2, y1, x2, y1 + corner),
        (x1, y2, x1 + corner, y2),
        (x1, y2, x1, y2 - corner),
        (x2, y2, x2 - corner, y2),
        (x2, y2, x2, y2 - corner),
    ):
        r.draw.line([ax, ay, bx, by], fill=line_rgba, width=4)

    name_font = _font(T.FONT_SEMIBOLD, 17)
    sub_font = _font(T.FONT_REGULAR, 11)
    chip_font = _font(T.FONT_SEMIBOLD, 10)

    if source == IdentitySource.FACE:
        status = f"FACE {conf:.2f}"
    elif source == IdentitySource.BODY:
        status = f"BODY {conf:.2f}"
    elif source == IdentitySource.SIDE:
        status = f"SIDE {conf:.2f}"
    elif source == IdentitySource.PARTIAL:
        status = f"PARTIAL {conf:.2f}"
    elif source == IdentitySource.TRACK:
        status = f"TRACK {conf:.2f}"
    elif is_unknown:
        status = "NEW PERSON"
    else:
        status = "TRACKED"

    max_pill_w = min(360, max(190, r.w - 24))
    label_text = _ellipsize(r, label, name_font, max_pill_w - 38)
    label_w, label_h = r.text_size(label_text, name_font)
    status_w, status_h = r.text_size(status, sub_font)

    chip_items: list[tuple[str, int, int]] = []
    chip_h = 0
    if classes:
        visible = list(classes[:3])
        overflow = len(classes) - len(visible)
        if overflow > 0:
            visible.append(f"+{overflow}")
        for class_name in visible:
            text = _ellipsize(r, class_name, chip_font, 92)
            cw, ch = r.text_size(text, chip_font)
            chip_items.append((text, cw + 16, ch + 7))
        chip_h = max((h for _, _, h in chip_items), default=0)

    chips_w = sum(w for _, w, _ in chip_items) + max(0, len(chip_items) - 1) * 6
    inner_w = min(max_pill_w - 24, max(label_w + 18, status_w + 18, chips_w))
    pill_w = inner_w + 24
    pill_h = 14 + label_h + 4 + status_h + (8 + chip_h if chip_items else 0) + 12

    px = _clamp(x1, 8, r.w - pill_w - 8)
    py = y1 - pill_h - 9
    if py < 8:
        py = y1 + 8

    r.rounded_rect(
        [px, py, px + pill_w, py + pill_h],
        radius=8,
        fill=(*T.CARD_BG, 232),
        outline=(*color, 210),
        width=1,
    )
    r.draw.ellipse([px + 12, py + 16, px + 20, py + 24], fill=(*color, 255))
    r.text((px + 28, py + 10), label_text, name_font, fill=(*T.TEXT_PRIMARY, 255))
    r.text((px + 28, py + 10 + label_h + 3), status, sub_font, fill=(*T.TEXT_SECONDARY, 245))

    if chip_items:
        cx = px + 12
        cy = py + 10 + label_h + 3 + status_h + 8
        for text, cw, ch in chip_items:
            if cx + cw > px + pill_w - 12:
                break
            r.rounded_rect(
                [cx, cy, cx + cw, cy + ch],
                radius=ch // 2,
                fill=(*T.ACCENT, 38),
                outline=(*T.ACCENT, 190),
                width=1,
            )
            r.text((cx + 8, cy + 3), text, chip_font, fill=(*T.TEXT_PRIMARY, 240))
            cx += cw + 6

    if description:
        desc_font = _font(T.FONT_REGULAR, 13)
        chip_max_w = min(max(210, x2 - x1), r.w - 24)
        lines = _wrap(r, description, desc_font, chip_max_w - 24, max_lines=2)
        if lines:
            line_heights = [r.text_size(ln, desc_font)[1] for ln in lines]
            max_lw = max(r.text_size(ln, desc_font)[0] for ln in lines)
            total_h = sum(line_heights) + 4 * (len(lines) - 1)
            chip_w = max_lw + 24
            chip_h = total_h + 16
            cx = _clamp(x1, 8, r.w - chip_w - 8)
            cy = y2 + 8
            bottom_reserved = 64
            if cy + chip_h > r.h - bottom_reserved:
                cy = max(8, y2 - chip_h - 8)
            if cy + chip_h > r.h - bottom_reserved:
                cy = max(8, r.h - bottom_reserved - chip_h)
            r.rounded_rect(
                [cx, cy, cx + chip_w, cy + chip_h],
                radius=8,
                fill=(*T.CARD_BG, 210),
                outline=(*T.BORDER, 210),
                width=1,
            )
            ty = cy + 8
            for line, line_h in zip(lines, line_heights):
                r.text((cx + 12, ty), line, desc_font, fill=(*T.TEXT_PRIMARY, 245))
                ty += line_h + 4


def draw_status_hud(r: Renderer, fps: float, n_people: int, vlm_active: bool):
    title_font = _font(T.FONT_SEMIBOLD, 13)
    label_font = _font(T.FONT_REGULAR, 10)
    value_font = _font(T.FONT_MONO, 13)

    rows = [
        ("FPS", f"{fps:4.1f}", T.KNOWN if fps >= 12 else T.TENTATIVE),
        ("IDs", str(n_people), T.TEXT_PRIMARY),
        ("VLM", "ON" if vlm_active else "OFF", T.KNOWN if vlm_active else T.TEXT_DIM),
    ]

    box_w, box_h = 178, 102
    x = r.w - box_w - 14
    y = 14
    r.rounded_rect([x, y, x + box_w, y + box_h], radius=8, fill=(*T.CARD_BG, 210), outline=(*T.BORDER, 220), width=1)
    r.text((x + 13, y + 10), "CameraLM", title_font, fill=(*T.TEXT_PRIMARY, 255))
    r.draw.line([x + 12, y + 34, x + box_w - 12, y + 34], fill=(*T.BORDER, 180), width=1)

    cx = x + 13
    cy = y + 44
    col_w = (box_w - 26) // 3
    for label, value, color in rows:
        r.text((cx, cy), label, label_font, fill=(*T.TEXT_DIM, 255))
        r.text((cx, cy + 16), value, value_font, fill=(*color, 255))
        cx += col_w


def draw_help(r: Renderer):
    font = _font(T.FONT_REGULAR, 11)
    key_font = _font(T.FONT_SEMIBOLD, 10)
    items = [("Click", "Name"), ("Enter", "Save"), ("Esc", "Cancel"), ("S", "DB"), ("Q", "Quit")]

    widths = []
    for key, label in items:
        key_w = r.text_size(key, key_font)[0] + 12
        label_w = r.text_size(label, font)[0]
        widths.append(key_w + 6 + label_w)
    total_w = sum(widths) + (len(items) - 1) * 14 + 22
    box_h = 40
    x = 14
    y = r.h - box_h - 14

    if total_w > r.w - 28:
        items = [("Click", "Name"), ("Q", "Quit")]
        widths = []
        for key, label in items:
            widths.append(r.text_size(key, key_font)[0] + 18 + r.text_size(label, font)[0])
        total_w = sum(widths) + (len(items) - 1) * 14 + 22

    r.rounded_rect([x, y, x + total_w, y + box_h], radius=8, fill=(*T.CARD_BG, 190), outline=(*T.BORDER, 210), width=1)
    cx = x + 11
    for key, label in items:
        key_w = _draw_keycap(r, cx, y + 8, key, key_font)
        cx += key_w + 6
        r.text((cx, y + 12), label, font, fill=(*T.TEXT_SECONDARY, 255))
        cx += r.text_size(label, font)[0] + 14


def draw_naming_modal(r: Renderer, buffer: str):
    r.dim_full(128)

    title_font = _font(T.FONT_SEMIBOLD, 15)
    input_font = _font(T.FONT_SEMIBOLD, 30)
    hint_font = _font(T.FONT_REGULAR, 12)

    box_w, box_h = min(580, r.w - 48), 184
    x = (r.w - box_w) // 2
    y = (r.h - box_h) // 2

    r.rounded_rect([x, y, x + box_w, y + box_h], radius=10, fill=(*T.CARD_BG, 248), outline=(*T.ACCENT, 255), width=2)
    r.text((x + 24, y + 20), "Name this person", title_font, fill=(*T.TEXT_PRIMARY, 255))

    field_x1 = x + 24
    field_x2 = x + box_w - 24
    field_y = y + 60
    r.rounded_rect([field_x1, field_y, field_x2, field_y + 58], radius=8, fill=(*T.CARD_BG_LIGHT, 255), outline=(*T.BORDER, 255), width=1)
    cursor = "|" if (int(time.time() * 2) % 2 == 0) else " "
    input_text = _ellipsize(r, buffer + cursor, input_font, field_x2 - field_x1 - 30)
    r.text((field_x1 + 15, field_y + 9), input_text, input_font, fill=(*T.TEXT_PRIMARY, 255))
    r.text((x + 24, y + box_h - 28), "Enter save   Esc cancel   Backspace delete", hint_font, fill=(*T.TEXT_DIM, 255))


def draw_class_modal(
    r: Renderer,
    person_name: str,
    buffer: str,
    assigned: list[str],
    suggestions: list[str],
):
    r.dim_full(128)

    title_font = _font(T.FONT_SEMIBOLD, 15)
    name_font = _font(T.FONT_SEMIBOLD, 22)
    input_font = _font(T.FONT_SEMIBOLD, 22)
    chip_font = _font(T.FONT_SEMIBOLD, 11)
    small_font = _font(T.FONT_REGULAR, 11)

    box_w, box_h = min(660, r.w - 48), 292
    x = (r.w - box_w) // 2
    y = (r.h - box_h) // 2

    r.rounded_rect([x, y, x + box_w, y + box_h], radius=10, fill=(*T.CARD_BG, 248), outline=(*T.ACCENT, 255), width=2)
    r.text((x + 24, y + 20), "Assign classes", title_font, fill=(*T.TEXT_PRIMARY, 255))
    r.text((x + 24, y + 44), _ellipsize(r, person_name, name_font, box_w - 48), name_font, fill=(*T.TEXT_SECONDARY, 255))

    chip_y = y + 88
    chip_x = x + 24
    chip_h = 24
    if assigned:
        for name in assigned:
            text = _ellipsize(r, name, chip_font, 130)
            tw, _ = r.text_size(text, chip_font)
            w = tw + 20
            if chip_x + w > x + box_w - 24:
                chip_x = x + 24
                chip_y += chip_h + 7
            r.rounded_rect([chip_x, chip_y, chip_x + w, chip_y + chip_h], radius=12, fill=(*T.ACCENT, 48), outline=(*T.ACCENT, 210), width=1)
            r.text((chip_x + 10, chip_y + 4), text, chip_font, fill=(*T.TEXT_PRIMARY, 255))
            chip_x += w + 7
    else:
        r.text((x + 24, chip_y + 5), "No classes assigned yet.", small_font, fill=(*T.TEXT_DIM, 255))

    field_y = y + 154
    r.rounded_rect([x + 24, field_y, x + box_w - 24, field_y + 46], radius=8, fill=(*T.CARD_BG_LIGHT, 255), outline=(*T.BORDER, 255), width=1)
    cursor = "|" if (int(time.time() * 2) % 2 == 0) else " "
    if buffer:
        text = _ellipsize(r, buffer + cursor, input_font, box_w - 72)
        r.text((x + 39, field_y + 11), text, input_font, fill=(*T.TEXT_PRIMARY, 255))
    else:
        r.text((x + 39, field_y + 11), "Class name", input_font, fill=(*T.TEXT_DIM, 255))

    available = [name for name in suggestions if name not in assigned][:5]
    if available:
        sy = y + 218
        r.text((x + 24, sy), "Existing", small_font, fill=(*T.TEXT_DIM, 255))
        sx = x + 78
        for name in available:
            text = _ellipsize(r, name, chip_font, 100)
            tw, _ = r.text_size(text, chip_font)
            w = tw + 18
            if sx + w > x + box_w - 24:
                break
            r.rounded_rect([sx, sy - 4, sx + w, sy + 20], radius=12, fill=(*T.CARD_BG_SOFT, 245), outline=(*T.BORDER, 230), width=1)
            r.text((sx + 9, sy), text, chip_font, fill=(*T.TEXT_SECONDARY, 255))
            sx += w + 7

    r.text((x + 24, y + box_h - 28), "Enter add class   Enter empty or Esc done", small_font, fill=(*T.TEXT_DIM, 255))


def draw_recording_indicator(r: Renderer, n_in_frame: int):
    """Always-on notice that the camera is actively recognizing people.

    A privacy guardrail: anyone in front of the camera should be able to see
    that the system is running and recognizing faces.
    """
    font = _font(T.FONT_SEMIBOLD, 12)
    blink_on = int(time.time() * 1.5) % 2 == 0
    dot = T.UNKNOWN if blink_on else tuple(c // 2 for c in T.UNKNOWN)

    text = "RECOGNIZING" if n_in_frame == 0 else f"RECOGNIZING - {n_in_frame} in frame"
    tw, th = r.text_size(text, font)
    box_w = tw + 40
    box_h = 28
    x = (r.w - box_w) // 2
    y = 14

    r.rounded_rect(
        [x, y, x + box_w, y + box_h],
        radius=14,
        fill=(*T.CARD_BG, 224),
        outline=(*T.UNKNOWN, 210),
        width=1,
    )
    r.draw.ellipse([x + 14, y + 10, x + 22, y + 18], fill=(*dot, 255))
    r.text((x + 30, y + 7), text, font, fill=(*T.TEXT_PRIMARY, 255))
