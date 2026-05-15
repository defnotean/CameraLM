"""Visual theme: palette and font paths. Pure data, no rendering."""

from .types import IdentitySource


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# --- Palette (camera overlay) ---
KNOWN = _hex_to_rgb("#22D39A")          # confident match (face)
TENTATIVE = _hex_to_rgb("#F2C94C")      # body-only match
UNKNOWN = _hex_to_rgb("#FF6F6F")        # not in DB
ACCENT = _hex_to_rgb("#6AA7FF")         # modal focus
CARD_BG = _hex_to_rgb("#0D0F12")
CARD_BG_LIGHT = _hex_to_rgb("#1B1F26")
CARD_BG_SOFT = _hex_to_rgb("#222832")
TEXT_PRIMARY = _hex_to_rgb("#F6F8FB")
TEXT_SECONDARY = _hex_to_rgb("#AAB2BF")
TEXT_DIM = _hex_to_rgb("#727D8C")
BORDER = _hex_to_rgb("#303741")

# --- Fonts: Segoe UI + Consolas (Windows defaults) ---
FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"
FONT_SEMIBOLD = "C:/Windows/Fonts/seguisb.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"


def source_color(source: IdentitySource | None) -> tuple[int, int, int]:
    if source == IdentitySource.FACE:
        return KNOWN
    if source in (IdentitySource.BODY, IdentitySource.SIDE):
        return TENTATIVE
    if source in (IdentitySource.PARTIAL, IdentitySource.TRACK):
        return ACCENT
    return UNKNOWN
