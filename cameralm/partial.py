import cv2
import numpy as np

from .config import PARTIAL_DIM, PARTIAL_MIN_PIXELS


class PartialAppearanceEmbedder:
    """Visible-region appearance signature for occluded person views.

    This is intentionally not treated as strong identity evidence. It captures
    stable visual cues from whatever is visible in the person box: clothing
    color, head/hair tones, arm/skin tones, accessories, and coarse texture.
    """

    def embed(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] * crop.shape[1] < PARTIAL_MIN_PIXELS:
            return None

        crop = self._normalize_size(crop)
        regions = self._regions(crop)
        features = []
        for region in regions:
            features.append(self._region_hist(region))
        emb = np.concatenate(features).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm < 1e-6:
            return None
        emb /= norm
        if emb.shape[0] != PARTIAL_DIM:
            return None
        return emb

    def _normalize_size(self, crop):
        h, w = crop.shape[:2]
        scale = 160.0 / max(h, w)
        if scale < 1.0:
            crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(crop, (3, 3), 0)

    def _regions(self, crop):
        h, w = crop.shape[:2]
        y_top = max(1, int(h * 0.38))
        y_mid_1 = max(1, int(h * 0.25))
        y_mid_2 = max(y_mid_1 + 1, int(h * 0.75))
        x_mid_1 = max(1, int(w * 0.18))
        x_mid_2 = max(x_mid_1 + 1, int(w * 0.82))
        return [
            crop,
            crop[:y_top, :],
            crop[y_mid_1:y_mid_2, x_mid_1:x_mid_2],
            crop[y_top:, :],
        ]

    def _region_hist(self, region):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [12], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        g_hist = cv2.calcHist([gray], [0], None, [4], [0, 256]).flatten()

        feat = np.concatenate([h_hist, s_hist, v_hist, g_hist]).astype(np.float32)
        total = float(feat.sum())
        if total > 0:
            feat /= total
        return np.sqrt(feat)
