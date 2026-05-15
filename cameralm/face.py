import logging

import numpy as np
from insightface.app import FaceAnalysis

from .config import (
    FACE_DET_SIZE,
    FACE_MODEL_PACK,
    FACE_PROFILE_FLIP_FALLBACK,
    FACE_UPPER_CROP_RATIO,
)

log = logging.getLogger(__name__)


class FaceEmbedder:
    """ArcFace embeddings via InsightFace. Returns None when no face is found inside the bbox."""

    def __init__(self):
        self.app = FaceAnalysis(
            name=FACE_MODEL_PACK,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
        self._log_device()

    def _log_device(self) -> None:
        """Report the provider each InsightFace model *actually* loaded with.

        onnxruntime silently falls back to CPUExecutionProvider when CUDA can't
        initialize, so requesting CUDA is not the same as running on it - this
        reads back the truth and warns loudly on a fallback.
        """
        providers: set[str] = set()
        for model in getattr(self.app, "models", {}).values():
            session = getattr(model, "session", None)
            if session is not None:
                providers.update(session.get_providers())
        if "CUDAExecutionProvider" in providers:
            log.info("InsightFace (ArcFace) on GPU - active providers: %s", sorted(providers))
        else:
            log.warning(
                "InsightFace fell back to CPU (active providers: %s) - face recognition "
                "will be slow. Check the onnxruntime-gpu / CUDA / cuDNN install.",
                sorted(providers) or ["<unknown>"],
            )

    def embed(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 20 or y2 - y1 < 40:
            return None

        crops = self._candidate_crops(frame, x1, y1, x2, y2)
        face = self._best_face(crops)
        if face is None and FACE_PROFILE_FLIP_FALLBACK:
            flipped = [np.ascontiguousarray(crop[:, ::-1]) for crop in crops]
            face = self._best_face(flipped)
        if face is None:
            return None
        emb = face.normed_embedding.astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm < 1e-6:
            return None
        return emb / norm

    def _candidate_crops(self, frame, x1, y1, x2, y2):
        full = frame[y1:y2, x1:x2]
        crops = [full]

        box_h = y2 - y1
        upper_y2 = min(y2, y1 + max(40, int(box_h * FACE_UPPER_CROP_RATIO)))
        if upper_y2 > y1:
            crops.append(frame[y1:upper_y2, x1:x2])

        return [crop for crop in crops if crop.size > 0]

    def _best_face(self, crops):
        best = None
        best_score = -1.0
        for crop in crops:
            faces = self.app.get(crop)
            for face in faces:
                x1, y1, x2, y2 = face.bbox
                area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
                det_score = float(getattr(face, "det_score", 1.0))
                score = area * max(det_score, 0.01)
                if score > best_score:
                    best = face
                    best_score = score
        return best
