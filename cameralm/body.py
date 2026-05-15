import logging

import numpy as np
import torch
from boxmot.reid.core.reid import ReID

from .config import REID_WEIGHTS

log = logging.getLogger(__name__)


class BodyEmbedder:
    """OSNet body embeddings via boxmot. Weights auto-download on first use.

    boxmot's postprocess L2-normalizes the output, so the returned vector is ready
    for cosine similarity against the face/body FAISS index.
    """

    def __init__(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        # FP16 halves OSNet latency + VRAM on the GTX 1650; boxmot still
        # L2-normalizes the output so cosine similarity is unaffected.
        self.reid = ReID(weights=REID_WEIGHTS, device=device, half=use_cuda)
        if use_cuda:
            log.info("OSNet body ReID on GPU (cuda, FP16).")
        else:
            log.warning("OSNet body ReID on CPU - torch.cuda.is_available() is False.")

    def embed(self, frame, bbox):
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 - x1 < 10 or y2 - y1 < 20:
            return None
        boxes = np.asarray([[x1, y1, x2, y2]], dtype=np.float32)
        payload = self.reid.preprocess(frame, boxes=boxes)
        features = self.reid.process(payload)
        embeddings = self.reid.postprocess(features)
        if embeddings.size == 0:
            return None
        return embeddings[0].astype(np.float32)
