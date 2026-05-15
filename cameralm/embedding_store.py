"""One embedding channel: a FAISS inner-product index + its parallel pid list +
shadow array, for a single signal (face / body / partial).

`IdentityDB` used to carry three near-identical copies of this state plus a set
of `kind`-dispatch ladders (`_embedding_store_locked`, `_index_for_kind_locked`,
`_rebuild_index_locked`, ...). Collapsing that into one composable class means a
fourth signal (gait / skeleton) is one `EmbeddingStore(...)` line, not another
copy of every ladder rung.

Not thread-safe on its own - `IdentityDB` owns the lock and only calls in while
holding it.
"""

import logging

import faiss
import numpy as np

log = logging.getLogger(__name__)


def _normalize(emb) -> np.ndarray | None:
    """Reshape to (1, dim) float32 and L2-normalize. None if missing/degenerate."""
    if emb is None:
        return None
    emb = np.asarray(emb).reshape(1, -1).astype(np.float32)
    norm = float(np.linalg.norm(emb))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    return emb / norm


class EmbeddingStore:
    """A FAISS IndexFlatIP + parallel `pids` list + `embeddings` shadow array.

    The shadow array is the source of truth; the index is rebuilt from it after
    any in-place edit. `pids[i]` owns `embeddings[i]`.
    """

    def __init__(self, dim: int, dup_sim: float, cap: int):
        self.dim = dim
        self.dup_sim = dup_sim          # cosine sim at/above which a vector is a near-duplicate
        self.cap = cap                  # max stored vectors per person
        self.index = faiss.IndexFlatIP(dim)
        self.embeddings = np.empty((0, dim), dtype=np.float32)
        self.pids: list[int] = []

    def reset(self) -> None:
        """Drop everything - empty index, array, and pid list."""
        self.embeddings = np.empty((0, self.dim), dtype=np.float32)
        self.pids = []
        self.rebuild_index()

    def rebuild_index(self) -> None:
        """Rebuild the FAISS index from the current shadow array."""
        self.index = faiss.IndexFlatIP(self.dim)
        if len(self.embeddings):
            self.index.add(self.embeddings)

    def count_for(self, pid: int) -> int:
        return self.pids.count(pid)

    def add(self, pid: int, emb) -> int | None:
        """Append `emb` for `pid`, or replace the most-redundant slot once the
        person is at `cap`. Returns the person's new vector count, or None if
        `emb` was missing/degenerate or a near-duplicate of an existing vector.

        Keeping side/back/profile views learnable even after the first `cap`
        frontal sightings have filled the budget is the reason for the
        replace-when-full path.
        """
        normed = _normalize(emb)
        if normed is None:
            return None

        person_idxs = [i for i, p in enumerate(self.pids) if p == pid]
        existing = self.embeddings[person_idxs] if person_idxs else np.empty((0, self.dim), dtype=np.float32)
        sims = existing @ normed[0] if existing.size else np.empty((0,), dtype=np.float32)
        if sims.size and float(np.max(sims)) >= self.dup_sim:
            return None  # near-duplicate of a vector this person already has

        if len(person_idxs) < self.cap:
            self.embeddings = np.vstack([self.embeddings, normed])
            self.pids.append(pid)
            self.index.add(normed)
            return len(person_idxs) + 1

        replace_idx = self._redundant_slot(existing, person_idxs)
        self.embeddings[replace_idx] = normed[0]
        self.rebuild_index()
        return len(person_idxs)

    @staticmethod
    def _redundant_slot(existing: np.ndarray, global_idxs: list[int]) -> int:
        """Pick which of a person's vectors to overwrite: the one most redundant
        with the rest (drops the least information)."""
        if len(global_idxs) == 1:
            return global_idxs[0]
        sims = existing @ existing.T
        np.fill_diagonal(sims, -1.0)
        a, b = np.unravel_index(int(np.argmax(sims)), sims.shape)
        mean_a = float(np.mean(np.delete(sims[a], a)))
        mean_b = float(np.mean(np.delete(sims[b], b)))
        return global_idxs[a if mean_a >= mean_b else b]

    def search(self, emb):
        """Return (pid, best_sim, uniqueness_margin).

        `margin` is best_sim minus the best similarity of any *other* pid (1.0 if
        only one pid is present). (None, 0.0, 0.0) for an empty index or a
        missing/degenerate query.
        """
        if self.index.ntotal == 0:
            return None, 0.0, 0.0
        normed = _normalize(emb)
        if normed is None:
            return None, 0.0, 0.0

        k = min(64, self.index.ntotal)
        sims, idxs = self.index.search(normed, k)
        best_pid = None
        best_sim = -1.0
        next_other_sim = -1.0
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            pid = self.pids[int(idx)]
            sim = float(sim)
            if best_pid is None:
                best_pid = pid
                best_sim = sim
            elif pid != best_pid:
                next_other_sim = sim
                break

        if best_pid is None:
            return None, 0.0, 0.0
        margin = 1.0 if next_other_sim < 0 else best_sim - next_other_sim
        return best_pid, best_sim, margin

    def similarity_to_pid(self, pid: int, emb) -> float | None:
        """Max cosine similarity between `emb` and the vectors stored for `pid`.
        None if `emb` is missing/degenerate or `pid` has no vectors here."""
        idxs = [i for i, p in enumerate(self.pids) if p == pid]
        if not idxs:
            return None
        normed = _normalize(emb)
        if normed is None:
            return None
        sims = self.embeddings[idxs] @ normed[0]
        return float(np.max(sims)) if sims.size else None

    def drop_person(self, pid: int) -> None:
        """Remove every vector belonging to `pid` and rebuild the index."""
        keep = [i for i, p in enumerate(self.pids) if p != pid]
        if len(keep) == len(self.pids):
            return  # this person had no vectors here - index already correct
        self.embeddings = (
            self.embeddings[keep] if keep else np.empty((0, self.dim), dtype=np.float32)
        )
        self.pids = [self.pids[i] for i in keep]
        self.rebuild_index()

    def install(self, raw, pids, kind_name: str = "") -> None:
        """Load a persisted (embeddings, pids) pair, validating shape + alignment.

        Tolerates a wrong-shape array (drops the whole channel) or an
        embeddings/pids length mismatch (truncates both to the shorter), logging
        a warning either way - a corrupt load must never crash startup.
        """
        if raw is None:
            self.reset()
            return
        emb = np.asarray(raw, dtype=np.float32)
        if emb.size == 0:
            self.reset()
            return
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            log.warning(
                "%s embeddings have wrong shape %s (expected dim %d) - dropping all.",
                kind_name, emb.shape, self.dim,
            )
            self.reset()
            return
        pids = [int(p) for p in pids]
        if emb.shape[0] != len(pids):
            n = min(emb.shape[0], len(pids))
            log.warning(
                "%s embeddings/pids length mismatch (%d vs %d) - truncating to %d.",
                kind_name, emb.shape[0], len(pids), n,
            )
            emb, pids = emb[:n], pids[:n]
        self.embeddings = emb
        self.pids = pids
        self.rebuild_index()
