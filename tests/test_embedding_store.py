"""Direct unit tests for EmbeddingStore.

EmbeddingStore is also exercised end-to-end through test_identity_db.py (every
add/search/dedup/cap path runs via IdentityDB); these cover it in isolation so a
regression points straight at the store rather than the DB wiring.
"""

import numpy as np

from cameralm.embedding_store import EmbeddingStore
from helpers import unit_vec


def _store(dim=8, dup_sim=0.985, cap=3):
    return EmbeddingStore(dim=dim, dup_sim=dup_sim, cap=cap)


def test_add_appends_and_counts_per_person():
    s = _store()
    assert s.add(1, unit_vec(8, seed=1)) == 1
    assert s.add(1, unit_vec(8, seed=2)) == 2
    assert s.add(2, unit_vec(8, seed=3)) == 1      # different person, own count
    assert s.index.ntotal == 3
    assert s.count_for(1) == 2
    assert s.count_for(2) == 1


def test_add_rejects_degenerate_and_near_duplicate():
    s = _store()
    assert s.add(1, None) is None                  # missing
    assert s.add(1, np.zeros(8, dtype=np.float32)) is None  # zero vector
    v = unit_vec(8, seed=5)
    assert s.add(1, v) == 1
    assert s.add(1, v) is None                     # exact duplicate - skipped
    assert s.count_for(1) == 1


def test_add_at_cap_replaces_instead_of_growing():
    s = _store(cap=3)
    for seed in range(3):
        s.add(1, unit_vec(8, seed=seed))
    assert s.count_for(1) == 3
    # A fourth distinct vector replaces the most-redundant slot - count holds at cap.
    assert s.add(1, unit_vec(8, seed=99)) == 3
    assert s.count_for(1) == 3
    assert s.index.ntotal == 3


def test_search_returns_pid_similarity_and_margin():
    s = _store()
    va, vb = unit_vec(8, seed=1), unit_vec(8, seed=2)
    s.add(1, va)
    s.add(2, vb)
    pid, sim, margin = s.search(va)
    assert pid == 1
    assert sim > 0.99
    assert 0.0 < margin <= 1.0                     # two pids present -> finite margin
    assert s.search(None) == (None, 0.0, 0.0)
    assert _store().search(va) == (None, 0.0, 0.0)  # empty index


def test_similarity_to_pid_and_drop_person():
    s = _store()
    v = unit_vec(8, seed=7)
    s.add(1, v)
    s.add(2, unit_vec(8, seed=8))
    assert s.similarity_to_pid(1, v) > 0.99
    assert s.similarity_to_pid(99, v) is None      # unknown pid
    s.drop_person(1)
    assert s.count_for(1) == 0
    assert s.similarity_to_pid(1, v) is None
    assert s.index.ntotal == 1                     # person 2 survives


def test_install_validates_shape_and_alignment():
    s = _store(dim=8)
    # Wrong width -> whole channel dropped.
    s.install(np.zeros((3, 5), dtype=np.float32), [1, 2, 3], "face")
    assert s.index.ntotal == 0 and s.pids == []
    # More pids than rows -> truncated to the shorter.
    s.install(np.vstack([unit_vec(8, seed=i) for i in range(2)]), [1, 2, 3, 4], "face")
    assert s.index.ntotal == 2 and s.pids == [1, 2]
    # None / empty -> empty store.
    s.install(None, [1, 2], "face")
    assert s.index.ntotal == 0 and s.pids == []
