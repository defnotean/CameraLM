"""Shared pytest fixtures for the CameraLM test suite.

The ``db`` fixture redirects *all* IdentityDB persistence paths into a pytest
tmp dir before the instance is constructed, so no test ever reads or writes the
real ``data/`` folder.

``cameralm/identity_db.py`` references its path constants by bare name:
  - ``IDENTITY_FILE`` / ``EMBEDDINGS_FILE`` are imported from ``cameralm.config``
    into the ``cameralm.identity_db`` namespace.
  - ``THUMBNAIL_DIR`` / ``AUDIT_FILE`` are module-level in ``cameralm.identity_db``.
  - ``DATA_DIR`` is imported from ``cameralm.config`` into ``cameralm.identity_db``
    and used by ``_backup_corrupt_files`` (corrupt-backup dir).
So every one of those names must be patched on the ``cameralm.identity_db``
module object. ``cameralm.config.DATA_DIR`` is patched too for completeness.
"""

import pytest

import cameralm.config as config
import cameralm.identity_db as identity_db
from cameralm.identity_db import IdentityDB

from helpers import body_vec, face_vec, partial_vec


@pytest.fixture
def db(monkeypatch, tmp_path):
    """A fresh, empty ``IdentityDB`` with all persistence redirected to tmp_path.

    Paths are repointed *before* ``IdentityDB()`` is constructed, so ``load()``
    finds nothing and the instance starts empty.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir = data_dir / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    # Patch the bare names actually referenced inside cameralm.identity_db.
    monkeypatch.setattr(identity_db, "DATA_DIR", data_dir)
    monkeypatch.setattr(identity_db, "IDENTITY_FILE", data_dir / "identities.json")
    monkeypatch.setattr(identity_db, "EMBEDDINGS_FILE", data_dir / "embeddings.npz")
    monkeypatch.setattr(identity_db, "THUMBNAIL_DIR", thumbnail_dir)
    monkeypatch.setattr(identity_db, "AUDIT_FILE", data_dir / "audit.log")
    # Also patch the source of truth in cameralm.config.
    monkeypatch.setattr(config, "DATA_DIR", data_dir)

    return IdentityDB()


@pytest.fixture
def populated_db(db):
    """An ``IdentityDB`` with 3 people, each with face/body/partial embeddings.

    Per person: 3 face + 2 body + 2 partial embeddings, all added through the
    public ``add_face`` / ``add_body`` / ``add_partial`` methods using distinct
    seeds so the vectors are non-duplicate.
    """
    seed = 0
    for i in range(3):
        pid = db.create_person(f"Person {i + 1}")
        for _ in range(3):
            db.add_face(pid, face_vec(seed=seed))
            seed += 1
        for _ in range(2):
            db.add_body(pid, body_vec(seed=seed))
            seed += 1
        for _ in range(2):
            db.add_partial(pid, partial_vec(seed=seed))
            seed += 1
    return db
