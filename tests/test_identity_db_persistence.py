"""Persistence tests for IdentityDB.

The DB persists everything - metadata, pid lists, and embedding vectors - into
ONE npz (EMBEDDINGS_FILE) committed with a single os.replace(), so a crash can
never desync pid lists from their vectors. IDENTITY_FILE is only a best-effort
human-readable mirror and is NOT load-bearing. These tests cover the single-file
round-trip, corruption recovery, the legacy two-file migration path, the
_next_pid anti-collision rule, and the _load_failed save-disable guard.
"""

import json

import numpy as np
import pytest

import cameralm.identity_db as idb
from cameralm.identity_db import IdentityDB
from helpers import body_vec, face_vec, partial_vec


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Redirect every IdentityDB persistence path into a fresh tmp dir."""
    d = tmp_path / "data"
    d.mkdir()
    thumbs = d / "thumbnails"
    thumbs.mkdir()
    monkeypatch.setattr(idb, "DATA_DIR", d)
    monkeypatch.setattr(idb, "IDENTITY_FILE", d / "identities.json")
    monkeypatch.setattr(idb, "EMBEDDINGS_FILE", d / "embeddings.npz")
    monkeypatch.setattr(idb, "THUMBNAIL_DIR", thumbs)
    monkeypatch.setattr(idb, "AUDIT_FILE", d / "audit.log")
    return d


def _populate(db: IdentityDB) -> IdentityDB:
    """3 people, each with classes + face/body/partial embeddings."""
    for i in range(1, 4):
        pid = db.create_person(f"Person {i}")
        db.add_face(pid, face_vec(seed=i * 10 + 1))
        db.add_face(pid, face_vec(seed=i * 10 + 2))
        db.add_body(pid, body_vec(seed=i * 10 + 3))
        db.add_partial(pid, partial_vec(seed=i * 10 + 4))
        db.add_to_class(pid, "Grade 8A" if i < 3 else "Staff")
    return db


def test_round_trip_preserves_everything(data_dir):
    db = _populate(IdentityDB())
    db.save()

    fresh = IdentityDB()
    assert fresh.count_people() == 3
    assert {p["name"] for p in fresh.people.values()} == {p["name"] for p in db.people.values()}
    assert fresh.classes == db.classes
    assert fresh._next_pid == db._next_pid
    np.testing.assert_array_equal(fresh.face_embeddings, db.face_embeddings)
    np.testing.assert_array_equal(fresh.body_embeddings, db.body_embeddings)
    np.testing.assert_array_equal(fresh.partial_embeddings, db.partial_embeddings)
    assert fresh.face_pids == db.face_pids
    assert fresh.body_pids == db.body_pids
    assert fresh.partial_pids == db.partial_pids
    assert fresh.face_index.ntotal == db.face_index.ntotal


def test_save_is_single_atomic_file_no_tmp_left(data_dir):
    db = _populate(IdentityDB())
    db.save()
    assert (data_dir / "embeddings.npz").exists()    # the authoritative file
    assert (data_dir / "identities.json").exists()   # the human-readable mirror
    assert list(data_dir.glob("*.tmp")) == []        # no leftover temp files


def test_json_mirror_is_not_load_bearing(data_dir):
    db = _populate(IdentityDB())
    db.save()
    # Corrupt the human-readable mirror - load() must ignore it and read the npz.
    (data_dir / "identities.json").write_text("}{ this is not json")
    fresh = IdentityDB()
    assert fresh.count_people() == 3
    assert fresh.face_index.ntotal == db.face_index.ntotal


def test_corrupt_npz_recovers_empty_with_backup(data_dir):
    db = _populate(IdentityDB())
    db.save()
    (data_dir / "embeddings.npz").write_bytes(b"\x00\x01 not a valid npz \xff")
    fresh = IdentityDB()                              # must NOT raise
    assert fresh.count_people() == 0
    assert fresh.face_index.ntotal == 0
    backups = list((data_dir / "corrupt-backup").glob("embeddings.npz.*"))
    assert backups, "the corrupt npz should have been backed up before reset"


def test_dim_mismatch_drops_that_index_only(data_dir):
    db = _populate(IdentityDB())
    db.save()
    # Rebuild the npz with a wrong-width `face` array; everything else stays valid.
    arr = dict(np.load(data_dir / "embeddings.npz", allow_pickle=False))
    arr["face"] = np.zeros((len(db.face_pids), idb.FACE_DIM + 9), dtype=np.float32)
    np.savez(data_dir / "embeddings.npz", **arr)
    fresh = IdentityDB()                              # must NOT raise
    assert fresh.count_people() == 3                  # people still load
    assert fresh.face_index.ntotal == 0               # bad face index dropped
    assert fresh.body_index.ntotal == db.body_index.ntotal      # body survives
    assert fresh.partial_index.ntotal == db.partial_index.ntotal


def test_pids_longer_than_rows_is_realigned(data_dir):
    db = _populate(IdentityDB())
    db.save()
    arr = dict(np.load(data_dir / "embeddings.npz", allow_pickle=False))
    # Append two phantom pids with no matching embedding rows.
    arr["face_pids"] = np.concatenate([arr["face_pids"], np.array([999, 1000], dtype=np.int64)])
    np.savez(data_dir / "embeddings.npz", **arr)
    fresh = IdentityDB()                              # must NOT raise
    assert len(fresh.face_pids) == fresh.face_index.ntotal     # realigned + consistent
    assert 999 not in fresh.face_pids


def test_next_pid_advances_past_all_pids(data_dir):
    db = _populate(IdentityDB())
    db.save()
    arr = dict(np.load(data_dir / "embeddings.npz", allow_pickle=False))
    meta = json.loads(str(arr["meta"]))
    meta["next_pid"] = 1                               # stale: lower than existing pids
    arr["meta"] = np.asarray(json.dumps(meta))
    # Tag an embedding row with a high pid that has no `people` entry.
    arr["body_pids"] = np.concatenate([arr["body_pids"], np.array([777], dtype=np.int64)])
    arr["body"] = np.vstack([arr["body"], body_vec(seed=999).reshape(1, -1)])
    np.savez(data_dir / "embeddings.npz", **arr)
    fresh = IdentityDB()
    assert fresh._next_pid > 777
    assert fresh.create_person("Newcomer") > 777


def test_load_failed_flag_disables_save(data_dir):
    db = _populate(IdentityDB())
    db.save()
    original = (data_dir / "embeddings.npz").read_bytes()
    # Simulate "loaded into a failed, un-backed-up state": save() must refuse to
    # overwrite the on-disk data.
    db._load_failed = True
    db.create_person("Should Not Persist")
    db.save()
    assert (data_dir / "embeddings.npz").read_bytes() == original


def test_legacy_two_file_format_still_loads(data_dir):
    """A pre-single-file install: metadata + pid lists in identities.json,
    vectors in a pidless / meta-less embeddings.npz."""
    legacy_meta = {
        "next_pid": 3,
        "people": {
            "1": {"name": "Legacy One", "n_face": 1, "n_body": 0, "n_partial": 0, "classes": ["Old Class"]},
            "2": {"name": "Legacy Two", "n_face": 0, "n_body": 1, "n_partial": 0, "classes": []},
        },
        "classes": {"Old Class": [1]},
        "face_pids": [1],
        "body_pids": [2],
        "partial_pids": [],
    }
    (data_dir / "identities.json").write_text(json.dumps(legacy_meta))
    np.savez(
        data_dir / "embeddings.npz",
        face=face_vec(seed=1).reshape(1, -1),
        body=body_vec(seed=2).reshape(1, -1),
        partial=np.empty((0, idb.PARTIAL_DIM), dtype=np.float32),
    )  # no `meta` / `*_pids` keys -> triggers the legacy load path
    db = IdentityDB()
    assert db.count_people() == 2
    assert db.get_name(1) == "Legacy One"
    assert db.face_index.ntotal == 1
    assert db.body_index.ntotal == 1
    assert "Old Class" in db.classes


def test_schedule_round_trips(data_dir):
    """save() -> fresh IdentityDB preserves every block's class; the rest of
    the (empty) per-block schedule is rebuilt from the configured dimensions."""
    db = IdentityDB()
    db.create_class("Math 10")
    db.create_class("ELA 9")
    db.set_schedule_slot("A", "Math 10")
    db.set_schedule_slot("B", "ELA 9")
    db.save()

    fresh = IdentityDB()
    assert fresh.schedule["A"] == "Math 10"
    assert fresh.schedule["B"] == "ELA 9"
    # Empty blocks stay empty strings (not None / missing).
    assert fresh.schedule["C"] == ""
    assert fresh.schedule["D"] == ""


def test_schedule_dangling_reference_cleared_on_load(data_dir):
    """A schedule block that points at a class the load doesn't know about
    gets cleared on the next load - no zombie references survive."""
    import json

    db = IdentityDB()
    db.create_class("Will Vanish")
    db.set_schedule_slot("A", "Will Vanish")
    db.save()

    # Reach into the persisted meta and remove the class while leaving the
    # schedule entry that referenced it.
    arr = dict(np.load(data_dir / "embeddings.npz", allow_pickle=False))
    meta = json.loads(str(arr["meta"]))
    meta["classes"].pop("Will Vanish", None)
    arr["meta"] = np.asarray(json.dumps(meta))
    np.savez(data_dir / "embeddings.npz", **arr)

    fresh = IdentityDB()
    assert "Will Vanish" not in fresh.classes
    assert fresh.schedule["A"] == ""


def test_legacy_per_day_schedule_migrates_to_per_block(data_dir):
    """A file saved by the previous per-(day, block) schedule shape collapses
    on load - each block keeps the first non-empty class it had across days."""
    import json

    db = IdentityDB()
    db.create_class("Math 10")
    db.create_class("ELA 9")
    db.save()

    # Rewrite meta with the OLD shape: {day: {block: class}}.
    arr = dict(np.load(data_dir / "embeddings.npz", allow_pickle=False))
    meta = json.loads(str(arr["meta"]))
    meta["schedule"] = {
        "1": {"A": "Math 10", "B": "", "C": "", "D": ""},
        "2": {"A": "Math 10", "B": "ELA 9", "C": "", "D": ""},  # B's first non-empty
        "3": {"A": "", "B": "", "C": "", "D": ""},
    }
    arr["meta"] = np.asarray(json.dumps(meta))
    np.savez(data_dir / "embeddings.npz", **arr)

    fresh = IdentityDB()
    assert fresh.schedule["A"] == "Math 10"
    assert fresh.schedule["B"] == "ELA 9"
    assert fresh.schedule["C"] == ""
    assert fresh.schedule["D"] == ""
