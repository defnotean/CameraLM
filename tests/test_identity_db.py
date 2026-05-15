"""Unit tests for cameralm.identity_db.IdentityDB.

Covers people CRUD, embedding add/search, dedup + per-person cap behavior, and
class membership consistency. Persistence (save/load) is intentionally NOT
exercised here - another test module owns that.

Uses the shared ``db`` fixture (empty IdentityDB with persistence pointed at a
tmp dir) and the deterministic vector helpers in ``tests/helpers.py``.
"""

import numpy as np
import pytest

from cameralm.config import (
    BODY_DUPLICATE_SIM,
    FACE_DUPLICATE_SIM,
    MAX_EMBEDDINGS_PER_PERSON,
    PARTIAL_DUPLICATE_SIM,
    SCHEDULE_BLOCKS,
    SCHEDULE_DAYS,
)
from helpers import body_vec, face_vec, near_vec, partial_vec


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _class_members(snapshot, class_name):
    """Pull the member pid list for ``class_name`` out of a snapshot()."""
    for cls in snapshot["classes"]:
        if cls["name"] == class_name:
            return cls["members"]
    return None


def _snapshot_person(snapshot, pid):
    for person in snapshot["people"]:
        if person["pid"] == pid:
            return person
    return None


# --------------------------------------------------------------------------
# people CRUD
# --------------------------------------------------------------------------

def test_create_person_assigns_unique_ids_and_registers(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    assert a != b
    assert db.has_person(a)
    assert db.has_person(b)
    assert db.count_people() == 2
    assert db.get_name(a) == "Alice"
    assert db.get_name(b) == "Bob"


def test_new_person_starts_with_zero_embeddings_and_no_classes(db):
    pid = db.create_person("Alice")
    person = db.people[pid]
    assert person["n_face"] == 0
    assert person["n_body"] == 0
    assert person["n_partial"] == 0
    assert person["classes"] == []
    assert db.classes_of(pid) == []


def test_has_person_and_count_people_on_empty_db(db):
    assert db.count_people() == 0
    assert not db.has_person(1)
    assert not db.has_person(999)


def test_rename_person_changes_name(db):
    pid = db.create_person("Alice")
    assert db.rename_person(pid, "Alicia") is True
    assert db.get_name(pid) == "Alicia"
    assert db.count_people() == 1


def test_rename_missing_person_returns_false(db):
    assert db.rename_person(424242, "Ghost") is False


def test_delete_person_removes_it(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    assert db.delete_person(a) is True
    assert not db.has_person(a)
    assert db.has_person(b)
    assert db.count_people() == 1


def test_delete_missing_person_returns_false(db):
    assert db.delete_person(999) is False


def test_deleted_pid_is_not_reused(db):
    a = db.create_person("Alice")
    db.delete_person(a)
    b = db.create_person("Bob")
    assert b != a


def test_find_person_by_name_exact(db):
    pid = db.create_person("Alice")
    assert db.find_person_by_name("Alice") == pid


def test_find_person_by_name_is_case_insensitive(db):
    pid = db.create_person("Alice")
    assert db.find_person_by_name("alice") == pid
    assert db.find_person_by_name("ALICE") == pid
    assert db.find_person_by_name("aLiCe") == pid


def test_find_person_by_name_is_whitespace_insensitive(db):
    pid = db.create_person("Alice")
    assert db.find_person_by_name("  Alice  ") == pid
    assert db.find_person_by_name("\tAlice\n") == pid


def test_find_person_by_name_case_and_whitespace_combined(db):
    pid = db.create_person("Alice")
    assert db.find_person_by_name("   aLICE  ") == pid


def test_find_person_by_name_missing_returns_none(db):
    db.create_person("Alice")
    assert db.find_person_by_name("Bob") is None


def test_find_person_by_name_empty_returns_none(db):
    db.create_person("Alice")
    assert db.find_person_by_name("") is None
    assert db.find_person_by_name("   ") is None


# --------------------------------------------------------------------------
# embedding add + search
# --------------------------------------------------------------------------

def test_add_face_then_search_returns_that_pid(db):
    pid = db.create_person("Alice")
    vec = face_vec(seed=1)
    db.add_face(pid, vec)
    assert db.people[pid]["n_face"] == 1

    found_pid, sim, _margin = db.search_face_detailed(vec)
    assert found_pid == pid
    assert sim == pytest.approx(1.0, abs=1e-3)


def test_add_body_then_search_returns_that_pid(db):
    pid = db.create_person("Alice")
    vec = body_vec(seed=2)
    db.add_body(pid, vec)
    assert db.people[pid]["n_body"] == 1

    found_pid, sim, _margin = db.search_body_detailed(vec)
    assert found_pid == pid
    assert sim == pytest.approx(1.0, abs=1e-3)


def test_add_partial_then_search_returns_that_pid(db):
    pid = db.create_person("Alice")
    vec = partial_vec(seed=3)
    db.add_partial(pid, vec)
    assert db.people[pid]["n_partial"] == 1

    found_pid, sim, _margin = db.search_partial(vec)
    assert found_pid == pid
    assert sim == pytest.approx(1.0, abs=1e-3)


def test_search_discriminates_between_two_people(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    va = face_vec(seed=10)
    vb = face_vec(seed=11)
    db.add_face(a, va)
    db.add_face(b, vb)

    assert db.search_face_detailed(va)[0] == a
    assert db.search_face_detailed(vb)[0] == b


def test_search_on_empty_index_returns_none(db):
    pid = db.create_person("Alice")  # exists, but has no embeddings
    assert db.search_face_detailed(face_vec(seed=1))[0] is None
    assert db.search_body_detailed(body_vec(seed=1))[0] is None
    assert db.search_partial(partial_vec(seed=1))[0] is None


def test_add_embedding_for_missing_person_is_ignored(db):
    db.add_face(999, face_vec(seed=1))
    assert db.face_index.ntotal == 0
    assert len(db.face_pids) == 0


def test_distinct_face_vectors_accumulate(db):
    pid = db.create_person("Alice")
    for seed in range(5):
        db.add_face(pid, face_vec(seed=100 + seed))
    assert db.people[pid]["n_face"] == 5
    assert db.face_index.ntotal == 5


# --------------------------------------------------------------------------
# DEDUP - a near-identical vector must not grow the count
# --------------------------------------------------------------------------

def test_dedup_face_near_vector_does_not_increase_count(db):
    pid = db.create_person("Alice")
    base = face_vec(seed=1)
    db.add_face(pid, base)
    assert db.people[pid]["n_face"] == 1

    dup = near_vec(base, jitter=0.001, seed=7)
    # sanity: the near vector really is above the dedup threshold
    assert db.similarity_to_pid("face", pid, dup) >= FACE_DUPLICATE_SIM

    db.add_face(pid, dup)
    assert db.people[pid]["n_face"] == 1
    assert db.face_index.ntotal == 1


def test_dedup_body_near_vector_does_not_increase_count(db):
    pid = db.create_person("Alice")
    base = body_vec(seed=2)
    db.add_body(pid, base)
    assert db.people[pid]["n_body"] == 1

    dup = near_vec(base, jitter=0.001, seed=8)
    assert db.similarity_to_pid("body", pid, dup) >= BODY_DUPLICATE_SIM

    db.add_body(pid, dup)
    assert db.people[pid]["n_body"] == 1
    assert db.body_index.ntotal == 1


def test_dedup_partial_near_vector_does_not_increase_count(db):
    pid = db.create_person("Alice")
    base = partial_vec(seed=3)
    db.add_partial(pid, base)
    assert db.people[pid]["n_partial"] == 1

    dup = near_vec(base, jitter=0.001, seed=9)
    assert db.similarity_to_pid("partial", pid, dup) >= PARTIAL_DUPLICATE_SIM

    db.add_partial(pid, dup)
    assert db.people[pid]["n_partial"] == 1
    assert db.partial_index.ntotal == 1


def test_dedup_is_per_person_not_global(db):
    """A near-duplicate of Alice's vector is still a fresh vector for Bob."""
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    base = face_vec(seed=1)
    db.add_face(a, base)

    dup = near_vec(base, jitter=0.001, seed=7)
    db.add_face(b, dup)
    # Bob has never seen this vector before, so it must be stored for him.
    assert db.people[b]["n_face"] == 1
    assert db.face_index.ntotal == 2


# --------------------------------------------------------------------------
# CAP - past MAX_EMBEDDINGS_PER_PERSON the count holds at the cap
# --------------------------------------------------------------------------

def test_face_cap_holds_at_max(db):
    pid = db.create_person("Alice")
    # Add well past the cap with distinct (near-orthogonal) vectors.
    for seed in range(MAX_EMBEDDINGS_PER_PERSON + 7):
        db.add_face(pid, face_vec(seed=500 + seed))
    assert db.people[pid]["n_face"] == MAX_EMBEDDINGS_PER_PERSON
    assert db.face_pids.count(pid) == MAX_EMBEDDINGS_PER_PERSON
    # The redundant-slot-replace path rebuilds the index, so it stays in sync.
    assert db.face_index.ntotal == MAX_EMBEDDINGS_PER_PERSON


def test_body_cap_holds_at_max(db):
    pid = db.create_person("Alice")
    for seed in range(MAX_EMBEDDINGS_PER_PERSON + 5):
        db.add_body(pid, body_vec(seed=700 + seed))
    assert db.people[pid]["n_body"] == MAX_EMBEDDINGS_PER_PERSON
    assert db.body_pids.count(pid) == MAX_EMBEDDINGS_PER_PERSON
    assert db.body_index.ntotal == MAX_EMBEDDINGS_PER_PERSON


def test_cap_replace_keeps_person_searchable(db):
    """Even after slot-replacement, the person is still found by a stored vector."""
    pid = db.create_person("Alice")
    last_vec = None
    for seed in range(MAX_EMBEDDINGS_PER_PERSON + 4):
        last_vec = face_vec(seed=900 + seed)
        db.add_face(pid, last_vec)
    # The most recently added vector overwrote a redundant slot, so it is present.
    found_pid, sim, _margin = db.search_face_detailed(last_vec)
    assert found_pid == pid
    assert sim == pytest.approx(1.0, abs=1e-3)


def test_cap_does_not_leak_across_people(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    for seed in range(MAX_EMBEDDINGS_PER_PERSON + 3):
        db.add_face(a, face_vec(seed=1000 + seed))
    db.add_face(b, face_vec(seed=2000))
    assert db.people[a]["n_face"] == MAX_EMBEDDINGS_PER_PERSON
    assert db.people[b]["n_face"] == 1


# --------------------------------------------------------------------------
# CLASSES - person.classes list and the classes dict stay bidirectional
# --------------------------------------------------------------------------

def test_create_class(db):
    assert db.create_class("students") is True
    assert "students" in db.classes
    assert db.classes["students"] == []
    assert "students" in db.class_names()


def test_create_class_duplicate_and_blank_return_false(db):
    assert db.create_class("students") is True
    assert db.create_class("students") is False
    assert db.create_class("") is False
    assert db.create_class("   ") is False


def test_add_to_class_updates_both_sides(db):
    pid = db.create_person("Alice")
    db.create_class("students")
    assert db.add_to_class(pid, "students") is True

    # person side
    assert "students" in db.classes_of(pid)
    assert "students" in db.people[pid]["classes"]
    # class side
    assert pid in db.classes["students"]

    # snapshot() agrees with classes_of()
    snap = db.snapshot()
    assert _class_members(snap, "students") == [pid]
    assert _snapshot_person(snap, pid)["classes"] == ["students"]


def test_add_to_class_autocreates_class(db):
    pid = db.create_person("Alice")
    assert db.add_to_class(pid, "vips") is True
    assert "vips" in db.classes
    assert pid in db.classes["vips"]
    assert "vips" in db.classes_of(pid)


def test_add_to_class_missing_person_returns_false(db):
    db.create_class("students")
    assert db.add_to_class(999, "students") is False
    assert db.classes["students"] == []


def test_add_to_class_is_idempotent(db):
    pid = db.create_person("Alice")
    db.add_to_class(pid, "students")
    db.add_to_class(pid, "students")
    assert db.classes["students"].count(pid) == 1
    assert db.classes_of(pid).count("students") == 1


def test_person_in_multiple_classes(db):
    pid = db.create_person("Alice")
    db.add_to_class(pid, "students")
    db.add_to_class(pid, "vips")
    assert sorted(db.classes_of(pid)) == ["students", "vips"]
    assert pid in db.classes["students"]
    assert pid in db.classes["vips"]


def test_remove_from_class_updates_both_sides(db):
    pid = db.create_person("Alice")
    db.add_to_class(pid, "students")
    assert db.remove_from_class(pid, "students") is True

    assert "students" not in db.classes_of(pid)
    assert "students" not in db.people[pid]["classes"]
    assert pid not in db.classes["students"]
    # the class itself still exists, just empty
    assert "students" in db.classes

    snap = db.snapshot()
    assert _class_members(snap, "students") == []
    assert _snapshot_person(snap, pid)["classes"] == []


def test_remove_from_class_missing_class_or_person_returns_false(db):
    pid = db.create_person("Alice")
    assert db.remove_from_class(pid, "nonexistent") is False
    db.create_class("students")
    assert db.remove_from_class(999, "students") is False


def test_remove_from_one_class_leaves_others(db):
    pid = db.create_person("Alice")
    db.add_to_class(pid, "students")
    db.add_to_class(pid, "vips")
    db.remove_from_class(pid, "students")
    assert db.classes_of(pid) == ["vips"]
    assert pid not in db.classes["students"]
    assert pid in db.classes["vips"]


def test_delete_class_removes_it_from_every_member(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    db.add_to_class(a, "students")
    db.add_to_class(b, "students")
    assert db.delete_class("students") is True

    assert "students" not in db.classes
    assert "students" not in db.classes_of(a)
    assert "students" not in db.classes_of(b)

    snap = db.snapshot()
    assert _class_members(snap, "students") is None


def test_delete_missing_class_returns_false(db):
    assert db.delete_class("ghosts") is False


def test_delete_class_leaves_other_class_memberships(db):
    pid = db.create_person("Alice")
    db.add_to_class(pid, "students")
    db.add_to_class(pid, "vips")
    db.delete_class("students")
    assert db.classes_of(pid) == ["vips"]
    assert pid in db.classes["vips"]


def test_class_membership_bidirectional_after_sequence(db):
    """Run a churn of class ops and verify both indexes still agree."""
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    c = db.create_person("Carol")

    db.add_to_class(a, "students")
    db.add_to_class(b, "students")
    db.add_to_class(c, "students")
    db.add_to_class(a, "vips")
    db.remove_from_class(b, "students")
    db.add_to_class(b, "vips")
    db.delete_class("vips")
    db.add_to_class(c, "alumni")

    # Cross-check: every class member lists the class, and vice versa.
    for class_name, members in db.classes.items():
        for pid in members:
            assert class_name in db.classes_of(pid), (
                f"{pid} in classes[{class_name!r}] but not classes_of({pid})"
            )
    for pid in db.people:
        for class_name in db.classes_of(pid):
            assert pid in db.classes.get(class_name, []), (
                f"classes_of({pid}) has {class_name!r} but classes dict disagrees"
            )

    # And the concrete expected end state.
    assert db.classes_of(a) == ["students"]
    assert db.classes_of(b) == []
    assert sorted(db.classes_of(c)) == ["alumni", "students"]
    assert "vips" not in db.classes


# --------------------------------------------------------------------------
# delete_person - drops embeddings and class memberships
# --------------------------------------------------------------------------

def test_delete_person_removes_from_all_classes(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    db.add_to_class(a, "students")
    db.add_to_class(a, "vips")
    db.add_to_class(b, "students")

    db.delete_person(a)

    assert a not in db.classes["students"]
    assert a not in db.classes["vips"]
    # Bob's membership is untouched.
    assert b in db.classes["students"]

    snap = db.snapshot()
    assert _class_members(snap, "students") == [b]
    assert _class_members(snap, "vips") == []


def test_delete_person_drops_their_embeddings(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    a_face = face_vec(seed=1)
    a_body = body_vec(seed=2)
    a_partial = partial_vec(seed=3)
    db.add_face(a, a_face)
    db.add_body(a, a_body)
    db.add_partial(a, a_partial)
    db.add_face(b, face_vec(seed=20))

    db.delete_person(a)

    # Alice is gone from people, pid lists, and the FAISS indexes.
    assert not db.has_person(a)
    assert a not in db.face_pids
    assert a not in db.body_pids
    assert a not in db.partial_pids
    assert db.face_index.ntotal == 1  # only Bob's face remains
    assert db.body_index.ntotal == 0
    assert db.partial_index.ntotal == 0

    # Searching with Alice's own vectors must no longer return her.
    assert db.search_face_detailed(a_face)[0] != a
    assert db.search_body_detailed(a_body)[0] is None
    assert db.search_partial(a_partial)[0] is None

    # Bob is still intact and searchable.
    assert db.people[b]["n_face"] == 1


def test_delete_person_count_keys_consistent_for_survivors(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    db.add_face(a, face_vec(seed=1))
    db.add_face(b, face_vec(seed=2))
    db.add_body(b, body_vec(seed=3))

    db.delete_person(a)

    assert db.people[b]["n_face"] == 1
    assert db.people[b]["n_body"] == 1
    assert db.body_index.ntotal == 1


def test_delete_person_with_multiple_embeddings_clears_all(db):
    pid = db.create_person("Alice")
    for seed in range(6):
        db.add_face(pid, face_vec(seed=300 + seed))
    assert db.face_index.ntotal == 6

    db.delete_person(pid)
    assert db.face_index.ntotal == 0
    assert db.face_pids == []
    assert db.face_embeddings.shape[0] == 0


# --------------------------------------------------------------------------
# delete_all_data - full wipe
# --------------------------------------------------------------------------

def test_delete_all_data_wipes_everything(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    db.add_face(a, face_vec(seed=1))
    db.add_body(a, body_vec(seed=2))
    db.add_partial(b, partial_vec(seed=3))
    db.add_to_class(a, "students")
    db.add_to_class(b, "students")
    db.create_class("vips")

    removed = db.delete_all_data()
    assert removed == 2

    # No people.
    assert db.count_people() == 0
    assert not db.has_person(a)
    assert not db.has_person(b)

    # No classes.
    assert db.classes == {}
    assert db.class_names() == []

    # No embeddings in any index or pid list.
    assert db.face_index.ntotal == 0
    assert db.body_index.ntotal == 0
    assert db.partial_index.ntotal == 0
    assert db.face_pids == []
    assert db.body_pids == []
    assert db.partial_pids == []
    assert db.face_embeddings.shape[0] == 0
    assert db.body_embeddings.shape[0] == 0
    assert db.partial_embeddings.shape[0] == 0

    # Snapshot reflects the empty state.
    snap = db.snapshot()
    assert snap["people"] == []
    assert snap["classes"] == []


def test_delete_all_data_on_empty_db_returns_zero(db):
    assert db.delete_all_data() == 0
    assert db.count_people() == 0


def test_db_is_usable_after_delete_all_data(db):
    db.create_person("Alice")
    db.delete_all_data()

    # A fresh person can still be created and gets embeddings/classes normally.
    pid = db.create_person("Bob")
    db.add_face(pid, face_vec(seed=42))
    db.add_to_class(pid, "students")
    assert db.count_people() == 1
    assert db.people[pid]["n_face"] == 1
    assert db.search_face_detailed(face_vec(seed=42))[0] == pid
    assert db.classes_of(pid) == ["students"]


# --------------------------------------------------------------------------
# clear_embeddings - drop a person's vectors, keep their roster entry
# --------------------------------------------------------------------------

def test_clear_embeddings_keeps_person_and_classes(db):
    pid = db.create_person("Ian")
    db.add_face(pid, face_vec(seed=1))
    db.add_body(pid, body_vec(seed=2))
    db.add_partial(pid, partial_vec(seed=3))
    db.add_to_class(pid, "Staff")

    assert db.clear_embeddings(pid) is True

    # Embeddings gone from counts and indexes...
    assert db.people[pid]["n_face"] == 0
    assert db.people[pid]["n_body"] == 0
    assert db.people[pid]["n_partial"] == 0
    assert db.face_index.ntotal == 0
    assert db.body_index.ntotal == 0
    assert db.partial_index.ntotal == 0
    assert pid not in db.face_pids
    # ...but the person record + class membership survive.
    assert db.has_person(pid)
    assert db.get_name(pid) == "Ian"
    assert db.classes_of(pid) == ["Staff"]
    assert pid in db.classes["Staff"]


def test_clear_embeddings_only_affects_target(db):
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    db.add_face(a, face_vec(seed=1))
    db.add_face(b, face_vec(seed=2))

    db.clear_embeddings(a)

    assert db.people[a]["n_face"] == 0
    assert db.people[b]["n_face"] == 1            # Bob untouched
    assert db.face_index.ntotal == 1
    assert db.search_face_detailed(face_vec(seed=2))[0] == b


def test_clear_embeddings_unknown_pid_returns_false(db):
    assert db.clear_embeddings(999) is False


def test_db_is_usable_after_clear_embeddings(db):
    """After clearing, the person can be re-enrolled normally - the point of
    the operation (re-enroll a poisoned identity without losing the roster)."""
    pid = db.create_person("Ian")
    db.add_face(pid, face_vec(seed=1))
    db.clear_embeddings(pid)

    db.add_face(pid, face_vec(seed=99))           # fresh enrollment
    assert db.people[pid]["n_face"] == 1
    new_pid, new_sim, _margin = db.search_face_detailed(face_vec(seed=99))
    assert new_pid == pid and new_sim > 0.99      # the new vector matches strongly
    # The old (seed=1) vector is gone: it no longer resembles anything stored.
    # (search returns the nearest pid regardless of similarity - so assert the
    # SIMILARITY collapsed, not the pid.)
    _old_pid, old_sim, _ = db.search_face_detailed(face_vec(seed=1))
    assert old_sim < 0.5


# --------------------------------------------------------------------------
# schedule (per-block; the display grid is rotated client-side)
# --------------------------------------------------------------------------

def test_initial_schedule_is_empty_per_block(db):
    """A fresh DB has one empty entry per block - the schedule is per-BLOCK,
    not per (day, block). SCHEDULE_DAYS only shapes the display grid."""
    assert set(db.schedule.keys()) == set(SCHEDULE_BLOCKS)
    assert all(v == "" for v in db.schedule.values())


def test_set_schedule_slot_valid_class_succeeds(db):
    db.create_class("Math 10")
    assert db.set_schedule_slot("A", "Math 10") is True
    assert db.schedule["A"] == "Math 10"


def test_set_schedule_slot_clears_with_empty_string(db):
    db.create_class("Math 10")
    db.set_schedule_slot("A", "Math 10")
    assert db.set_schedule_slot("A", "") is True
    assert db.schedule["A"] == ""


def test_set_schedule_slot_unknown_class_is_rejected(db):
    """Schedules may never carry a reference to a class that doesn't exist."""
    assert db.set_schedule_slot("A", "Phantom") is False
    assert db.schedule["A"] == ""


def test_set_schedule_slot_out_of_range_block_is_rejected(db):
    db.create_class("Math 10")
    assert db.set_schedule_slot("Z", "Math 10") is False


def test_delete_class_clears_schedule_references(db):
    """Deleting a class sweeps the schedule so no block still points at it."""
    db.create_class("Math 10")
    db.create_class("ELA 9")
    db.set_schedule_slot("A", "Math 10")
    db.set_schedule_slot("C", "Math 10")
    db.set_schedule_slot("B", "ELA 9")

    db.delete_class("Math 10")

    assert db.schedule["A"] == ""
    assert db.schedule["C"] == ""
    assert db.schedule["B"] == "ELA 9"          # untouched
    assert "Math 10" not in db.classes


def test_snapshot_includes_schedule_and_dims(db):
    db.create_class("Math 10")
    db.set_schedule_slot("B", "Math 10")
    snap = db.snapshot()
    assert snap["schedule_dims"]["days"] == SCHEDULE_DAYS
    assert snap["schedule_dims"]["blocks"] == list(SCHEDULE_BLOCKS)
    assert snap["schedule"]["B"] == "Math 10"
