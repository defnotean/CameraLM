import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

import cv2
import numpy as np

from .config import (
    BODY_DUPLICATE_SIM,
    DATA_DIR,
    EMBEDDINGS_FILE,
    FACE_DIM,
    FACE_DUPLICATE_SIM,
    IDENTITY_FILE,
    MAX_PARTIAL_EMBEDDINGS_PER_PERSON,
    MAX_EMBEDDINGS_PER_PERSON,
    PARTIAL_DUPLICATE_SIM,
    PARTIAL_DIM,
    REID_DIM,
    SCHEDULE_BLOCKS,
    SCHEDULE_DAYS,
)
from .embedding_store import EmbeddingStore

log = logging.getLogger(__name__)

THUMBNAIL_DIR = DATA_DIR / "thumbnails"
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_FILE = DATA_DIR / "audit.log"


def _empty_schedule() -> dict[str, str]:
    """A fresh schedule with every block unassigned.

    The schedule is per-BLOCK (one class per A/B/C/D, not per (day, block)) -
    a standard rotating block schedule has the same class meeting under the
    same block on every day; only the position of each block within the day
    rotates. SCHEDULE_DAYS only shapes the display grid.
    """
    return {b: "" for b in SCHEDULE_BLOCKS}


def _empty_consent() -> dict[str, str]:
    """Default consent block for a fresh person.

    `status` is the source of truth. The other fields are filled in when consent
    is recorded so the admin UI and audit log can show who attested it and when.
    """
    return {"status": "none", "granted_at": "", "granted_by": "", "notes": ""}


def _now_iso() -> str:
    """Local-time ISO 8601 timestamp to seconds resolution. Same shape as the
    audit log lines so date arithmetic in `purge_stale` is straightforward."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _audit(action: str, detail: str = "") -> None:
    """Append-only accountability log for identity-database mutations."""
    try:
        with open(AUDIT_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"{_now_iso()}\t{action}\t{detail}\n")
    except OSError:
        log.warning("Could not write audit log entry: %s %s", action, detail)


class IdentityDB:
    """FAISS inner-product indexes backed by parallel pid lists and shadow arrays.

    Three `EmbeddingStore` channels carry the appearance evidence: face/body hold
    strong identity signal, partial holds weak visible-region signatures for
    occluded cases. Persons can belong to many classes. All public methods are
    guarded by an internal lock so the admin server and webcam thread can share
    one instance safely.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()   # serializes save() so concurrent callers can't interleave temp files
        self._load_failed = False            # set when load failed AND corrupt files couldn't be backed up - disables save()
        self.face = EmbeddingStore(FACE_DIM, FACE_DUPLICATE_SIM, MAX_EMBEDDINGS_PER_PERSON)
        self.body = EmbeddingStore(REID_DIM, BODY_DUPLICATE_SIM, MAX_EMBEDDINGS_PER_PERSON)
        self.partial = EmbeddingStore(PARTIAL_DIM, PARTIAL_DUPLICATE_SIM, MAX_PARTIAL_EMBEDDINGS_PER_PERSON)
        # kind -> (store, count-key in people[pid]). The single dispatch table that
        # replaced the old _embedding_store_locked / _index_for_kind_locked ladders.
        self._channels = {
            "face": (self.face, "n_face"),
            "body": (self.body, "n_body"),
            "partial": (self.partial, "n_partial"),
        }
        self.people: dict[int, dict] = {}
        self.classes: dict[str, list[int]] = {}
        # Rotating block schedule: {block -> class_name_or_empty}.  The display
        # grid (SCHEDULE_DAYS x len(SCHEDULE_BLOCKS)) is rendered client-side by
        # right-rotating the block order on each day. Validated against
        # `self.classes` so it never points at a phantom class.
        self.schedule: dict[str, str] = _empty_schedule()
        self._next_pid = 1
        self.load()

    # ---- backward-compat attribute shims ----
    # The store internals moved onto EmbeddingStore (db.face.index, db.face.pids,
    # db.face.embeddings). These read-only properties keep the old flat names
    # working for existing callers and the (deliberately frozen) test suite.

    @property
    def face_index(self):
        return self.face.index

    @property
    def body_index(self):
        return self.body.index

    @property
    def partial_index(self):
        return self.partial.index

    @property
    def face_pids(self) -> list[int]:
        return self.face.pids

    @property
    def body_pids(self) -> list[int]:
        return self.body.pids

    @property
    def partial_pids(self) -> list[int]:
        return self.partial.pids

    @property
    def face_embeddings(self) -> np.ndarray:
        return self.face.embeddings

    @property
    def body_embeddings(self) -> np.ndarray:
        return self.body.embeddings

    @property
    def partial_embeddings(self) -> np.ndarray:
        return self.partial.embeddings

    # ---- people ----

    def create_person(self, name: str) -> int:
        with self._lock:
            pid = self._next_pid
            self._next_pid += 1
            self.people[pid] = {
                "name": name,
                "n_face": 0,
                "n_body": 0,
                "n_partial": 0,
                "classes": [],
                "consent": _empty_consent(),
                "last_seen_at": "",
                "created_at": _now_iso(),
            }
            _audit("create_person", f"pid={pid} name={name!r}")
            return pid

    def rename_person(self, pid: int, new_name: str) -> bool:
        with self._lock:
            if pid not in self.people:
                return False
            old = self.people[pid]["name"]
            self.people[pid]["name"] = new_name
            _audit("rename_person", f"pid={pid} {old!r} -> {new_name!r}")
            return True

    def delete_person(self, pid: int) -> bool:
        """Drop person + all their embeddings + class memberships."""
        with self._lock:
            if pid not in self.people:
                return False
            del self.people[pid]

            # Remove from class membership
            for cls_name in list(self.classes.keys()):
                self.classes[cls_name] = [p for p in self.classes[cls_name] if p != pid]

            # Drop the person's vectors from every channel (each rebuilds its index).
            for store, _count_key in self._channels.values():
                store.drop_person(pid)

            # Remove thumbnail file
            tp = self.thumbnail_path(pid)
            if tp.exists():
                try:
                    tp.unlink()
                except OSError:
                    pass
            _audit("delete_person", f"pid={pid}")
            return True

    def clear_embeddings(self, pid: int) -> bool:
        """Drop every face/body/partial vector for `pid` but KEEP the person
        record - name and class memberships are preserved.

        Use to re-enroll a person whose stored identity has drifted or been
        poisoned by auto-learning, without having to rebuild the roster.
        """
        with self._lock:
            if pid not in self.people:
                return False
            for store, count_key in self._channels.values():
                store.drop_person(pid)
                self.people[pid][count_key] = 0
            _audit("clear_embeddings", f"pid={pid}")
            return True

    def delete_all_data(self) -> int:
        """Privacy control: wipe every person, embedding, class, and thumbnail file."""
        with self._lock:
            count = len(self.people)
            for pid in list(self.people.keys()):
                tp = self.thumbnail_path(pid)
                if tp.exists():
                    try:
                        tp.unlink()
                    except OSError:
                        pass
            self._reset_to_empty_locked()
            _audit("delete_all_data", f"{count} people removed")
            return count

    def get_name(self, pid: int) -> str:
        with self._lock:
            return self.people.get(pid, {}).get("name", f"Unknown {pid}")

    def has_person(self, pid: int) -> bool:
        with self._lock:
            return pid in self.people

    def count_people(self) -> int:
        with self._lock:
            return len(self.people)

    def find_person_by_name(self, name: str) -> int | None:
        target = name.strip().casefold()
        if not target:
            return None
        with self._lock:
            for pid, person in self.people.items():
                if str(person.get("name", "")).strip().casefold() == target:
                    return pid
        return None

    def snapshot(self) -> dict:
        """Thread-safe JSON-ready view for the admin server."""
        with self._lock:
            people = []
            for pid in sorted(self.people.keys()):
                person = self.people[pid]
                thumbnail = self.thumbnail_path(pid)
                thumbnail_version = 0
                if thumbnail.exists():
                    thumbnail_version = int(thumbnail.stat().st_mtime * 1000)
                consent = person.get("consent") or _empty_consent()
                people.append({
                    "pid": pid,
                    "name": person.get("name", f"Unknown {pid}"),
                    "classes": list(person.get("classes", [])),
                    "n_face": person.get("n_face", 0),
                    "n_body": person.get("n_body", 0),
                    "n_partial": person.get("n_partial", 0),
                    "has_thumbnail": thumbnail_version > 0,
                    "thumbnail_version": thumbnail_version,
                    "consent": {
                        "status": consent.get("status", "none"),
                        "granted_at": consent.get("granted_at", ""),
                        "granted_by": consent.get("granted_by", ""),
                        "notes": consent.get("notes", ""),
                    },
                    "last_seen_at": person.get("last_seen_at", ""),
                    "created_at": person.get("created_at", ""),
                })
            classes = []
            for name in sorted(self.classes.keys(), key=str.lower):
                members = [pid for pid in self.classes[name] if pid in self.people]
                classes.append({"name": name, "members": members})
            return {
                "people": people,
                "classes": classes,
                "schedule": dict(self.schedule),
                "schedule_dims": {"days": SCHEDULE_DAYS, "blocks": list(SCHEDULE_BLOCKS)},
            }

    def class_names(self) -> list[str]:
        with self._lock:
            return sorted(self.classes.keys(), key=str.lower)

    # ---- embeddings ----

    def _add_to_channel(self, kind: str, pid: int, emb: np.ndarray) -> None:
        """Add one vector to a channel and keep the person's count key in sync."""
        if pid not in self.people:
            return
        store, count_key = self._channels[kind]
        count = store.add(pid, emb)
        if count is not None:   # None == degenerate vector or near-duplicate; count unchanged
            self.people[pid][count_key] = count

    def add_face(self, pid: int, emb: np.ndarray) -> None:
        with self._lock:
            self._add_to_channel("face", pid, emb)

    def add_body(self, pid: int, emb: np.ndarray) -> None:
        with self._lock:
            self._add_to_channel("body", pid, emb)

    def add_partial(self, pid: int, emb: np.ndarray) -> None:
        with self._lock:
            self._add_to_channel("partial", pid, emb)

    def search_face(self, emb: np.ndarray):
        with self._lock:
            pid, sim, _margin = self.face.search(emb)
            return pid, sim

    def search_face_detailed(self, emb: np.ndarray):
        with self._lock:
            return self.face.search(emb)

    def search_body(self, emb: np.ndarray):
        with self._lock:
            pid, sim, _margin = self.body.search(emb)
            return pid, sim

    def search_body_detailed(self, emb: np.ndarray):
        with self._lock:
            return self.body.search(emb)

    def search_partial(self, emb: np.ndarray):
        """Return (pid, similarity, uniqueness_margin) for weak partial evidence."""
        with self._lock:
            return self.partial.search(emb)

    def similarity_to_pid(self, kind: str, pid: int, emb: np.ndarray) -> float | None:
        with self._lock:
            if pid not in self.people:
                return None
            channel = self._channels.get(kind)
            if channel is None:
                raise ValueError(f"Unknown embedding kind: {kind}")
            return channel[0].similarity_to_pid(pid, emb)

    # ---- classes ----

    def create_class(self, name: str) -> bool:
        with self._lock:
            name = name.strip()
            if not name:
                return False
            if name in self.classes:
                return False
            self.classes[name] = []
            _audit("create_class", f"name={name!r}")
            return True

    def delete_class(self, name: str) -> bool:
        with self._lock:
            if name not in self.classes:
                return False
            for pid in self.classes[name]:
                if pid in self.people and name in self.people[pid]["classes"]:
                    self.people[pid]["classes"].remove(name)
            del self.classes[name]
            # Sweep the schedule so no block still points at the deleted class.
            for b in self.schedule:
                if self.schedule[b] == name:
                    self.schedule[b] = ""
            _audit("delete_class", f"name={name!r}")
            return True

    def add_to_class(self, pid: int, class_name: str) -> bool:
        with self._lock:
            class_name = class_name.strip()
            if not class_name or pid not in self.people:
                return False
            if class_name not in self.classes:
                self.classes[class_name] = []
            if pid not in self.classes[class_name]:
                self.classes[class_name].append(pid)
            if class_name not in self.people[pid]["classes"]:
                self.people[pid]["classes"].append(class_name)
            _audit("add_to_class", f"pid={pid} class={class_name!r}")
            return True

    def remove_from_class(self, pid: int, class_name: str) -> bool:
        with self._lock:
            if class_name not in self.classes or pid not in self.people:
                return False
            if pid in self.classes[class_name]:
                self.classes[class_name].remove(pid)
            if class_name in self.people[pid]["classes"]:
                self.people[pid]["classes"].remove(class_name)
            _audit("remove_from_class", f"pid={pid} class={class_name!r}")
            return True

    def classes_of(self, pid: int) -> list[str]:
        with self._lock:
            return list(self.people.get(pid, {}).get("classes", []))

    # ---- privacy: consent, retention, audit ----

    def record_consent(self, pid: int, granted_by: str, notes: str = "") -> bool:
        """Mark consent as granted for `pid`. Stamps the time + who attested it.

        `granted_by` is required (free-text operator name / role). `notes` is
        optional context - e.g. "verbal consent at 2026-03-04 staff meeting" or
        "guardian signed paper form on file". Both are written into the audit
        log.
        """
        granted_by = (granted_by or "").strip()
        if not granted_by:
            return False
        with self._lock:
            if pid not in self.people:
                return False
            consent = self.people[pid].setdefault("consent", _empty_consent())
            consent["status"] = "granted"
            consent["granted_at"] = _now_iso()
            consent["granted_by"] = granted_by[:120]
            consent["notes"] = (notes or "").strip()[:500]
            _audit(
                "record_consent",
                f"pid={pid} by={granted_by!r} notes={consent['notes']!r}",
            )
            return True

    def revoke_consent(self, pid: int) -> bool:
        """Revoke consent for `pid` AND drop all their stored embeddings.

        Revocation is a hard break: the system stops processing this person's
        biometrics. Name + class memberships survive so the admin UI shows a
        revoked entry (operators can then `delete_person` if they want a full
        wipe). Idempotent - revoking a not-granted person still drops vectors
        and audits the action.
        """
        with self._lock:
            if pid not in self.people:
                return False
            for store, count_key in self._channels.values():
                store.drop_person(pid)
                self.people[pid][count_key] = 0
            consent = self.people[pid].setdefault("consent", _empty_consent())
            consent["status"] = "revoked"
            # Keep granted_at / granted_by / notes as the historical record of
            # when consent had been granted - the admin UI can show that this
            # was an active subject before revocation.
            _audit("revoke_consent", f"pid={pid}")
            return True

    def is_consent_granted(self, pid: int) -> bool:
        """Fast read for the live-view consent gate. Defensive: an unknown pid
        is treated as not-granted so a stale display can't accidentally surface
        a deleted identity."""
        with self._lock:
            person = self.people.get(pid)
            if not person:
                return False
            consent = person.get("consent") or {}
            return consent.get("status") == "granted"

    def touch_last_seen(self, pids: Iterable[int], at: str | None = None) -> None:
        """Stamp the current ISO time on every pid in `pids` that exists.

        Called by the pipeline on each recognized track. Takes the lock ONCE
        for the whole batch so the per-frame cost is one acquire/release no
        matter how many people are in view.
        """
        stamp = at or _now_iso()
        with self._lock:
            for pid in pids:
                person = self.people.get(pid)
                if person is not None:
                    person["last_seen_at"] = stamp

    def purge_stale(self, retention_days: int) -> list[dict]:
        """Drop the biometric vectors of anyone not seen in `retention_days`.

        Names + class memberships survive so the audit trail still has a
        meaningful identifier; only the face/body/partial vectors go (the
        identifying biometric data). People with no `last_seen_at` AND no
        `created_at` are conservatively LEFT ALONE - we cannot prove they are
        stale, only that they predate this feature. People with `created_at`
        but no `last_seen_at` are aged from `created_at`.

        `retention_days <= 0` disables the sweep (no-op). Returns the list of
        people whose embeddings were cleared (pid, name, last_seen_at, age_days).
        """
        if retention_days <= 0:
            return []
        cutoff = datetime.now() - timedelta(days=retention_days)
        purged: list[dict] = []
        with self._lock:
            for pid, person in list(self.people.items()):
                stamp = person.get("last_seen_at") or person.get("created_at") or ""
                if not stamp:
                    continue
                try:
                    seen = datetime.fromisoformat(stamp)
                except ValueError:
                    continue
                if seen >= cutoff:
                    continue
                # Stale: drop the vectors but keep the roster entry.
                had_any = any(
                    store.count_for(pid) > 0 for store, _ in self._channels.values()
                )
                if not had_any:
                    # Nothing to purge for this person; skip the audit churn.
                    continue
                for store, count_key in self._channels.values():
                    store.drop_person(pid)
                    person[count_key] = 0
                age_days = (datetime.now() - seen).days
                purged.append({
                    "pid": pid,
                    "name": person.get("name", f"Unknown {pid}"),
                    "last_seen_at": stamp,
                    "age_days": age_days,
                })
                _audit(
                    "purge_stale",
                    f"pid={pid} last_seen={stamp} age_days={age_days}",
                )
        return purged

    def read_audit(self, limit: int = 200, since: str | None = None) -> list[dict]:
        """Read the most recent audit-log entries (newest last).

        `since` is an ISO datetime string; only entries strictly after it are
        returned. Each entry is `{ts, action, detail}`. Malformed lines are
        skipped silently rather than failing the whole read - the audit log is
        append-only and a partial write should not lock out the dashboard.
        """
        if not AUDIT_FILE.exists():
            return []
        try:
            with open(AUDIT_FILE, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return []
        rows: list[dict] = []
        for line in lines:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 2:
                continue
            ts = parts[0]
            action = parts[1]
            detail = parts[2] if len(parts) > 2 else ""
            if since and ts <= since:
                continue
            rows.append({"ts": ts, "action": action, "detail": detail})
        if limit > 0 and len(rows) > limit:
            rows = rows[-limit:]
        return rows

    # ---- schedule ----

    def set_schedule_slot(self, block: str, class_name: str) -> bool:
        """Set or clear one block's class.

        Empty / whitespace `class_name` clears the block. Anything else must be
        the name of an existing class - unknown names are rejected so the
        schedule never carries a dangling reference. Returns True on success.
        """
        with self._lock:
            if block not in self.schedule:
                return False
            name = (class_name or "").strip()
            if name and name not in self.classes:
                return False
            self.schedule[block] = name
            _audit("set_schedule_slot", f"block={block!r} class={name!r}")
            return True

    # ---- thumbnails ----

    def thumbnail_path(self, pid: int):
        return THUMBNAIL_DIR / f"{pid}.jpg"

    def has_thumbnail(self, pid: int) -> bool:
        with self._lock:
            return pid in self.people and self.thumbnail_path(pid).exists()

    def save_thumbnail(self, pid: int, image_bgr: np.ndarray) -> bool:
        with self._lock:
            if pid not in self.people or image_bgr is None or image_bgr.size == 0:
                return False
            h, w = image_bgr.shape[:2]
            if h < 20 or w < 20:
                return False
            # Square-crop centered, then resize to 256 for a stable admin card image.
            side = min(h, w)
            y = (h - side) // 2
            x = (w - side) // 2
            crop = image_bgr[y:y + side, x:x + side]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)

            target = self.thumbnail_path(pid)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp.jpg")
            ok = cv2.imwrite(str(tmp), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return False
            tmp.replace(target)
            return True

    # ---- persistence ----

    def save(self) -> None:
        # Snapshot consistently under the data lock (fast), then do disk I/O
        # outside it. EVERYTHING - metadata, pid lists, and vectors - goes into
        # ONE npz committed with a single os.replace(), so a crash can never
        # desync the pid lists from their embedding rows. EMBEDDINGS_FILE is the
        # source of truth; IDENTITY_FILE is only a human-readable mirror.
        with self._lock:
            if self._load_failed:
                log.warning(
                    "Refusing to save - the DB failed to load and its corrupt files "
                    "could not be backed up; not overwriting potentially recoverable data."
                )
                return
            self._normalize_classes_locked()
            meta_json = json.dumps({
                "next_pid": self._next_pid,
                "people": {str(k): v for k, v in self.people.items()},
                "classes": self.classes,
                "schedule": dict(self.schedule),
            }, indent=2)
            face = self.face.embeddings.copy()
            body = self.body.embeddings.copy()
            partial = self.partial.embeddings.copy()
            face_pids = np.asarray(self.face.pids, dtype=np.int64)
            body_pids = np.asarray(self.body.pids, dtype=np.int64)
            partial_pids = np.asarray(self.partial.pids, dtype=np.int64)

        # Serialize saves so the webcam autosave and an admin mutation can never
        # interleave temp files.
        with self._save_lock:
            tmp = EMBEDDINGS_FILE.with_name(f"{EMBEDDINGS_FILE.name}.{os.getpid()}.tmp")
            try:
                with open(tmp, "wb") as fh:
                    np.savez(
                        fh,
                        face=face, body=body, partial=partial,
                        face_pids=face_pids, body_pids=body_pids, partial_pids=partial_pids,
                        meta=np.asarray(meta_json),
                    )
                os.replace(tmp, EMBEDDINGS_FILE)   # single atomic commit point
            except OSError as exc:
                log.error("Failed to save identity DB: %s", exc)
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                return
        # Best-effort human-readable mirror - NOT load-bearing; load() ignores it.
        try:
            IDENTITY_FILE.write_text(meta_json)
        except OSError:
            pass

    def load(self) -> None:
        with self._lock:
            try:
                self._load_locked()
            except Exception as exc:
                log.error(
                    "Identity DB load failed (%s) - backing up the bad files and starting empty.",
                    exc,
                )
                backed_up = self._backup_corrupt_files()
                self._reset_to_empty_locked()
                if not backed_up:
                    # The on-disk data is corrupt AND we could not preserve a copy.
                    # Disable saving so the next autosave can't overwrite the only
                    # (hand-recoverable) copy with an empty DB.
                    self._load_failed = True
                    log.error(
                        "Could not back up the corrupt DB files - saving is disabled for "
                        "this session to avoid destroying recoverable data."
                    )

    def _load_locked(self) -> None:
        if not EMBEDDINGS_FILE.exists():
            # A pre-single-file install may have only the JSON sidecar.
            if IDENTITY_FILE.exists():
                self._load_legacy_locked()
            return
        arr = np.load(EMBEDDINGS_FILE, allow_pickle=False)
        if "meta" not in arr.files:
            # Old-format npz (vectors only) - metadata still lives in the JSON sidecar.
            self._load_legacy_locked(arr)
            return
        self._apply_meta_locked(json.loads(str(arr["meta"])))
        self._normalize_classes_locked()
        self._install_channel_locked("face", arr)
        self._install_channel_locked("body", arr)
        self._install_channel_locked("partial", arr)
        self._finalize_next_pid_locked()

    def _load_legacy_locked(self, arr=None) -> None:
        """Load the older two-file format: metadata + pid lists in identities.json, vectors in embeddings.npz."""
        if not IDENTITY_FILE.exists():
            return
        meta = json.loads(IDENTITY_FILE.read_text())
        self._apply_meta_locked(meta)
        self._normalize_classes_locked()
        if arr is None and EMBEDDINGS_FILE.exists():
            arr = np.load(EMBEDDINGS_FILE, allow_pickle=False)
        for kind in ("face", "body", "partial"):
            raw = arr[kind] if arr is not None and kind in arr.files else None
            pids = meta.get(f"{kind}_pids", [])
            self._channels[kind][0].install(raw, pids, kind)
        self._finalize_next_pid_locked()

    def _install_channel_locked(self, kind: str, arr) -> None:
        """Install one channel from a single-file npz (vectors + pid list)."""
        raw = arr[kind] if kind in arr.files else None
        pids = arr[f"{kind}_pids"] if f"{kind}_pids" in arr.files else []
        self._channels[kind][0].install(raw, pids, kind)

    def _apply_meta_locked(self, meta: dict) -> None:
        self._next_pid = int(meta.get("next_pid", 1))
        self.people = {int(k): v for k, v in meta.get("people", {}).items()}
        for p in self.people.values():
            p.setdefault("classes", [])
            p.setdefault("n_partial", 0)
            # Pre-privacy installs have no consent / last_seen / created_at
            # fields. Backfill with empty defaults so the rest of the code can
            # read them unconditionally; existing data is treated as "consent
            # never recorded" (which is the safe default).
            consent = p.get("consent")
            if not isinstance(consent, dict):
                consent = _empty_consent()
            else:
                for k, default in _empty_consent().items():
                    consent.setdefault(k, default)
                if consent["status"] not in ("none", "granted", "revoked"):
                    consent["status"] = "none"
            p["consent"] = consent
            p.setdefault("last_seen_at", "")
            p.setdefault("created_at", "")
        self.classes = meta.get("classes", {})
        # Schedule: per-block. Tolerates a legacy per-(day,block) file by
        # collapsing each block to the first non-empty class it had under any
        # day. Out-of-range / wrong-shape / non-string values are dropped.
        raw_sched = meta.get("schedule", {})
        self.schedule = _empty_schedule()
        if isinstance(raw_sched, dict):
            if raw_sched and all(isinstance(v, str) for v in raw_sched.values()):
                for b, cls in raw_sched.items():
                    if b in self.schedule and isinstance(cls, str):
                        self.schedule[b] = cls.strip()
            else:
                # Legacy shape: {day: {block: class}}. Migrate by taking each
                # block's first non-empty assignment across days.
                for blocks in raw_sched.values():
                    if not isinstance(blocks, dict):
                        continue
                    for b, cls in blocks.items():
                        if (
                            b in self.schedule
                            and isinstance(cls, str)
                            and cls.strip()
                            and not self.schedule[b]
                        ):
                            self.schedule[b] = cls.strip()

    def _finalize_next_pid_locked(self) -> None:
        # Never let a freshly created pid collide with an existing person or an
        # embedding row tagged with that pid from a prior life.
        max_pid = max([
            0, *self.people.keys(),
            *self.face.pids, *self.body.pids, *self.partial.pids,
        ])
        self._next_pid = max(self._next_pid, max_pid + 1)
        self._sync_counts_locked()

    def _backup_corrupt_files(self) -> bool:
        """Move the (corrupt) DB files aside so they stay hand-recoverable. Returns True on success."""
        backup_dir = DATA_DIR / "corrupt-backup"
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.error("Could not create corrupt-backup dir: %s", exc)
            return False
        stamp = time.strftime("%Y%m%d-%H%M%S")
        ok = True
        for path in (IDENTITY_FILE, EMBEDDINGS_FILE):
            if path.exists():
                try:
                    path.replace(backup_dir / f"{path.name}.{stamp}")
                except OSError as exc:
                    log.error("Could not back up %s: %s", path, exc)
                    ok = False
        return ok

    def _reset_to_empty_locked(self) -> None:
        for store, _count_key in self._channels.values():
            store.reset()
        self.people = {}
        self.classes = {}
        self.schedule = _empty_schedule()
        self._next_pid = 1

    def _normalize_classes_locked(self) -> None:
        """Keep person-side and class-side memberships in sync."""
        cleaned: dict[str, list[int]] = {}

        for class_name, members in self.classes.items():
            name = str(class_name).strip()
            if not name:
                continue
            cleaned.setdefault(name, [])
            for pid in members:
                try:
                    pid = int(pid)
                except (TypeError, ValueError):
                    continue
                if pid in self.people and pid not in cleaned[name]:
                    cleaned[name].append(pid)

        for pid, person in self.people.items():
            person_classes = []
            for class_name in person.get("classes", []):
                name = str(class_name).strip()
                if not name or name in person_classes:
                    continue
                person_classes.append(name)
                cleaned.setdefault(name, [])
                if pid not in cleaned[name]:
                    cleaned[name].append(pid)
            person["classes"] = person_classes

        for name, members in cleaned.items():
            for pid in members:
                classes = self.people[pid].setdefault("classes", [])
                if name not in classes:
                    classes.append(name)

        self.classes = cleaned

        # Drop schedule entries that point at a class which no longer exists,
        # so the loaded state never carries a dangling reference. The schedule
        # may not have been initialized yet on the first load path - guard.
        if not hasattr(self, "schedule"):
            self.schedule = _empty_schedule()
        known = set(self.classes.keys())
        for b in self.schedule:
            if self.schedule[b] and self.schedule[b] not in known:
                self.schedule[b] = ""

    def _sync_counts_locked(self) -> None:
        for pid, person in self.people.items():
            person["n_face"] = self.face.count_for(pid)
            person["n_body"] = self.body.count_for(pid)
            person["n_partial"] = self.partial.count_for(pid)
