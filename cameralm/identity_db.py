import json
import logging
import os
import threading
import time

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


def _audit(action: str, detail: str = "") -> None:
    """Append-only accountability log for identity-database mutations."""
    try:
        with open(AUDIT_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{action}\t{detail}\n")
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
            self.people[pid] = {"name": name, "n_face": 0, "n_body": 0, "n_partial": 0, "classes": []}
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
                people.append({
                    "pid": pid,
                    "name": person.get("name", f"Unknown {pid}"),
                    "classes": list(person.get("classes", [])),
                    "n_face": person.get("n_face", 0),
                    "n_body": person.get("n_body", 0),
                    "n_partial": person.get("n_partial", 0),
                    "has_thumbnail": thumbnail_version > 0,
                    "thumbnail_version": thumbnail_version,
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
