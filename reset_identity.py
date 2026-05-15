"""Clear stored face/body/partial embeddings for people whose identity has
drifted or been poisoned by auto-learning - keeping their name + class
memberships, so only the face needs re-enrolling, not the whole roster.

Run from the repo root with the CameraLM app stopped:

    python reset_identity.py --list                  # show everyone + counts
    python reset_identity.py 8                        # clear pid 8
    python reset_identity.py Ian "Lenny Lenmour"      # clear by name
    python reset_identity.py --all                    # clear everyone's embeddings

A timestamped copy of embeddings.npz is saved under data/backups/ before
anything is cleared. After clearing, relaunch CameraLM and re-enroll each
cleared person by clicking their box in the live view and typing their name.
"""

import shutil
import sys
import time

from cameralm.config import DATA_DIR, EMBEDDINGS_FILE
from cameralm.identity_db import IdentityDB


def _list(db: IdentityDB) -> None:
    print(f"{'pid':>4}  {'name':<22} {'face':>5} {'body':>5} {'partial':>8}  classes")
    print("-" * 72)
    for person in db.snapshot()["people"]:
        print(
            f"{person['pid']:>4}  {person['name']:<22} "
            f"{person['n_face']:>5} {person['n_body']:>5} {person['n_partial']:>8}  "
            f"{', '.join(person['classes'])}"
        )


def _backup():
    if not EMBEDDINGS_FILE.exists():
        return None
    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{EMBEDDINGS_FILE.name}.{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(EMBEDDINGS_FILE, dest)
    return dest


def _resolve_targets(db: IdentityDB, args: list[str]) -> list[int]:
    """Map CLI args (numeric pids or names) to a list of pids. --all => everyone."""
    people = db.snapshot()["people"]
    if "--all" in args:
        return [p["pid"] for p in people]
    by_pid = {p["pid"] for p in people}
    by_name = {p["name"].strip().casefold(): p["pid"] for p in people}
    targets: list[int] = []
    for arg in args:
        if arg.isdigit() and int(arg) in by_pid:
            targets.append(int(arg))
        elif arg.strip().casefold() in by_name:
            targets.append(by_name[arg.strip().casefold()])
        else:
            print(f"  ! no person matches {arg!r} - skipping")
    return targets


def main() -> None:
    args = sys.argv[1:]
    db = IdentityDB()

    if not args or "--list" in args:
        _list(db)
        if not args:
            print("\nPass pids or names to clear, or --all. Nothing cleared.")
        return

    targets = _resolve_targets(db, args)
    if not targets:
        print("No matching people. Nothing cleared.")
        return

    backup = _backup()
    if backup is not None:
        print(f"Backed up embeddings.npz -> {backup}")

    for pid in targets:
        name = db.get_name(pid)
        if db.clear_embeddings(pid):
            print(f"  cleared embeddings for pid {pid} ({name}) - name + classes kept")
    db.save()
    n = len(targets)
    print(
        f"\nDone. Relaunch CameraLM and re-enroll {n} "
        f"{'person' if n == 1 else 'people'} (click the box in the live view, type the name)."
    )


if __name__ == "__main__":
    main()
