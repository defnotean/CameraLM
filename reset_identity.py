"""Offline privacy + identity tools for the CameraLM data directory.

Run from the repo root with the CameraLM app stopped. Subcommands:

    python reset_identity.py --list                   # show everyone + counts
    python reset_identity.py 8                         # clear pid 8 (keep name + classes)
    python reset_identity.py Ian "Lenny Lenmour"       # clear by name
    python reset_identity.py --all                     # clear everyone's embeddings

    python reset_identity.py --audit [N]               # dump the audit log (default last 500 lines)
    python reset_identity.py --purge-stale DAYS        # run retention sweep with the given cutoff
    python reset_identity.py --consent-report          # report consent status per person

A timestamped copy of embeddings.npz is saved under data/backups/ before any
clearing or purge. After clearing, relaunch CameraLM and re-enroll each cleared
person by clicking their box in the live view and typing their name.
"""

import shutil
import sys
import time

from cameralm.config import DATA_DIR, EMBEDDINGS_FILE
from cameralm.identity_db import IdentityDB


def _list(db: IdentityDB) -> None:
    snap = db.snapshot()
    print(f"{'pid':>4}  {'name':<22} {'face':>5} {'body':>5} {'partial':>8}  {'consent':<10} classes")
    print("-" * 90)
    for person in snap["people"]:
        consent = (person.get("consent") or {}).get("status", "none")
        print(
            f"{person['pid']:>4}  {person['name']:<22} "
            f"{person['n_face']:>5} {person['n_body']:>5} {person['n_partial']:>8}  "
            f"{consent:<10} {', '.join(person['classes'])}"
        )


def _consent_report(db: IdentityDB) -> None:
    """Show consent status + last-seen for every person. Useful for compliance
    review without spinning up the admin UI."""
    snap = db.snapshot()
    rows = snap["people"]
    print(f"{'pid':>4}  {'name':<22} {'status':<10} {'granted_at':<20} {'by':<20} last_seen")
    print("-" * 100)
    for p in rows:
        consent = p.get("consent") or {}
        print(
            f"{p['pid']:>4}  {p['name']:<22} "
            f"{consent.get('status','none'):<10} "
            f"{consent.get('granted_at','') or '-':<20} "
            f"{(consent.get('granted_by','') or '-')[:20]:<20} "
            f"{p.get('last_seen_at','') or '-'}"
        )
    counts = {"none": 0, "granted": 0, "revoked": 0}
    for p in rows:
        s = (p.get("consent") or {}).get("status", "none")
        counts[s if s in counts else "none"] += 1
    print(
        f"\nTotal: {len(rows)} people. "
        f"granted={counts['granted']} none={counts['none']} revoked={counts['revoked']}."
    )


def _audit(db: IdentityDB, limit: int) -> None:
    rows = db.read_audit(limit=limit)
    if not rows:
        print("(audit log empty)")
        return
    for row in rows:
        print(f"{row['ts']}  {row['action']:<22} {row['detail']}")
    print(f"\n({len(rows)} entries shown)")


def _purge_stale(db: IdentityDB, days: int) -> None:
    if days <= 0:
        print("Refusing to purge with retention_days <= 0 (pass a positive day count).")
        return
    backup = _backup()
    if backup is not None:
        print(f"Backed up embeddings.npz -> {backup}")
    purged = db.purge_stale(retention_days=days)
    if not purged:
        print(f"Nothing past the {days}-day window. No vectors purged.")
        return
    print(f"Dropped embeddings for {len(purged)} people:")
    for row in purged:
        print(
            f"  pid={row['pid']:>4}  {row['name']:<22}  "
            f"last_seen={row['last_seen_at']}  age={row['age_days']}d"
        )
    db.save()


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
        if arg.startswith("--"):
            continue
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

    # --- privacy subcommands (no clearing happens here) ---
    if "--consent-report" in args:
        _consent_report(db)
        return

    if "--audit" in args:
        idx = args.index("--audit")
        limit = 500
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            limit = int(args[idx + 1])
        _audit(db, limit=limit)
        return

    if "--purge-stale" in args:
        idx = args.index("--purge-stale")
        if idx + 1 >= len(args) or not args[idx + 1].isdigit():
            print("Usage: --purge-stale DAYS  (DAYS = positive integer)")
            return
        _purge_stale(db, days=int(args[idx + 1]))
        return

    # --- legacy: clear embeddings for one or more pids/names ---
    if not args or "--list" in args:
        _list(db)
        if not args:
            print("\nPass pids or names to clear, or --all. Nothing cleared.")
            print("Run with --help for the full list of subcommands.")
        return

    if "--help" in args or "-h" in args:
        print(__doc__)
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
