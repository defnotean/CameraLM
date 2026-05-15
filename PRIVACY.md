# Privacy notes for CameraLM

CameraLM captures and stores biometric data: face embeddings, body
embeddings, clothing-color signatures, and small face thumbnails of every
person you name. This document describes what the system does about that, what
it does not, and what you, the operator, must still do yourself.

This is not legal advice and not a compliance program. It is a map of the
mechanisms the codebase actually has, so you can decide whether they meet your
local rules.

---

## 1. What CameraLM stores

Everything is on the machine running CameraLM, under `data/`. Nothing is sent
anywhere off-host.

| File | Contents |
| ---- | -------- |
| `data/embeddings.npz` | Face / body / partial vectors, the pid lists, and metadata (names, classes, schedule, consent records, timestamps). One atomic file. |
| `data/identities.json` | Human-readable mirror of the metadata. Not load-bearing - the npz is the source of truth. |
| `data/thumbnails/<pid>.jpg` | One small face thumbnail per enrolled person. |
| `data/audit.log` | Append-only log of every identity-database mutation. |
| `data/backups/` | Pre-clear / pre-purge backups created by `reset_identity.py` and the corrupt-file recovery path. |

`data/` is in `.gitignore` and must not be committed. The admin server is bound
to `127.0.0.1` (loopback only) and protected by a Host-header allowlist and a
SameSite=Strict session cookie.

---

## 2. Per-person record

Every person you enroll is stored as a record like this:

```
pid:          7
name:         "Ada Lovelace"
classes:      ["Math 10"]
n_face / n_body / n_partial: counts of stored vectors
consent:
  status:     "none" | "granted" | "revoked"
  granted_at: "2026-05-15T14:32:11"  (ISO local time, empty if not granted)
  granted_by: "Ms. Smith"             (operator who attested consent)
  notes:      "verbal consent at homeroom"  (free text, <= 500 chars)
last_seen_at: "2026-05-15T15:01:44"   (ISO time of last recognized sighting)
created_at:   "2026-05-15T14:30:02"   (ISO time of first enrollment)
```

Older records that pre-date this schema are backfilled with safe defaults on
load: `consent.status = "none"`, `last_seen_at = ""`, `created_at = ""`.

---

## 3. Consent

CameraLM records consent as metadata. The operator records consent
affirmatively, with their name and optional notes; the record is timestamped
and written to the audit log.

### 3.1 Recording consent

Open the admin UI at `http://localhost:8765`, find the person, click the
**shield** icon on their card. You will be prompted for:

1. **Attestor** (required): your name or role. The system rejects empty values.
2. **Notes** (optional): how / when / where consent was obtained.

The card's consent pill changes to green and shows the date + attestor.

### 3.2 Revoking consent

Click the shield icon on a consented person. Confirm. CameraLM will:

1. Drop every stored face / body / partial vector for that person immediately.
2. Set `consent.status` to `"revoked"`.
3. Keep the name and class memberships so the audit trail still references a
   real entity.

To fully remove the person (name, classes, thumbnail), use the trash icon
afterwards.

### 3.3 The "REQUIRE_CONSENT_FOR_RECOGNITION" gate

By default the live overlay shows everyone the pipeline recognizes, regardless
of consent status. Set `REQUIRE_CONSENT_FOR_RECOGNITION = True` in
`cameralm/config.py` to suppress the live name / class chips for anyone whose
consent is not granted. They will display as `Unknown` until you record consent
in the admin UI.

This is a display gate, not a processing gate: stored vectors still drive the
match. The intent is that recognition only **surfaces** for people who have
opted in. If you need to also stop **storing** vectors for unconsented
subjects, see section 6.

---

## 4. Retention

Two configuration knobs in `cameralm/config.py`:

* `RETENTION_DAYS = 0` (default disabled). Number of days a person can go
  unseen before their biometric vectors are dropped on the next sweep. Their
  name and class memberships are preserved, so the audit log still resolves
  the pid to a real name; only the data that can identify them is purged.
* `RETENTION_PURGE_ON_STARTUP = True`. Run a sweep when CameraLM starts.
  No-op when `RETENTION_DAYS = 0`.

### 4.1 What the sweep does

For every person, the sweep looks at `last_seen_at` (or `created_at` for
people who have never been recognized). If that is older than
`RETENTION_DAYS`, every stored face / body / partial vector is dropped. People
with neither field (predating this feature) are left alone - the sweep cannot
prove they are stale.

### 4.2 Running the sweep manually

From the admin UI: open the Privacy panel and click "Run now."

From the CLI:

```powershell
.\.venv\Scripts\python.exe reset_identity.py --purge-stale 90
```

Both paths back up `embeddings.npz` to `data/backups/` before any drop and
write a line to the audit log for each person cleared.

---

## 5. Audit log

Every mutating operation appends one tab-separated line to `data/audit.log`:

```
2026-05-15T14:32:11    create_person      pid=7 name='Ada Lovelace'
2026-05-15T14:33:02    record_consent     pid=7 by='Ms. Smith' notes='verbal'
2026-05-15T19:05:10    revoke_consent     pid=7
2026-05-15T19:05:10    purge_stale        pid=3 last_seen=2025-12-01... age_days=165
```

Actions covered: `create_person`, `rename_person`, `delete_person`,
`clear_embeddings`, `delete_all_data`, `record_consent`, `revoke_consent`,
`purge_stale`, `create_class`, `delete_class`, `add_to_class`,
`remove_from_class`, `set_schedule_slot`.

To read the audit log:

* Admin UI: open the Privacy panel; the most recent 200 entries render at the
  bottom and refresh every 30 seconds. Use **Reload** to refresh sooner.
* CLI: `python reset_identity.py --audit 1000` prints the last 1000 entries.
* Or just open `data/audit.log` in any text editor.

The file is append-only. Rotate it manually if it grows large.

---

## 6. What is **not** built in

These are real privacy obligations that CameraLM does **not** solve for you.

### 6.1 Encryption at rest

`data/` is plaintext on disk. Anyone with read access to the user's filesystem
can read the embeddings, thumbnails, and audit log. If your situation requires
encryption at rest:

* Encrypt the whole disk (BitLocker on Windows, FileVault on macOS, LUKS on
  Linux) and treat the OS login as the auth boundary.
* Or wrap `data/` in a per-user encrypted volume.

A future version may integrate Windows DPAPI for transparent per-user
encryption of the npz - see `FOLLOWUPS.md`.

### 6.2 Camera-side consent gate

Click-to-enroll in the live view does NOT prompt for consent at the moment of
enrollment. The biometric data is captured first; you record consent in the
admin UI afterwards. If your policy requires consent BEFORE biometric capture:

* Train your operators to not click-enroll without prior consent on file.
* Do not deploy in spaces where bystanders may be captured without consent.
* Consider the camera placement and field of view as part of the data flow.

### 6.3 Storage suppression for unconsented people

Setting `REQUIRE_CONSENT_FOR_RECOGNITION = True` hides recognition results for
unconsented people, but their vectors are still stored. To make the storage
itself consent-gated, you would need to either:

* Discard the enrollment until consent is recorded (lose the ability to
  pre-enroll), or
* Delete the unconsented record before the autosave (run the retention sweep
  with `RETENTION_DAYS = 1`).

### 6.4 Compliance certifications

CameraLM is not certified or audited against FERPA, BIPA, COPPA, GDPR, HIPAA,
SOC 2, or any other framework. The local-only design, the consent metadata,
the audit log, and the retention sweep are pieces you can compose into a
compliance posture; CameraLM does not assert compliance on your behalf.

---

## 7. Subject rights checklist

When a subject (or guardian) asks you to act on their data:

| Request | How to handle |
| ------- | ------------- |
| Tell me what you have on me | Open the admin UI; show their record (consent state, classes, n_face / n_body / n_partial counts, last_seen, created_at). The thumbnail is `data/thumbnails/<pid>.jpg`. |
| Delete everything | Admin UI: trash icon on the person card. Or CLI: `python reset_identity.py <pid>` (keeps name) followed by deleting via UI; or `delete_all_data` for a full wipe. |
| Stop recognizing me but keep the record | Revoke consent (shield icon on the card). Drops the vectors immediately; keeps the name. |
| Show me what you have changed | The audit log. Filter by their pid to see every action affecting them. |
| Export everything | The relevant rows of `data/audit.log` plus their entry in `data/identities.json`. No automated export exists yet. |

Every one of these actions is itself logged.

---

## 8. Hardening checklist before a real deployment

1. Set `RETENTION_DAYS` to something sane (90, 180, 365 - depending on your
   policy). Confirm `RETENTION_PURGE_ON_STARTUP = True`.
2. Set `REQUIRE_CONSENT_FOR_RECOGNITION = True` so the live view does not
   surface unconsented people.
3. Encrypt the host filesystem (see 6.1).
4. Decide on an audit log rotation schedule.
5. Document the data flow, the lawful basis, the retention period, and the
   subject-rights contact for your local rules (FERPA, BIPA, COPPA, GDPR, etc.).
6. Decide who has filesystem access to `data/` and the admin UI URL. The admin
   UI is loopback-only by default; do not change that without understanding the
   threat model.
7. Run `python reset_identity.py --consent-report` after every enrollment
   session to confirm no consent gaps remain.

---

## 9. Reporting a privacy bug

If you find a way to make CameraLM leak biometric data, bypass the consent
gate, or evade the audit log, please open an issue at the GitHub repo with
"PRIVACY" in the title.
