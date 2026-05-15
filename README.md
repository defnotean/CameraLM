# CameraLM

Local-first real-time webcam person recognition with classroom-style organization. Everything runs on your machine: face and body recognition, identity storage, the admin web UI. No cloud, no API keys, no network calls.

The recognition pipeline (YOLOv8 detection, ByteTrack ID tracking, InsightFace ArcFace for faces, OSNet for bodies, FAISS for matching) feeds a small Flask admin server that you open at `http://localhost:8765` to manage who is who, group people into classes, and lay out a rotating block schedule.

---

## Important: this is a biometric system

CameraLM stores face embeddings, body embeddings, clothing-color signatures, and face thumbnails of every person you name. That is biometric data. Some of it is regulated under FERPA (US students), COPPA (US under-13s), BIPA (Illinois), GDPR Article 9 (EU), and other laws. **Do not deploy this against real people until you have the consent and legal review your situation needs.** This project is provided as a working reference; the privacy obligations are yours.

What ships in the box:

* All identity data lives under `data/`, which is excluded from git by default. Do not commit it.
* The admin server binds to loopback only (`127.0.0.1:8765`). It is not reachable from other devices on the network.
* The admin UI has a Host-header allowlist plus a SameSite=Strict session cookie as CSRF / DNS-rebinding defense.
* Every identity mutation is recorded in `data/audit.log`.

What is **not** built in (and is on you):

* Encryption at rest.
* A consent / enrollment-gated flow.
* A retention or end-of-term purge policy.
* Anything resembling formal compliance.

If any of that matters to you, read `FOLLOWUPS.md` section 1 before you turn this on for a real classroom.

---

## What it can do

* Detect and track people from a webcam in real time.
* Recognize a person across sessions using face, body, and a partial-appearance signature.
* Click a bounding box and type a name to enroll someone.
* Auto-improve a person's stored identity over time, but only from high-confidence face matches (so a false match cannot poison the database).
* Group people into named classes ("Math 10", "Grade 8A", "Staff").
* Lay out a rotating block schedule (6 days, blocks A through D, configurable).
* Show a live "Now" pointer that maps the current Day and Period to a Block and the class scheduled for it.
* Filter the identity list to just the people enrolled in the currently-scheduled class.
* Run on a 4 GB GPU at around 25-30 fps with one person in frame, with per-frame work capped so the framerate does not fall as more people enter.

---

## Hardware and software requirements

* Windows 10 or 11. (Linux and macOS may work but are untested.)
* Python 3.11 (exact). Check with `py -3.11 --version`.
* An NVIDIA GPU with current drivers is strongly recommended. Tested on a GTX 1650 with 4 GB. CUDA 12.x runtime is bundled with the PyTorch wheels installed by `setup.ps1`.
* A working webcam.
* About 3 GB of disk for the virtual environment plus model weights.

The optional Vision-Language Model (Qwen3-VL via Ollama) is off by default because it does not fit alongside the four CV models on a 4 GB GPU. See the "Vision-Language Model" section below if you have 8 GB or more of VRAM.

---

## Setup

From the repository root, in PowerShell:

```powershell
.\setup.ps1
```

`setup.ps1` is safe to re-run. It will:

1. Create `.venv\` if it does not exist.
2. Install PyTorch 2.5.1 with CUDA 12.1 wheels.
3. Install everything in `requirements.txt`.
4. Print a verification line for each package.

If `py -3.11` is not found, install Python 3.11 from python.org or the Microsoft Store and re-run.

---

## Running

```powershell
.\start.bat
```

`start.bat` launches the app in its own console window and opens `http://localhost:8765` in your default browser. The camera window appears once the models finish loading (a few seconds, longer on the first run because model weights are downloaded).

Or, manually:

```powershell
.\.venv\Scripts\python.exe -m cameralm.main
```

To stop: press `q` in the camera window, or close it.

---

## Enrolling a person

1. Stand in front of the webcam. An "Unknown" box should appear around you.
2. Click the box.
3. Type your name in the modal that appears.
4. Press Enter.
5. (Optional) Type the name of one or more classes to add yourself to.

That captures one clean enrollment. From there, the system will auto-learn additional views of you, but only from frames where the face match is well above the display threshold (so a wrong face cannot get permanently enrolled into your identity). See the "How auto-learning works" section.

---

## The admin UI

`http://localhost:8765`. The page has five sections, top to bottom:

1. **Top bar.** Brand + four live counters: People, Classes, Unassigned, Images.

2. **Toolbar.** Search box (filters by name or class), and a "Create class" form.

3. **Now bar.** Pick a Day and a Period. The bar shows you the Block in that slot (using the rotation) and the Class scheduled for that Block, plus an enrollment count. A toggle on the right narrows the identities panel to only that class.

4. **Schedule panel** (collapsible). A grid: 6 days across, 4 periods down. Each cell shows the block letter for that day-period pair (the rotation) and a dropdown to assign a class to that block. Changing any "A" cell changes Block A everywhere, because the schedule is per-block.

5. **Identities by class.** A list of collapsible folders, one per class, plus an Unassigned folder at the bottom. Click a folder to expand it. Drag a person card onto a folder header to add them to that class. Each card has rename / add-class / delete buttons and shows the class chips for that person.

What survives the 1.5-second polling refresh: open/closed folder state, open `<select>` dropdowns (the grid only re-renders when its data changes), Day / Period / filter selections (saved to `localStorage`).

---

## How the schedule rotation works

This implements the standard rotating block schedule that many schools use. Same class meets every day under the same block, but the order of blocks in the day rotates.

For 4 blocks across 6 days the rotation is:

| Period | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 |
| ------ | ----- | ----- | ----- | ----- | ----- | ----- |
| 1      | A     | D     | C     | B     | A     | D     |
| 2      | B     | A     | D     | C     | B     | A     |
| 3      | C     | B     | A     | D     | C     | B     |
| 4      | D     | C     | B     | A     | D     | C     |

This falls out of one helper: each day's order is the canonical `[A, B, C, D]` right-rotated by `(day - 1) mod 4`. Day 5 wraps back to the Day 1 order. The number of days and the block letters are both configurable in `config.py`; the helper handles any cycle length.

The schedule itself is per-block. Picking "Math 10" in any cell labelled Block A sets Block A's class everywhere, on every day. The grid is purely a *display* of the rotation.

---

## How auto-learning works

When the system sees a confirmed identity, it can add the current frame's face / body / partial vectors to that person's stored identity so it learns new angles over time. The trigger has a deliberately narrow gate:

* The current frame's identity must come from a **face** match. Body, partial-appearance, side-fusion, and track-memory matches never enroll.
* The face match similarity must be at or above `FACE_LEARN_THRESH` (default 0.62), which is well above `FACE_MATCH_THRESH` (0.52). "Confident enough to display" is intentionally separate from "confident enough to permanently learn".
* The person ID must still exist in the database (it could have been deleted via the admin UI between the search and now).

When that gate passes, the face / body / partial vectors from the same frame are all added. Body and partial only ride along on a confirmed face, never on their own.

The capacity per person is bounded: 20 face vectors, 20 body vectors, 12 clothing-color signatures. When at capacity, a new vector replaces the *most redundant* existing one (the one most similar to the rest), so the stored set stays diverse instead of becoming 20 near-identical frontal shots.

If you suspect a person's identity has drifted (false matches got learned into them), run:

```powershell
.\.venv\Scripts\python.exe reset_identity.py --list             # show stored counts per person
.\.venv\Scripts\python.exe reset_identity.py Ian                # clear by name
.\.venv\Scripts\python.exe reset_identity.py --all              # clear everyone's embeddings
```

`reset_identity.py` writes a timestamped backup of `data/embeddings.npz` to `data/backups/` before clearing anything. Names and class memberships survive; only the embeddings go. Re-enroll afterwards by clicking each person's box and re-typing the name.

---

## How the recognition pipeline works (short version)

Three threads, two latest-wins handoffs:

```
camera ----> [capture thread] ----> latest raw frame
                                            |
                                            v
                                 [inference thread]
                                  YOLO + ByteTrack
                                  identity resolve
                                            |
                                            v
                            latest (frame, frame_info)
                                            |
                                            v
                                  [main thread]
                              render overlay + show
                              handle mouse / keys
                              run the admin server
```

The inference thread runs the three embedders (face, body, partial) per stale track, fuses them with `decide_identity()`, and updates the per-track state. With a per-frame cap (`MAX_RESOLVES_PER_FRAME`, default 1) the per-frame cost stays bounded no matter how many people are on screen. Tracks past the cap keep their previous identity and resolve on a following frame, longest-waiting first, so nothing starves.

For more on individual modules, see "Project layout" below.

---

## Configuration

Every tunable lives in `cameralm/config.py`. The file ends with `_validate_config()`, which runs at import and rejects bad combinations (for example: a weak threshold above its strong threshold) with a precise error message.

Knobs you are most likely to touch:

| Constant | Default | What it controls |
| -------- | ------- | ---------------- |
| `CAMERA_INDEX` | 0 | Which webcam OpenCV opens |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | 1280, 720 | Target capture resolution. Falls back automatically if the camera cannot deliver it. |
| `YOLO_IMGSZ` | 512 | YOLO inference size. Smaller is faster with a small accuracy cost. |
| `FACE_MATCH_THRESH` | 0.52 | Cosine similarity needed to display a face match. Raise if you see false positives, lower if real matches are being missed. |
| `FACE_MATCH_MARGIN` | 0.08 | Match must beat the runner-up person by this much. |
| `FACE_LEARN_THRESH` | 0.62 | Cosine similarity needed to permanently enroll a new face. Higher than the display threshold on purpose. |
| `MAX_RESOLVES_PER_FRAME` | 1 | Cap on per-frame embedder work. With 1, framerate does not depend on how many people are on screen. |
| `SCHEDULE_DAYS` | 6 | Days in the rotating block schedule |
| `SCHEDULE_BLOCKS` | `("A", "B", "C", "D")` | Block names |
| `USE_VLM` | False | Enable the optional Vision-Language Model (see below) |

---

## Vision-Language Model (optional, off by default)

There is plumbing for Qwen3-VL via Ollama that generates short descriptions of each person ("dark hoodie, holding a coffee cup"). It is disabled by default because qwen3-vl:2b is about 3 GB and does not fit on a 4 GB GPU alongside YOLO, InsightFace, OSNet, and partial-appearance.

To enable on an 8 GB+ GPU:

1. Install Ollama from <https://ollama.com/download>.
2. `ollama pull qwen3-vl:2b`
3. In `cameralm/config.py`, set `USE_VLM = True` and (optionally) `VLM_USE_GPU = True`.

The descriptions appear under each person's name in the live overlay, refreshed every few seconds. The worker is asynchronous and drops crops under load, so even a slow VLM cannot stall the camera loop.

---

## Tools

* `start.bat` - launches the app and opens the admin UI in your default browser.
* `reset_identity.py` - clears a person's stored embeddings while keeping their name and classes. Used to undo auto-learning drift. Backs up `embeddings.npz` first.
* `camera_probe.py` - diagnostic that measures raw camera throughput at YUY2 versus MJPG. Useful if FPS feels wrong before the recognition pipeline is in the picture.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

There are 165 tests covering: the identity database (CRUD, dedup, cap behaviour, class membership, embedding store internals), the matcher's identity decision logic, the track-state resolver, the pipeline's snapshot isolation and per-frame resolve cap, the schedule (per-block model, delete-class sweep, persistence, legacy migration), and the config invariants.

The suite is torch-free and runs in about 1.5 seconds. CI: `.github/workflows/test.yml` runs it on every push on Windows.

---

## Project layout

```
cameralm/
  main.py             Entry point. The display loop: render, imshow, waitKey, key handling.
  pipeline.py         PipelineWorker: capture thread + inference thread, latest-wins handoff.
  detector.py         YOLOv8n + a pose fallback for partial-person detections.
  face.py             InsightFace ArcFace (face embeddings, GPU via onnxruntime).
  body.py             OSNet body re-identification, via boxmot.
  partial.py          HSV + grayscale histogram signature for the visible region.
  tracking.py         TrackState dataclass and resolve_track_identity (the per-track engine).
  matcher.py          decide_identity: combines face / body / partial into one verdict.
  identity_db.py      People, classes, schedule. FAISS-backed, atomic single-file save.
  embedding_store.py  One channel (face / body / partial). IdentityDB composes three.
  ui.py               PIL overlay rendering: boxes, labels, modals, HUD.
  theme.py            Color palette and font paths.
  admin.py            Flask admin server. Loopback-only, with CSRF / DNS-rebinding defense.
  templates/admin.html The admin UI: class folders, schedule grid, Now bar.
  config.py           Every tunable plus _validate_config().
  vlm.py              Optional Qwen3-VL integration (off by default).
  naming.py           Typed dataclasses for the two-stage "name an unknown" flow.
  types.py            IdentitySource enum.

tests/                165 tests, torch-free.

start.bat             Launch the app and open the admin UI.
setup.ps1             Install dependencies.
reset_identity.py     Clear stored embeddings (keep names and classes).
camera_probe.py       Diagnose camera throughput.
```

---

## Known limitations

* **Clothes-change re-identification is hard.** Body (OSNet) and partial (HSV histogram) signatures are clothing-dependent. The face is the only signal that survives a full outfit change. Auto-learning is configured to honor that: only a confirmed face can permanently enroll new body / partial vectors.
* **No "this is not me" button in the live view.** If you spot a misidentification while watching the camera window, you cannot correct it inline. Workaround: use the admin UI to delete the person and re-enroll, or run `reset_identity.py` to clear and re-enroll cleanly.
* **The admin UI polls every 1.5 s.** Open dropdowns and folder state survive the refresh, but very fast typing in `prompt()` modals while the page is polling can feel sluggish.
* **No batching of embedder calls.** Each tracked person costs one face + one body + one partial inference. The per-frame cap (`MAX_RESOLVES_PER_FRAME`) keeps the framerate flat regardless of crowd size, but does not make a crowd resolve faster.
* **Numpy is pinned below 2.0.** `insightface 0.7.3` and `onnxruntime-gpu 1.20.1` predate Numpy 2.0. Unpinning will break the install until those packages catch up.

---

## License and disclaimer

Provided as-is for personal and educational use. The author makes no warranty and accepts no liability for any use, including unlawful surveillance or violations of biometric data law.

Before using CameraLM with real people, you are responsible for:

* Obtaining informed consent from every person whose data is captured.
* Complying with applicable laws in your jurisdiction. This README is not legal advice.
* Securing the `data/` directory. By default it is in plaintext on disk; encryption at rest is not built in.
* Documenting and honoring a retention and deletion policy.

This README, the audit log, and the `FOLLOWUPS.md` privacy section are starting points, not a compliance program.
