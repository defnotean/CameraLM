# CameraLM - Follow-ups

This file tracks work deliberately deferred after the large review-and-improve
pass. It was produced from ~30 parallel code reviews + a council session. The
council's rule for that session was **surgical, not structural**: stability,
security, correctness guards, localized performance, and cheap privacy controls
shipped; anything that reshapes the hot loop, the module structure, or the data
model was deferred - and deferred *together*, behind a test net.

Items are ordered roughly by the sequence they should be done in.

---

## 1. BLOCKING before any real classroom / minors deployment - privacy hardening

CameraLM stores biometric data (face + body embeddings, face thumbnails, names)
of people who may be minors. This session added the *cheap* guardrails
(on-screen "RECOGNIZING" indicator, an audit log at `data/audit.log`, a
"delete all data" admin endpoint). The following are **required before the tool
is used to recognize real people, especially minors**:

- **Consent + notice.** A startup consent screen; enrollment OFF by default
  (`ENROLLMENT_ENABLED = False`) so `create_person`/`add_face` only persist after
  an explicit operator action. An `enrolled_with_consent` flag per person.
- **Encryption at rest.** `identities.json`, `embeddings.npz`, and
  `thumbnails/*.jpg` are currently plaintext. Encrypt with an OS-keyring-derived
  key, or at minimum document requiring full-disk encryption.
- **Retention + expiry.** A `RETENTION_DAYS` setting; store `created_at` /
  `last_seen` per person; expire on load. An end-of-term purge flow.
- **`PRIVACY.md`** documenting what is collected, where it is stored, retention,
  deletion, data-subject rights, and the relevant legal frameworks
  (FERPA / COPPA / BIPA / GDPR Art. 9 - *not legal advice; get a real review*).
- **Purpose limitation.** Keep `VLM_PROMPT` to neutral physical description;
  do not expand it to emotion/behavior profiling. Remove any README text that
  encourages that.
- The admin UI should surface retention status and a link to `PRIVACY.md`.

> No legal or ethics specialist reviewed this project. The mechanisms can be
> built; a deployment decision needs a real review.

## 2. Test suite - DONE ✅

A 147-test pytest suite now exists under `tests/` (`conftest.py` + `helpers.py`
+ `test_matcher`, `test_identity_db`, `test_identity_db_persistence`,
`test_partial`, `test_tracking`, `test_vlm`, `test_ui`, `test_naming`,
`test_pipeline`, `test_embedding_store`, `test_config`), with a Windows CI
workflow at `.github/workflows/test.yml`. `matcher.decide_identity`, the
`IdentityDB` / `EmbeddingStore` CRUD/dedup/persistence logic, the
`resolve_track_identity` resolver, the pipeline snapshot isolation, and the
config invariants are covered. The suite stays torch-free (~1.2 s). The original
note below is kept for context.

There were zero tests. `matcher.decide_identity` and the identity /
track-memory state machine are pure-ish, branch-heavy, and the system's core IP
- and were untestable because welded to the camera loop.

First ~10 tests (all webcam-free, synthetic float32 vectors):
1. New track rejects body-only match when `REQUIRE_FACE_FOR_NEW_TRACK`.
2. New track rejects partial-only match.
3. Face above threshold + margin wins over a disagreeing body.
4. Locked track: body match for a *different* pid than `expected_pid` is rejected.
5. "side" fusion needs the configured number of weak hits.
6. `add_face` twice with near-identical vectors keeps `n_face == 1` (dedup).
7. Adding past the cap replaces the most-redundant slot, count stays at cap.
8. `_search_with_margin_locked` margin: 1 pid → 1.0; 2 pids → `best - next`.
9. `save()` → fresh `IdentityDB` → round-trips people/classes/embeddings/counts.
10. `partial.embed` returns shape `(192,)`, L2-norm ≈ 1.0; `None` for tiny crops.

Refactors needed to make this testable: inject `data_dir` into `IdentityDB`
instead of module-level path constants; extract the track-memory FSM (see #3);
let `matcher`/`identity_db` take an overridable config/policy object.

## 3. Architecture refactor - DONE ✅

The test net (#2) exists, so the refactor was unblocked. All five phases
(P1–P5) are complete.

- **P1 - naming FSM → dataclasses. ✅ DONE.** `cameralm/naming.py` defines
  `NameEntry` / `ClassEntry`; `main.py` discriminates them with `isinstance`,
  never a string key. This eliminated the `KeyError: 'tid'` bug *class* (one
  stage had a key the other didn't). `tests/test_naming.py` guards it.
- **P2 - `TrackState` extraction. ✅ DONE.** The per-track identity state is now
  a `TrackState` `@dataclass` in `cameralm/tracking.py`, which also holds the
  extracted resolver (`resolve_track_identity`) + its helpers (`bbox_iou`,
  `cache_ttl_for`, `track_memory_*`, `point_in_bbox`). `main.py`'s hot loop was
  rewired (~35 sites) from dict-access to typed attributes. The resolver - the
  system's core decision logic - now has direct unit tests in
  `tests/test_tracking.py` (it had zero before). `tracking.py` pulls in no
  torch, so those tests run in lightweight CI.
- **P3 - `IdentitySource` StrEnum. ✅ DONE.** `cameralm/types.py` defines the
  `IdentitySource` `StrEnum` (FACE/BODY/SIDE/PARTIAL/TRACK). The stringly-typed
  source literals in `matcher.py`, `tracking.py`, `theme.py`, and `ui.py` are
  replaced with enum members, and `TrackState.source` is typed `IdentitySource
  | None`. (The embedding-*kind* literals - `"face"/"body"/"partial"` - in
  `identity_db.py` are a separate set and fold into P4's `EmbeddingStore` work.)
- **P4 - generic `EmbeddingStore`. ✅ DONE.** `cameralm/embedding_store.py`
  defines `EmbeddingStore(dim, dup_sim, cap)`; `IdentityDB` composes three of
  them (`self.face` / `self.body` / `self.partial`) and the `_*_locked` dispatch
  ladders (`_embedding_store_locked`, `_index_for_kind_locked`,
  `_rebuild_index_locked`, `_install_embeddings_locked`, ...) are gone. Read-only
  compat properties (`face_index`, `face_pids`, `face_embeddings`, ...) keep
  existing callers and the frozen test suite working unchanged - which is what
  proves behaviour was preserved. `tests/test_embedding_store.py` adds 6 direct
  unit tests. Adding a 4th signal (gait/skeleton) is now one `EmbeddingStore(...)`
  line plus an entry in the `_channels` dispatch table.
- **P5 - config invariant checks. ✅ DONE (scoped).** `config.py` ends with
  `_validate_config()`, run at import: ~60 invariant checks (weak < strong
  thresholds, positive intervals/dims, in-range ports/ratios) that fail fast with
  a precise message instead of misbehaving far from the cause.
  `tests/test_config.py` guards it. The full flat-constants → typed-dataclasses
  restructure was *deliberately not done*: it is either ~100 sites of risky
  mechanical churn across ~12 files, or speculative unused abstraction - and this
  item was already marked "lower priority". The invariant-checking value (the
  part the item exemplified with `WEAK_THRESH < MATCH_THRESH`) is delivered
  without that churn.

## 4. Structural performance - DONE ✅ (camera-bound at ~30 fps)

The *localized* perf wins shipped earlier (pose-pass gating, OSNet FP16,
differentiated cache TTLs, the PIL transparent-overlay rewrite). The *structural*
wins then landed:

- ✅ **3-thread pipeline.** `cameralm/pipeline.py` - a capture thread (owns the
  camera) and an inference thread (detect + identity resolution) feed a
  latest-wins slot the main thread renders from. Stages overlap, so the frame
  rate approaches 1/max(stage). Verified ~15.7 → ~30 fps idle; ~13 → ~25 fps with
  a person in frame. `Renderer.finish()` was made non-mutating so the shared
  frame is never written in place; `TrackState` snapshots (`dataclasses.replace`)
  cross the thread boundary.
- ✅ **Detection cadence** - `DETECT_EVERY_N` config knob (reuse boxes between
  YOLO runs; ByteTrack ids stay stable, it is framerate-agnostic). Defaulted to 1
  because measurement showed the inference stage already keeps up with the 30 fps
  MJPG camera - skipping detection buys no fps and only adds box lag. The knob
  remains for heavier future models / a faster camera.
- ✅ **The camera was the hidden floor.** The webcam defaulted to YUY2
  (uncompressed, USB-bandwidth-limited - ~1 fps at 720p, ~16 fps at 480p);
  forcing MJPG in `_open_camera` was the single biggest win and also explains the
  old "black frames at 1280×720".
- Not done - **embedders on a `ThreadPoolExecutor`**: with a person in frame the
  inference stage is ~25 fps; running face/body/partial concurrently could
  recover a few fps. Lower priority now the system is camera-bound at ~30.
- Not done - **TensorRT FP16 export**: would help only if YOLO becomes the
  bottleneck again; it currently is not.

> Note: the VLM (Ollama / Qwen3-VL) was **removed** from the live path
> (`USE_VLM = False`) - on a 4 GB single-GPU box it cannot coexist with the four
> CV models without crippling the framerate. `vlm.py` is kept dormant behind the
> flag for an 8 GB+ GPU. See config.py.

## 5. Honest cross-clothing re-identification

Research finding: single-webcam clothes-changing re-ID *without a face* tops out
around ~50% rank-1 with current methods. OSNet (the current body channel) is
clothing-dependent by design - it is the wrong tool for the stated goal.

- Treat **face as the identity anchor** (already the case) and add a
  **skeleton/gait continuity channel** (pose keypoints → small GCN, or OpenGait
  silhouettes) as the clothes-invariant signal.
- Retire OSNet/partial-histogram as *identity* signals; keep them only as
  short-term tracking tiebreakers.
- Set user expectations: robust re-ID across a full outfit change without ever
  seeing the face is not reliably achievable here.

## 6. Product features (build on a sighting log)

Currently recognition events evaporate frame-to-frame - nothing is recorded.

1. **Sighting log** - append `(pid, class_context, first_seen, last_seen,
   confidence, source)` on track-confirm. Foundational; everything else needs it.
2. **"Who's here right now" + attendance/roll-call** - per-class present/absent,
   daily CSV export. The single highest-value feature for the classroom use case.
3. **Misidentification correction** - a per-track "that's wrong / relabel"
   action in the live view; on correction, purge embeddings learned from the
   wrong track. Without this, errors self-pollute the DB via auto-learning.
4. **Identity merge** - `merge_persons(src, dst)`; the camera double-enrolls
   people constantly and there is no way to combine them today.
5. **Guided multi-angle enrollment** - replace single-frame naming with a 3–5
   shot wizard (front/left/right/back) with a capture-quality meter.
6. **Per-person detail view** in the admin UI - sighting timeline, all
   thumbnails, per-embedding management (delete bad samples).

## 7. Admin UI polish

The admin web UI works but reads as a hobby project:

- Replace native `prompt()` / `confirm()` with inline editing + a styled confirm.
- Keyed/diff DOM updates instead of a full `innerHTML` rebuild every 1.5 s
  (the rebuild drops focus/scroll/drag state and reloads every thumbnail).
- Real connection state (the "connected" dot is currently hardcoded).
- Surface `response.ok` failures with a toast instead of silently reverting.
- Distinct empty / loading / disconnected states.
- Consider migrating off the browser entirely to **pywebview** - a native
  webview window has no network socket, which removes the whole CSRF / DNS-
  rebinding / auth surface this session had to defend against.

## 8. Smaller deferred items

- **Concurrency:** `admin.py` calls `db.save()` on every mutation. The atomic
  save now snapshots under the lock and writes outside it, so contention is much
  reduced - but a debounced "mark dirty, flush on a timer" write would be better.
- **NumPy 2.x:** `requirements.txt` pins `numpy>=1.26,<2.0`. This will get
  harder to satisfy as the ecosystem moves on. The pin exists because
  `insightface==0.7.3` and the boxmot/onnxruntime stack predate NumPy 2.0 and
  have ABI/ API breakage with it. Before unpinning: build a clean venv against
  NumPy 2.x, run the full pipeline (not just the unit tests - the unit tests
  don't touch insightface/boxmot), and bump `insightface`/`onnxruntime-gpu` to
  versions with confirmed NumPy 2 support. Do not unpin blindly - it will break
  the install.
- **Dead code - partly done.** Removed: `theme.FONT_BOLD`, and
  `draw_naming_modal`'s never-passed `hint_label` param. *Kept, with reason:* the
  `not REQUIRE_FACE_FOR_NEW_TRACK` branches in `matcher.py` - `test_matcher.py`
  has four `if not config.REQUIRE_FACE_FOR_NEW_TRACK: pytest.skip()` guards, i.e.
  the suite treats it as a live, flippable policy knob, not accidental dead code;
  `search_face` - its symmetric twin `search_body` *is* used by `test_matcher.py`,
  so dropping only `search_face` would leave a lopsided API; `camera_probe.py` -
  rewritten this pass into a real diagnostic (it is what caught the YUY2 camera
  bug). This original note predates those tests.
- **`partial.py`:** mask out background pixels (axis-aligned bbox for an
  arm/head is mostly wall), add colour-constancy normalization, drop the
  redundant overlapping regions - or replace it with a cheap CNN stem.
- **VLM - mostly moot.** `USE_VLM` is now `False` (see §4 - it can't share the
  4 GB GPU with the CV stack). The pre-resize idea *was* implemented
  (`VLM_IMAGE_MAX_PX`, in `vlm.py`) before the feature was disabled;
  pseudo-batching is moot while the VLM is off.
- **Structured stats line - ✅ DONE.** The pipeline logs three periodic INFO
  lines every `STATS_LOG_INTERVAL_SECONDS` - `capture:` (camera read fps),
  `inference:` (fps + detect/resolve ms), `display:` (fps + render/show ms).
  This is what turned the framerate investigation from guesswork into
  measurement.
