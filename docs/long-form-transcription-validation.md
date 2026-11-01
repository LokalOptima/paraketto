# Long-form transcription validation

29 October 2026. This records checks against the implemented personal workflow in
[the plan](long-form-transcription-plan.md). All destructive fault tests used
temporary data directories. Desktop checks use private Xvfb/D-Bus sessions; GPU
checks use owned workers and public speech. The user's service, recordings,
microphone, clipboard and input focus were not used.

## Transcription-first interface and capture pause

30 October 2026. The current interface uses a full Transcription tab and a
separate History tab. Original/Changes views and every playback action have been
removed. Inline dictionary highlights use validated source-to-rendered UTF-8 byte
ranges; hover reveals the original text. A single pill controls all replacements
for the session, including future groups and manual Copy. Schema v4 preserves
the setting across generations/restarts and migrates old mixed decisions to off.
Immutable Stop publications remain unchanged.

The new focused tests cover repeated aliases, Unicode prefixes, variable-length
replacements, disabled changes and invalid mapping fallback. Private real-GTK
checks exercise actual Pango hover coordinates, whole-transcript copy across
unloaded pages, the global pill, selection/highlight preservation during external
updates, and the Transcription/History tab relationship. Recording controls stay
at the same measured position and size through Pause, Resume and Stop; they are
unmapped outside active recording and when the History tab is visible.

Pause stops acquisition, drains and checkpoints accepted PCM, and flushes the
current speech group without sealing the session or publishing. Resume preserves
sample order and excludes paused wall time. Tests cover frozen counters, exact
accepted PCM, paced source continuity, Stop/Cancel while paused, checkpoint/write
failure, and service crash recovery. GTK lifecycle checks include a crash while
paused, reconnect without resuming capture, Resume, and explicit Quit while
paused with no unintended clipboard replay.

Private screenshots are generated under `scratch/desktop-redesign/` and scaled
variants. Current visual checks use synthetic/public fixture text, isolated
Xvfb/D-Bus, and no user display, microphone or clipboard. Older screenshot and
UI descriptions below are historical evidence from before this redesign.

## Architecture and implementation

The service owns capture, a bounded audio writer, CPU Silero planning, persistent
session work and one private Parakeet inference worker. SQLite stores metadata and
immutable revisions; audio remains in independently synchronized PCM files.
Database operations use one serialized connection and short transactions. The
audio callback only queues samples; the writer performs durability checkpoints.
SQLite is still useful without cleanup: it atomically tracks durable audio
coverage, ASR work, dictionary decisions and immutable Stop publications.

S1-mini, its worker, model preparation, queue, controls and validation/diff path
have been removed. The ASR scheduler already serializes its requests, so the
second-model GPU admission/preemption gate was removed too. CPU ggml remains for
Silero. Output is ASR text plus explicit, reversible dictionary replacements.

Stop publications contain enabled dictionary edits at ASR completion. Saved publications are
immutable. A durable delivery epoch prevents retries, reprocessing, cancellation,
failed inference and restart from triggering a delayed automatic clipboard copy.

## Checks after S1 removal

The revised Make and CMake service/desktop builds pass. Focused checks pass for
storage/dictionary/migration, capture faults, VAD, ASR scheduling, all seven
legacy service groups, all eleven session API scenarios, four desktop component
groups, and private desktop pane/lifecycle interactions.

Migration tests seed the previous schema with pending, running, completed, failed
and cancelled model runs, enabled/disabled model edits, a disabled dictionary
change, a publication, saved audio and running ASR. Upgrading retires every model
run, excludes its edits from current text and the decision API, preserves the
prior publication and audio, and resumes ASR. Both v1 and v2 upgrade paths create
backups. Fresh databases contain neither live nor archived model-run tables.

API checks confirm only ASR model/queue state remains, cleanup settings are absent
and removed model lifecycle routes return 404. The pane checks preserved manual
Copy/Copy original, Stop-copy delivery, dictionary reversal, paging, selection,
reconnect and absent cleanup controls. Benchmark syntax/help checks pass after
conversion to ASR-only; no fresh GPU benchmark or 62-minute soak was run for this
removal. The historical measurements below are labelled accordingly.

## Focused checks

| Check | Evidence |
| --- | --- |
| Store, dictionary and edits | Immutable originals/publications; revision fencing; idempotent commits; Unicode/source-span checks; conflicting/non-cascading aliases; protected terms; decision conflicts; failed and cancelled jobs; transactional migration with SQLite backup; schema v3 retires previous model runs/edits while preserving audio, raw text, dictionary decisions and Stop snapshots. |
| Capture faults | 30-second block rotation; bounded file ingestion; cross-block reads; rename-before-metadata failure; recoverable partial-file tail; queue overflow; real file-write failure; unavailable source. Errors retain a readable prefix and do not claim false durability. |
| Session API | Synthetic microphone capture, idempotent start/Stop, dictionary decisions, ranged WAV playback, reprocessing/generation reset, cancellation/tombstones, 181-second import, abrupt service death before any ASR job, missing-model retry, deletion during inference. |
| Legacy service | All existing API lifecycle groups pass: lazy loading, four-request queue, reservations, idle/manual unload, load failures/timeouts, worker crashes, port collision/discovery and shutdown. |
| ASR scheduling fixtures | One worker, serial legacy/session requests, queue bounds, failure/retry and process lifetime. |
| VAD reference | 451 saved probability values match the independently compiled pinned Jarvis implementation within tolerance; maximum observed errors 1.33e-5 (synthetic) and 5.73e-5 (public speech). Fixture regeneration reproduces the reference exactly. |
| Desktop lifecycle | GTK death leaves capture running; reconnect restores it; Stop copies once; cancellation/Quit retain audio; ASR failure preserves capture; external retry never copies; shared-worker/discovery compatibility. |
| Session pane | A real GTK interaction test seeds 60 sections in valid 30-second audio blocks and verifies source order, a 50-section widget bound, older-page navigation after an early section changes, selection/scroll preservation during background edits, disable/re-enable controls, immutable Original text, visible ASR failure and degraded speech-detection states, and Current recording without presenting a hidden window. The pane checks Transcript/Original/Changes, dictionary-only edits and ASR-only preferences; removed cleanup controls are absent. |

Reproduce these with the `test-*` targets documented in [README](../README.md#linux-desktop-dictation).
The Silero ONNX reference was unavailable locally, so this does not claim an ONNX
comparison. The independent CPU reference and pinned model/source identities are
documented under [tests/vadfixtures](../tests/vadfixtures/README.md).

`make test-desktop-pane` runs the pane interactions in private Xvfb/D-Bus against
a CPU-only service fixture. Before S1 removal, visual inspection checked the default transcript, Original and
Changes at the default window size and at 125% text scale, including the resizable
pane, progress, copy controls and history. Local inspection images are
`scratch/desktop-review/cleaned.png`, `original.png`, `changes.png` and their
`-large.png` variants; these are generated scratch artifacts, not release assets.
No user's display, clipboard, microphone or private recording was used.

## Quiet speech and source boundaries

The segmenter uses 0.15 onset / 0.10 continuation probabilities and applies
bounded gain only to the detector input (target RMS 0.01, at most 16×). Saved PCM
and ASR input remain unchanged. This corrected a measured low-volume omission:
with the earlier detector path, ASR lost “Ah” from “Ah, very well.”

The final actual-GPU comparison retained that complete phrase at gains 0.025 and
0.016, after both 3-second and 23-second pauses, matching unfiltered ASR in all
four cases. Inference input shrank from 8.17/28.17 seconds to 2.43–2.69 seconds.
The corresponding CPU checks retain over 99.5% of speech energy. Steady noise,
hum and quiet recorded background remain skipped in the checked cases; one
ambiguous breath-shaped case retains 0.864 seconds out of 12 seconds. Preserving
that uncertainty is preferable to dropping the demonstrated quiet word.

A continuous 113.75-second speech fixture forces a boundary near 88 seconds.
Restart probes around ordinary and forced boundaries preserve the same samples
without duplicate spans. Warmup now ends at the exact resumed sample, including
when the supplied context length is not a multiple of 512. Segmentation policy,
model checksum and actual filtering state are persisted with planned jobs.

## Historical GPU evidence for retained components

These measurements preceded S1 removal and are not fresh performance claims for
the revised build. Hardware: RTX 5070 Ti, 16 GiB, Parakeet Ultra FP16.

- Cold import to saved complete publication took **4,086 ms**, including per-load
  model/runtime verification.
- Warm Stop to saved complete publication measured **p95 201 ms** in one 20-run
  check and **226 ms** in a later 20-run timing sample.
- A 125-second speech recording produced bounded groups and exactly the same
  **212 normalized words** as the existing long-file path. Punctuation differed.
- Raw ASR matched legacy `/transcribe` for an unsplit clip.
- A live 30-second chunk took **37.45 ms** of ASR inference; its publication was
  available **130.33 ms** after Stop. Cold ASR loading took **3.79 seconds**.

These times end at publication, not clipboard delivery. Current ASR-only GPU
harnesses are `tests/bench_sessions.py` and `tests/bench_pipeline.py`; they check
GPU availability and use isolated services/public speech. S1 throughput and
preemption benchmarks are no longer part of this workflow.

One separate private desktop lifecycle fixture observed **112.5 ms** from the
first observed complete publication to the clipboard containing that snapshot,
and **261.3 ms** from the local Stop command to the observed clipboard change.
This used a CPU fixture worker, isolated Xvfb/D-Bus and a private clipboard. These
are polling observations from one run, not a desktop p95 or an addition to the
GPU percentile above. `tests/test_desktop_lifecycle.py` prints the observations.

## Long recording and remaining limits

The real-time 62-minute capture check alternates public speech with long silence,
using actual GStreamer pacing, disk writes, CPU VAD and a fake ASR worker. It
passed with exactly **59,520,000 saved samples**, completed transcription and
**3,136 KiB (3.06 MiB) steady-state RSS variation**. Service RSS stayed around
26 MiB rather than growing with the 119 MB recording. Its report is
`scratch/long-session.json`. The soak began before the final VAD/provenance
refinements; capture/writer code was unchanged, and those refinements have
separate focused and GPU checks. Real ASR throughput is covered separately above.

Process-crash and write-failure tests establish recovery behavior, not immunity
to arbitrary hardware or power failure. The roughly one-second unsynchronized
audio tail remains a durability target. Full raw audio is retained for explicit
reprocessing when VAD or ASR misses speech.

Dictionary replacements are deterministic configured rules, not a guarantee that
ASR recognized the right words. Original text and saved audio remain available
for checking and reprocessing recognition mistakes.
