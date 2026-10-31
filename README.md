# Paraketto

Minimal dictation for macOS. Tap Right Option, speak, tap it again: the text is
pasted into whatever app has focus. Choose English-only
[Parakeet v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) or multilingual
[Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra), running on the Apple
Neural Engine with Core ML. Audio and transcripts stay on the Mac.

- Menu bar only, no Dock icon
- Shortcut: tap `Right ⌥` to start and stop a recording, `Esc` discards it. A tap is a press and release under 0.6 s with no other key in between, so Option+letter typing is unaffected. Change it in Settings (`⌘,` from the menu) by tapping another modifier key or pressing any key combination
- Transcription runs while you speak: every pause of 0.8 s ends a segment and it is transcribed right away, so when you pause before stopping the text is already there
- Floating pill while recording: a level meter on a -65 to -25 dBFS scale with 30 ms attack / 160 ms release smoothing; brief peaks are retained between display updates
- Text is pasted only if the app that had focus when you started still has it; otherwise it is kept in history
- **Transcripts** (right-click the menu-bar icon, or choose Transcripts in its menu): recordings grouped by day, with playback, copy, delete, and word count / WPM stats. Entries show up to four lines; hover to reveal actions and expand longer text
- Drop an audio file onto Transcripts to transcribe it locally, including WAV, M4A and AAC files supported by macOS
- **eng / multi** buttons in the menu: English v2 or Ultra's 25 European languages with automatic detection. Both include punctuation and capitalization
- Models download on first use (about 464 MB for `eng`, 613 MB for `multi`); existing compatible FluidAudio caches are reused

The transcript also stays on the clipboard after pasting. The previous clipboard contents are not
restored, since a timed restore races slow readers.

## Recognition modes

Choose **eng** or **multi** using the segmented control in the menu. `multi` is the default.
A successful selection is remembered across launches. Switching is disabled while recording,
transcribing, or preparing a model. The menu shows a byte-based download progress bar,
then **Optimizing ENG…** or **Optimizing MULTI…** with a spinner for initial preparation.
Later starts show **Loading** and the mode name, followed by a green Ready label.
If switching fails, the previous loaded mode remains available;
use **Retry Mode Change** to try the failed selection again.

Each mode has a separate cache under `~/Library/Application Support/Paraketto/models/`:

| Mode | Recognizer | Cache folder |
|------|------------|--------------|
| `eng` | Parakeet TDT 0.6B v2, English only | `parakeet-v2` |
| `multi` | Parakeet Ultra, 25 European languages | `parakeet-ultra` |

The command-line runner uses the same modes:

```bash
.build/release/paraketto-cli --model eng --vad .build/VadHead.mlmodelc recording.wav
.build/release/paraketto-cli --model multi --vad .build/VadHead.mlmodelc recording.wav
```

The CLI defaults to `multi`. Ultra's optional optimized encoder and fused decoder are used
only for `multi`; `eng` always uses its own v2 assets. Downloads are installed only after all
required files arrive, so an interrupted first download cannot leave a seemingly ready model.

## Build and run

Requires macOS 14+, Apple Silicon, and Xcode 16 or later.

```bash
make run
```

This builds with SwiftPM, compiles the VAD model, assembles `Paraketto.app`, signs it,
prepares downloaded recognition modes for this Mac, updates `/Applications/Paraketto.app`,
and launches that installed copy. `make bundle` also updates the Applications copy on every
successful build, ready to open or compress in Finder for sharing.

Core ML's portable `.mlmodelc` files still need device-specific specialization when first
loaded. `make bundle` performs that load and warm-up through the signed app executable,
without opening windows, accessing the microphone, registering hotkeys, or changing the
selected mode. The first build can therefore spend tens of seconds preparing a model;
subsequent builds normally reuse Core ML's disk cache. Missing models are skipped.
Successful preparation is remembered per mode, model files, and macOS version, including
preparation during builds. This is a UI hint, not a query of Core ML's private cache;
a system cache purge can still make a later **Loading** take longer.

- `make prepare-models` downloads and prepares both `eng` and `multi` explicitly.
- `make bundle PREPARE_MODELS=0` skips preparation, useful when packaging for another Mac.
- `make bundle INSTALL_APP=0` keeps the build in the checkout without installing it.
- `INSTALL_DIR=~/Applications` installs to your personal Applications folder instead.

Quit and reopen the app after a rebuild to run the new version. `make clean` removes local
build outputs; it does not uninstall the Applications copy or remove models and transcripts.

To share a build, compress `/Applications/Paraketto.app` in Finder and send the ZIP. The recipient
needs Apple Silicon and macOS 14+, grants Microphone and Accessibility access, and downloads their
own model on first use. The bundle contains no personal recordings or transcripts. Local development
signing is not Developer ID signing or Apple notarization, so another Mac may require an
[Open Anyway exception](https://support.apple.com/en-us/102445).

Preparation is local to this Mac, model paths and compute configuration. It is not a portable
cache bundled for other machines; macOS can also invalidate it after an OS update or purge it
to reclaim space. A first load may need preparation again in those cases.

## Permissions

- **Microphone**: prompted on the first recording.
- **Accessibility**: needed to send the Cmd+V keystroke. Prompted at launch, granted in
  System Settings → Privacy & Security → Accessibility. Without it the transcript is only
  copied to the clipboard.

### Accessibility and rebuilds

macOS ties the Accessibility grant to the app's code signature. With ad-hoc signing every
rebuild is a different signature, so the toggle in System Settings looks on but the new build
is not trusted and text is only copied. Two ways out:

- **Per build:** `make trust` removes the stale entry and opens the Accessibility pane. Grant
  the app again.
- **Once:** create a self-signed code-signing certificate named `Paraketto Dev`. The Makefile
  picks it up automatically and the grant survives rebuilds.

  ```bash
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout key.pem -out cert.pem \
    -subj "/CN=Paraketto Dev" \
    -addext "keyUsage=critical,digitalSignature" \
    -addext "extendedKeyUsage=critical,codeSigning" \
    -addext "basicConstraints=critical,CA:false"
  openssl pkcs12 -export -legacy -out dev.p12 -inkey key.pem -in cert.pem -passout pass:x -name "Paraketto Dev"
  security import dev.p12 -k ~/Library/Keychains/login.keychain-db -P x -T /usr/bin/codesign
  rm key.pem cert.pem dev.p12
  make trust   # one last re-grant for the newly signed build
  ```

  The first `codesign` with the new key may show a keychain dialog; choose "Always Allow".

## Speech recognition

`Sources/ParakeetTDT` is a self-contained Core ML runner for Parakeet TDT 0.6B v2 and Ultra,
with no package dependencies. The tensor dimensions are shared, while the blank token,
vocabulary, model files and caches are selected per mode:

| File | Role |
|---|---|
| `ParakeetModel.swift` | mode-specific repository, filenames, vocabulary and blank token |
| `ModelStore.swift` | finds or downloads the selected mode's Core ML assets |
| `ModelFileDownload.swift` | cancellable file transfers with live byte progress and temporary-file cleanup |
| `ParakeetModels.swift` | loads the four Core ML models (preprocessor on CPU, the rest on the ANE), the VAD head and the vocabulary |
| `ParakeetTranscriber.swift` | `Session`: audio in, segments decoded as their cuts become known; mel → encoder → decode with 1 s of context on each side |
| `TdtDecoder.swift` | greedy Token-and-Duration Transducer loop with the 2-layer LSTM state |
| `VoiceActivity.swift` | places cuts at pauses found by the model's own VAD head |
| `EncoderSet.swift` | picks the shortest encoder window that holds a segment |
| `MultiArray.swift` | small `MLMultiArray` helpers |

The Ultra model files are FluidAudio's Core ML conversion (Apache 2.0) of moondream's checkpoint
(CC-BY-4.0): a fixed 15 s preprocessor and int8 encoder, an LSTM decoder, and a joint network
that returns token, probability and duration bin directly.

### Voice-activity head

Parakeet Ultra's weights carry a small VAD head on the encoder's subsampler, which moondream's
engine uses to cut long audio at pauses. FluidAudio's export leaves it out, so
`tools/export/export_vad_head.py` exports subsampler plus head to `Resources/VadHead.mlpackage`
(8.7 MB), fetching only the 9 MB of needed tensors from the safetensors file. The wiring
(SiLU after `proj` and `ctx`, no residual) follows parakeet.cpp's published analysis, since
moondream does not document the head; it gives one speech probability per 80 ms frame.

The bundled export includes its own subsampler, so the app also uses it as a standalone pause
detector in `eng` mode. It does not reuse Ultra's recognition encoder or vocabulary for v2.

A `Session` segments the recording while it is still going. Every 0.24 s it runs the VAD head
over the newest audio; a silence of 0.8 s ends the segment before it, which is then encoded and
decoded right away on the otherwise idle ANE and CPU. At stop only the audio after the last cut is
left: nothing if you paused, otherwise one short encoder pass.

Audio memory stays bounded: the recognizer discards consumed samples, retaining only the
next window's overlap and the incomplete VAD block. Capture writes a temporary Float32 WAV
as audio arrives; the transcriber reads small batches from it. Only the latest file-length
notification is queued, so inference lag accumulates on disk instead of in an audio queue.
At completion, the WAV moves into History without loading it into memory. Cancelled or
failed recordings are removed when their users release the temporary file. Transcript
tokens still grow with recording length. New recordings use about 3.84 MB of disk per minute;
existing PCM16 History recordings remain playable.

VAD blocks end on whole 80 ms frames (14.96 s per complete block), so rounding each block
does not accumulate a timestamp offset over long recordings. The encoder's maximum input remains 15 s.

Every segment is decoded with a fresh decoder, as moondream's runtime does. Carrying the LSTM state
from segment to segment looked like free language context, but over a few minutes of speech it
drifts until whole segments decode to blanks.

- **Cuts at pauses** lie in silence, so each side keeps the tokens on its side of the cut, with about
  a second of audio around it for the encoder.
- **Cuts through speech**, when 11 s pass without a 0.8 s pause, overlap: both segments encode the
  2 s on each side of the cut, their tokens there are aligned with a longest common subsequence
  (same token at nearly the same frame), and the text is joined at the matched token nearest the
  cut. If the two sides do not agree, the one that heard more of the overlap wins. This is the
  scheme NVIDIA uses for long audio with Parakeet; it replaced a one-second context plus exact
  dedupe that doubled or dropped a word at about a third of such cuts.
- **Empty windows are retried.** The model sometimes returns nothing at all for a window that
  starts mid-word (FluidAudio's own models do the same), so a window with speech that decodes
  empty is decoded again from a later start.

`tests/make_longform.py` joins the LibriSpeech clips into one to three minute files with 0.3 to 1 s
gaps, and `tests/score.py` computes word error rate the Open ASR Leaderboard way: Whisper's
EnglishTextNormalizer on both sides, then `jiwer`. `make compare` and `make longform` feed audio in
100 ms chunks like the app:

| Version | Long files | Short clips |
|---|---|---|
| state carried across cuts, 13 s windows | about 9% (counted before the scorer existed) | 0.20% |
| pause commits, overlap join, fresh decoder | 0.20% | 0.20% |

The remaining errors are the model's own ("bade fare" for "bade fair", "Murcheston"). Pause
decisions use only the audio heard so far, so feeding a whole file at once can cut differently
(0.40% on the long files).

### Faster than the stock export

Two optional exports, built locally with `uv` (they are too large for the repo) and picked up
automatically by the app and CLI from `~/Library/Application Support/Paraketto/models/`:

- `make encoder-export`: the encoder as one multi-function Core ML model with 5 s, 10 s and 15 s
  windows (605 MB, weights shared), traced from Hugging Face `transformers`' `ParakeetForTDT` and
  quantized to int8 per channel. A segment uses the shortest window that holds it, so a 4.5 s
  utterance costs a 17 ms encoder pass instead of 55 ms. Needs macOS 15 for multi-function models.
- `make decoder-export`: one Core ML model that advances the LSTM with the emitted token and
  decides the next frame, replacing two framework calls per token step with one (35 MB).

Measured on the 17 LibriSpeech clips on an M1 Max, same transcripts in every configuration:

| Configuration | RTFx |
|---|---|
| FluidAudio models, two-call decode | 80x |
| + fused decoder step | 93x |
| + multi-window encoder | 124x |

The remaining time is split between the encoder on the ANE (17 to 55 ms depending on window) and
the sequential decode on the CPU: 0.73 ms per emitted token inside the fused step, bound by
streaming the fp32 LSTM and joint weights through one core (Core ML's per-call floor is 35 µs, and
smaller weights do not help because the CPU path expands them at load). A Metal decoder that ran
the greedy loop on the GPU reached 182x with identical transcripts (commit `89c4e5e`), but hand-written
kernels failed silently twice on the way there, and for dictation the 30 ms it saves on a 10 s
utterance is below perception, so it was removed in favour of the framework path.

### Measuring

- `xcrun xctrace record --template 'Core ML' --launch -- .build/release/paraketto-cli clip.wav`
  records every Core ML prediction and the Neural Engine's hardware intervals; export tables with
  `xcrun xctrace export --input <trace> --toc`.
- `paraketto-cli --plan` prints each model's operation count, how Core ML assigns operations to
  CPU, GPU and ANE, and which operators force the CPU.
- `PARAKEET_TIMING=1` prints mel, encoder and decode time per recording, with the number and total
  time of fused, joint and decoder Core ML calls; `--fused <model>` swaps in another fused step
  export for A/B timing; `--chunk-ms N` feeds each file to a session in N ms chunks like the app
  and reports the time spent after the last chunk; `--tokens` prints each
  token with its frame; `--vad-probe` prints the VAD strip; `PARAKEET_DEBUG=1` logs every joint decision.

### Checking against FluidAudio

`make test` checks file storage, audio decoding, playback, paste delivery, and cache validation
without downloading recognition models. Tests that need the real models are skipped by default.
After `make prepare-models`, enable them with the LibriSpeech `1995-1826-0002.wav` fixture:

```bash
PARAKETTO_TEST_AUDIO=/path/to/1995-1826-0002.wav make test-models
```

The fixture defaults to `data/librispeech/1995-1826-0002.wav` inside this checkout. Opting into
model tests requires the models and fixture to exist; missing prerequisites fail the run.
These tests check a `multi → eng → multi` round trip, silence and quiet speech, long-recording
memory and VAD timing, file imports, and cancellation.
The remaining benchmark results in this section refer to `multi`.

`tests/fixtures/librispeech_fluidaudio.tsv` holds FluidAudio's transcripts of 17 LibriSpeech
clips. `make compare` transcribes `data/librispeech/*.wav` (set `CLIPS=` to another location) and reports every
difference and the real-time factor. Result at the time of writing: 15 of 17 identical. The two
differences are a British/American spelling choice
on a clip cut differently, and a trailing period that FluidAudio drops because it discards any
token whose duration runs past the end of the audio, which standard TDT decoding does not.

`paraketto-cli --tokens` prints each token with its emission frame, `--vad-probe` prints the
VAD strip, and `PARAKEET_DEBUG=1` logs every joint decision.

## Layout

| File | Role |
|---|---|
| `App.swift` | app entry point, menu-bar/Transcripts/Settings scenes, startup |
| `MenuBarViews.swift` | status, loading indicators, mode selection, and menu actions |
| `MenuControls.swift` | shared segmented control and menu button styling |
| `MenuBarSecondaryClick.swift` | right-click shortcut to Transcripts |
| `DictationController.swift` | record → transcribe → paste state machine, Esc cancel, meter ticker |
| `Hotkey.swift` | Carbon global hotkey for key combinations |
| `ModifierTap.swift` | bare modifier tap detection via global event monitors |
| `ShortcutSettings.swift` | shortcut model (modifier tap or chord), persisted in UserDefaults |
| `SettingsView.swift` | Settings window with the shortcut recorder |
| `AudioRecorder.swift` | `AVAudioEngine` mic capture to 16 kHz mono Float32, dBFS level smoothing |
| `RecordingFile.swift` | incremental WAV storage, bounded reads, History handoff and temporary-file cleanup |
| `AudioFileReader.swift` | bounded decoding/resampling of imported recordings |
| `Transcriber.swift` | thin adapter over `ParakeetTDT` |
| `Paster.swift` | destination check, pasteboard write, synthetic Cmd+V |
| `Overlay.swift` | floating recording pill |
| `History.swift` | persisted entries + WAV files under `~/Library/Application Support/Paraketto/history/` |
| `AudioPlayer.swift` | playback of one stored recording at a time |
| `TranscriptsView.swift` | grouped transcript list, import status and drop target |
| `TranscriptRow.swift` | four-line preview, expansion, hover actions and copy feedback |
| `ModelPreparation.swift` | headless model preparation for builds |
| `ParakeetArtwork.swift` | loads menu-bar assets derived from `Resources/paraketto.png` |
| `Log.swift` | `~/Library/Logs/Paraketto.log`, truncated per launch, never contains transcripts |

Application sources live in `Sources/Paraketto`; recognition is isolated in `Sources/ParakeetTDT`,
and the CLI is in `Sources/ParakettoCLI`. `tests/swift` contains regression tests; `tools/export`
contains optional model exporters and probes. Generated bundles, caches, local audio fixtures, and
build/test artifacts are ignored. The source icon lives in `Resources`; `tools/make_icon.swift`
generates the app icon and template menu-bar image together.
