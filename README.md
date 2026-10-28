# Paraketto

Minimal dictation for macOS. Tap Right Option, speak, tap it again: the text is
pasted into whatever app has focus. Speech recognition is
[Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) running on the Apple
Neural Engine with Core ML. Nothing leaves the Mac.

- Menu bar only, no Dock icon
- Shortcut: tap `Right ⌥` to start and stop a recording, `Esc` discards it. A tap is a press and release under 0.6 s with no other key in between, so Option+letter typing is unaffected. Change it in Settings (`⌘,` from the menu) by tapping another modifier key or pressing any key combination
- Transcription runs while you speak: every pause of 0.8 s ends a segment and it is transcribed right away, so when you pause before stopping the text is already there
- Floating pill while recording: a level meter on a -50 to -10 dBFS scale with 30 ms attack / 160 ms release smoothing
- Text is pasted only if the app that had focus when you started still has it; otherwise it is kept in history
- History window (`⌘H` from the menu): every dictation with its recording, grouped by day, with play, copy, delete, and word count / WPM stats
- 25 European languages with automatic detection, punctuation and capitalization included
- Model download on first launch (613 MB to `~/Library/Application Support/Paraketto/models/parakeet-ultra`; an existing FluidAudio cache is reused)

The transcript also stays on the clipboard after pasting. The previous clipboard contents are not
restored, since a timed restore races slow readers and the old contents are one undo away.

## Build and run

Requires macOS 14+, Apple Silicon, and Xcode 16 or later.

```bash
make run
```

This builds with SwiftPM, compiles the VAD model, assembles `Paraketto.app`, signs it, and launches it.

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

`Sources/ParakeetTDT` is a self-contained Core ML runner for the Parakeet TDT 0.6B v3 family
(v3, Ultra, Redux, Phonon-2 share one contract), about 600 lines with no dependencies:

| File | Role |
|---|---|
| `ModelStore.swift` | finds the models or downloads them from `FluidInference/parakeet-ultra-coreml` |
| `ParakeetModels.swift` | loads the four Core ML models (preprocessor on CPU, the rest on the ANE), the VAD head and the vocabulary |
| `ParakeetTranscriber.swift` | `Session`: audio in, segments decoded as their cuts become known; mel → encoder → decode with 1 s of context on each side |
| `TdtDecoder.swift` | greedy Token-and-Duration Transducer loop with the 2-layer LSTM state |
| `VoiceActivity.swift` | places cuts at pauses found by the model's own VAD head |
| `EncoderSet.swift` | picks the shortest encoder window that holds a segment |
| `MultiArray.swift` | small `MLMultiArray` helpers |

The model files are FluidAudio's Core ML conversion (Apache 2.0) of moondream's checkpoint
(CC-BY-4.0): a fixed 15 s preprocessor and int8 encoder, an LSTM decoder, and a joint network
that returns token, probability and duration bin directly.

### Voice-activity head

Parakeet Ultra's weights carry a small VAD head on the encoder's subsampler, which moondream's
engine uses to cut long audio at pauses. FluidAudio's export leaves it out, so
`tools/export/export_vad_head.py` exports subsampler plus head to `Resources/VadHead.mlpackage`
(8.7 MB), fetching only the 9 MB of needed tensors from the safetensors file. The wiring
(SiLU after `proj` and `ctx`, no residual) follows parakeet.cpp's published analysis, since
moondream does not document the head; it gives one speech probability per 80 ms frame.

A `Session` segments the recording while it is still going. Every 0.24 s it runs the VAD head
over the newest audio; a silence of 0.8 s ends the segment before it, which is then encoded and
decoded right away on the otherwise idle ANE and CPU. Speech with no such pause is cut when a 13 s
window fills, at the last shorter pause or the least speech-like frame near its end. Each segment
is encoded with about a second of neighbouring audio on both sides, tokens are emitted only for the
segment itself, and a word straddling a cut is finished before the next segment resumes. At stop
only the audio after the last cut is left: nothing if you paused, otherwise one short encoder pass.

A segment after a pause starts the decoder fresh, as moondream's runtime does; one after a window
cut keeps the decoder state, since it continues a sentence. Carrying the state across every cut
looked like free language context, but over a few minutes of speech it drifts until whole segments
decode to blanks, while a fresh state at a window cut loses the sentence in progress.
`tests/make_longform.py` joins the LibriSpeech clips into one to three minute files with 0.3 to
1 s gaps to test this. On those files, fed in 100 ms chunks:

| Setting | Long files, word errors of 994 (hand count) | Short clips, identical to FluidAudio |
|---|---|---|
| state carried across every cut, 13 s windows only (before) | 90 | 15 of 17 |
| fresh state at every cut, 0.8 s pause commits | 12 | 13 of 17 |
| fresh after pauses, carried across window cuts, 0.48 s pause commits | 76 | 13 of 17 |
| fresh after pauses, carried across window cuts, 0.8 s pause commits | 13 | 15 of 17 |

The last row is the default. Shorter commit pauses cut at breaths inside sentences; 0.8 s cuts
between them. Pause decisions are made on the audio heard so far, so feeding a whole file at once
can place cuts differently from the app, which feeds 100 ms chunks; `make compare` and
`make longform` use the app's chunked mode and score word error rate with `tests/score.py`: Whisper's
EnglishTextNormalizer on both sides, then `jiwer`, as the Open ASR Leaderboard does. Current result:
0.20% on the short clips (one "colourists"), 1.21% on the long files.

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

`tests/fixtures/librispeech_fluidaudio.tsv` holds FluidAudio's transcripts of 17 LibriSpeech
clips. `make compare` transcribes them (set `CLIPS=` to their location) and reports every
difference and the real-time factor. Result at the time of writing: 15 of 17 identical. The two
differences are a British/American spelling choice
on a clip cut differently, and a trailing period that FluidAudio drops because it discards any
token whose duration runs past the end of the audio, which standard TDT decoding does not.

`paraketto-cli --tokens` prints each token with its emission frame, `--vad-probe` prints the
VAD strip, and `PARAKEET_DEBUG=1` logs every joint decision.

## Layout

| File | Role |
|---|---|
| `App.swift` | `MenuBarExtra` shell, History and Settings scenes |
| `DictationController.swift` | record → transcribe → paste state machine, Esc cancel, meter ticker |
| `Hotkey.swift` | Carbon global hotkey for key combinations |
| `ModifierTap.swift` | bare modifier tap detection via global event monitors |
| `ShortcutSettings.swift` | shortcut model (modifier tap or chord), persisted in UserDefaults |
| `SettingsView.swift` | Settings window with the shortcut recorder |
| `AudioRecorder.swift` | `AVAudioEngine` mic capture to 16 kHz mono Float32, dBFS level smoothing |
| `Transcriber.swift` | thin adapter over `ParakeetTDT` |
| `Paster.swift` | destination check, pasteboard write, synthetic Cmd+V |
| `Overlay.swift` | floating recording pill |
| `History.swift` | persisted entries + WAV files under `~/Library/Application Support/Paraketto/history/`, playback |
| `HistoryWindow.swift` | history list grouped by day |
| `Log.swift` | `~/Library/Logs/Paraketto.log`, truncated per launch, never contains transcripts |
