# Silero CPU reference fixtures

`reference.json` contains frame probabilities from Jarvis's independently built
`vad_test` at source revision `7df37bab2381a28346c875de18b0e2decca4539a`.
The model and reference executable hashes are recorded beside the values. The
adapted Paraketto implementation uses the CPU backend of the pinned llama.cpp
ggml revision, rather than Jarvis's older ggml API.

The deterministic fixture is four seconds of silence, low amplitude noise,
periodic synthetic sound and silence, generated identically in the C++ test and
`generate.py`. The speech fixture refers to the existing optional LibriSpeech
test download; its audio is not duplicated here. The test verifies its WAV hash
before comparison and explicitly skips it when absent. It always runs the
deterministic fixture. No ONNX comparison is claimed: the local reference ONNX
model was unavailable during validation.

After building `vad_test` in a clean checkout of the pinned Jarvis revision,
regenerate with:

```sh
python3 tests/vadfixtures/generate.py \
  --jarvis /path/to/jarvis \
  --weights /path/to/silero_vad.bin
```

`make test-segmenter` compares every frame within `1e-4`, checks that reset
reproduces probabilities within `1e-6`, and tests silence coverage, real speech
retention, pause handling, bounded fallback groups, invalid model rejection and
the final partial audio frame. The observed maximum differences were `1.33e-5`
for the deterministic fixture and `5.74e-5` for speech. Preparation is explicit
through `scripts/prepare_vad.py`; recording never downloads model files.

The segmentation tests also exercise real speech at low levels, pauses up to
23 seconds, forced boundaries in 113.75 seconds of continuous speech, and
recovery before and after a finalized boundary. Warmup ends immediately before
the resumed sample, including when the preceding window is not frame aligned.

The detector uses onset/continuation thresholds of 0.15/0.10 and normalizes only
its analysis frames toward RMS 0.01, capped at 16x gain. Raw audio and ASR input
samples are unchanged. This addresses the quiet LibriSpeech utterance
`7127-75946-0001` (“AH VERY WELL”): without this analysis gain, ASR recognized
the first word but VAD could omit it. Both 0.025 and 0.016 input gains retain
over 99.5% of source energy after 3- and 23-second pauses in the regression.
Energy retention is checked alongside actual ASR comparisons, not treated as
proof of word accuracy.

Noise comparisons cover white/brown noise, 60 Hz hum, breath-shaped noise at
three levels and quiet background from a real recording. Steady noise and hum
remain skipped; one ambiguous breath case retains 0.864 seconds out of 12.
The test permits at most one second for that ambiguity, favoring retention of
uncertain quiet speech over maximum noise removal.
