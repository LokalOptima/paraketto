# paraketto vs Whisper (faster-whisper)

Speed and accuracy comparison of **paraketto** (Parakeet TDT 0.6B, custom CUDA engine)
against **OpenAI Whisper** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2), on identical audio with identical WER scoring.

- **Date:** 2026-06-17 (multilingual FP8 recalibrated with headroom 2026-07-22; see §5.3)
- **GPU:** NVIDIA RTX 5070 Ti (16 GB, Blackwell SM120), batch size 1
- **paraketto build:** FP8 (CUTLASS E4M3) — the flagship variant for this GPU
- **Whisper runtime:** faster-whisper 1.2.1 / CTranslate2 4.8.0, float16

> **TL;DR** — On batch-1, paraketto is **14–46× faster than even Whisper's best-case
> configuration**, while landing in **large-v3's accuracy class**: it wins on clean and
> long-form English and on Italian/French, and trades the rest — Whisper edges it on
> conversational/hard English and German. paraketto does this with a model **2.6× smaller
> than large-v3** and a **~2 MB, cudart-only deployment** vs Whisper's ~2.6 GB ONNX+TRT stack.

---

## 1. Methodology

The goal was a *fair* comparison, so the Whisper side mirrors the
[HF Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard)
faster-whisper harness exactly (which is also the source of paraketto's WER normalizer):

- **Same data:** the same WAV files and `manifest.json` references paraketto benchmarks on
  (`data/{librispeech,earnings22,long,difficult}` for English;
  `data/{german,italian,french}` curated FLEURS for multilingual).
- **Same WER scoring:** Whisper's `EnglishTextNormalizer` for English; the HF Open-ASR
  multilingual normalizer + `num2words` for de/it/fr. Identical to paraketto's own scripts.
- **RTFx = audio_seconds / inference_seconds**, model preloaded, with warmup. Audio is
  **pre-decoded to 16 kHz float32 arrays** so only `transcribe()` is timed — decode/resample
  is excluded, matching both the leaderboard and paraketto's server-side `inference_time_s`.
- **Models:** the latest version of each Whisper size — `tiny, base, small, medium,
  large-v3, large-v3-turbo`. English datasets use the `.en` variant where it exists
  (tiny/base/small/medium); large-v3 and turbo have no `.en`, so the multilingual model is
  used. Multilingual datasets always use the multilingual model with a language hint.

Two Whisper configurations were run:

| Config | Decoding | VAD | Notes |
|---|---|---|---|
| **A — sequential** | beam_size=5 | off | Open ASR Leaderboard standard ("best accuracy") |
| **B — batched** | beam_size=1 (greedy) | on (batch_size=16) | `BatchedInferencePipeline`; greedy matches paraketto's decoder |

paraketto numbers are from its README (RTX 5070 Ti, FP8 V2/V3).
Harness + raw results live in `/home/lapo/git/whisper-bench/`
(`bench_whisper.py`, `results.json` = config A, `results_batched_beam1.json` = config B).

---

## 2. Model size

| Model | Params | Weights on disk |
|---|---|---|
| **paraketto / Parakeet TDT 0.6B** | **0.6 B** | **0.63 GB (FP8)** / 1.24 GB (FP16) |
| whisper tiny | 39 M | 73 MB |
| whisper base | 74 M | 139 MB |
| whisper small | 244 M | 462 MB |
| whisper medium | 769 M | 1.5 GB |
| whisper large-v3-turbo | 809 M | 1.6 GB |
| **whisper large-v3** | **1.55 B** | **2.9 GB** |

paraketto runs a model **~2.6× smaller than large-v3** — smaller even than Whisper *medium*.

---

## 3. Headline results — paraketto FP8 vs Whisper's *best-case*

"Best-case Whisper" = the lowest WER achieved by *any* model size in *either* config for
that dataset (i.e. the most generous possible Whisper number — see §6 caveat).

### English

| Dataset | paraketto FP8 (WER / RTFx) | Best Whisper (WER / RTFx, config) | Accuracy | paraketto speed |
|---|---|---|---|---|
| librispeech | **1.42% / 1043×** | 1.38% / 34× — large-v3 batched | tie | **31× faster** |
| earnings22  | 11.82% / 979× | **10.34%** / 26× — large-v3 seq | Whisper | **38× faster** |
| long        | **1.81% / 1387×** | 1.86% / 97× — large-v3 batched | tie | **14× faster** |
| difficult   | 16.46% / 1210× | 16.62% / 31× — large-v3 seq | tie | **39× faster** |

### Multilingual

| Language | paraketto FP8 (WER / RTFx) | Best Whisper (WER / RTFx, config) | Accuracy | paraketto speed |
|---|---|---|---|---|
| german  | 7.90% / 1256× | **7.53%** / 41× — large-v3 batched | Whisper | **31× faster** |
| italian | 3.90% / 1490× | 3.98% / 93× — turbo seq | tie | **16× faster** |
| french  | **4.84%** / 1158× | 5.93% / 25× — large-v3 seq | paraketto | **46× faster** |

---

## 4. Verdict

**Speed — decisive win for paraketto.** 14–46× faster than even Whisper's *best* config,
and that compares paraketto at batch-1 against Whisper configurations that *include*
batching. There is no contest on throughput.

**Accuracy — same class, dataset-dependent.** paraketto ties or wins on clean read speech
(librispeech), long-form English, Italian (tie, 3.90% vs 3.98%) and French (4.84% vs 5.93%);
Whisper edges it on conversational audio (earnings22), the hardest set (difficult), and German
(7.90% vs 7.53%). Neither is uniformly more accurate. The multilingual FP8 numbers were
recalibrated with headroom (see below) — they now match the FP16 backend within 0.3 point and
produce no empty transcriptions across the full Common Voice test set.

**Attribution.** The *accuracy* is NVIDIA's Parakeet TDT 0.6B model. paraketto's contribution
is the *engine*: the speed, the FP8 quantization, the 1.2 GB VRAM footprint, and the
near-zero-dependency deployment (below).

---

## 5. Two findings worth knowing

### 5.1 Whisper large-v3 hallucinates on long-form (config A)

In sequential mode, `long` WER for large-v3 blows up to **9.28%** (medium.en gets 2.60%) —
the classic Whisper `condition_on_previous_text` repetition-loop failure, to which large-v3
is especially prone. A model getting *worse* as it scales up is the tell.

**Batched-VAD (config B) fixes it:** large-v3 `long` drops **9.28% → 1.86%**, turbo
**6.77% → 2.05%**, by chunking at silence so errors can't propagate across windows.

paraketto avoids this structurally: its Conformer encoder ingests up to **120 s in a single
pass** (`MAX_MEL_FRAMES`), and every clip in `long` is ≤120 s — so it transcribes them with
full context and no chunking, while Whisper (capped at 30 s) must stitch ~4 windows per clip.

### 5.2 Batched-VAD is a real tradeoff, not a free win

The same VAD chunking that fixes long-form *hurts* dense/noisy speech: under config B,
`earnings22` regresses (large-v3 10.34% → 14.03%) and `difficult` regresses badly
(large-v3 16.62% → **27.88%**). **No single Whisper config is best across all datasets** —
sequential wins hard audio, batched-VAD wins long-form/clean.

### 5.3 V3 FP8 multilingual scales were recalibrated with headroom

The multilingual FP8 numbers here are lower than earlier revisions (german 9.18% → 7.90%,
italian 4.83% → 3.90%, french 6.29% → 4.84%) because the earlier V3 FP8 activation scales were
too tight. They were calibrated on 16 clean studio clips, which left 179 of 218 quantization
sites below the activation ranges real audio produces (early conv sites up to 6.7× too small).
On Common Voice audio this saturated the E4M3 range and, combined with nondeterministic GEMM
accumulation order, collapsed the TDT decoder to an empty string on ~20% of German/French clips
— a failure invisible on the 50-clip FLEURS sets used here. Recalibrating over 162 clips
(including Common Voice) with a 1.5× headroom factor fixed it: FP8 now matches the FP16 backend
within 0.3 point and produces **0 scale-induced empty transcriptions across the full 38,918-clip
Common Voice test set**. The English V2 scales were untouched, so all §3 English numbers are
unchanged. Full write-up in `blog.md`.

---

## 6. Caveats (read before quoting these numbers)

- **"Best-case Whisper" is an upper envelope.** §3 cherry-picks the best of
  {6 sizes} × {2 configs} per dataset. No *single* deployable Whisper config achieves all of
  them simultaneously (config B fixes `long` but ruins `earnings22`/`difficult`). If forced
  to pick one config, **large-v3-turbo sequential** is the best all-rounder and still runs
  ~13–25× slower than paraketto FP8.
- **The dataset mix favors paraketto.** It is librispeech-heavy, and `long` is capped exactly
  at paraketto's 120 s single-pass ceiling. On the full public Open ASR Leaderboard
  (which adds AMI, gigaspeech, tedlium, voxpopuli…), large-v3 has a better *average* WER.
- **paraketto FP8 vs FP16.** FP8 is marginally less accurate on clean speech than the FP16
  CUTLASS backend (librispeech 1.42% vs 1.38%) but smaller/faster; numbers here use FP8.
- Whisper is **multilingual (99 languages)**; Parakeet TDT 0.6B v2 is English-only, v3 covers
  25 EU languages. This comparison only covers the languages both support.

---

## 7. Deployment footprint

The most concrete real-world differentiator, and verifiable in one command:

| | paraketto FP8 | faster-whisper (CTranslate2) | onnx-asr (ONNX Runtime + TensorRT) |
|---|---|---|---|
| Binary / package | **2.5 MB** | Python + CT2 wheels | Python + ORT + TRT |
| Non-system deps | **`libcudart.so` only** (statically linkable away) | CUDA + cuDNN libs | **~2.6 GB** of CUDA/cuDNN/TRT/ORT libs |
| VRAM | **1.2 GB** | ~3 GB (large-v3) | model-dependent |
| Python required | **no** | yes | yes |

paraketto also benchmarks **~2× faster than the standard ONNX-Runtime + TensorRT
(`onnx-asr`) Parakeet deployment** that most people actually use (529× → 1043× on
librispeech; ~1.6× on the other sets). Note this is vs the ONNX-Runtime+TRT path, *not* a
hand-tuned native TensorRT engine (against which the gap is ~10–15%).

---

## 8. Reproduce

```bash
# paraketto
make bench-fp8        # English V2
make bench-v3         # multilingual V3

# Whisper (isolated env, does not touch this repo)
cd /home/lapo/git/whisper-bench
uv venv .venv && source .venv/bin/activate
uv pip install faster-whisper jiwer whisper-normalizer num2words
python bench_whisper.py                       # config A: beam=5 sequential
BATCHED=1 BEAM=1 BATCH_SIZE=16 python bench_whisper.py   # config B: batched greedy + VAD
```

## Appendix — full Whisper results (all sizes, both configs)

WER % / RTFx. Config A = beam=5 sequential. Config B = batched greedy (beam=1) + VAD.

### English

| Model | librispeech A | B | earnings22 A | B | long A | B | difficult A | B |
|---|---|---|---|---|---|---|---|---|
| tiny     | 4.39 / 123 | 4.74 / 272 | 16.25 / 113 | 21.27 / 267 | 4.78 / 154 | 5.36 / 436 | 17.04 / 117 | 17.80 / 292 |
| base     | 3.53 / 97  | 3.92 / 206 | 14.03 / 96  | 19.35 / 192 | 3.83 / 130 | 3.69 / 406 | 17.80 / 80  | 15.95 / 217 |
| small    | 2.37 / 73  | 2.41 / 110 | 12.41 / 65  | 16.84 / 103 | 2.73 / 87  | 3.00 / 272 | 17.46 / 75  | 17.88 / 127 |
| medium   | 2.54 / 43  | 2.67 / 55  | 11.52 / 39  | 15.51 / 50  | 2.60 / 46  | 3.00 / 147 | 18.56 / 46  | 18.05 / 63  |
| large-v3 | 2.71 / 23  | 1.38 / 34  | 10.34 / 26  | 14.03 / 31  | 9.28 / 12  | 1.86 / 97  | 16.62 / 31  | 27.88 / 42  |
| turbo    | 1.81 / 64  | 1.64 / 76  | 11.23 / 48  | 17.43 / 61  | 6.77 / 66  | 2.05 / 190 | 16.96 / 62  | 26.53 / 87  |
| **paraketto FP8** | **1.42 / 1043** | | **11.82 / 979** | | **1.81 / 1387** | | **16.46 / 1210** | |

### Multilingual

| Model | german A | B | italian A | B | french A | B |
|---|---|---|---|---|---|---|
| tiny     | 32.60 / 132 | 36.00 / 233 | 27.03 / 166 | 39.32 / 257 | 36.15 / 96 | 40.13 / 213 |
| base     | 21.49 / 137 | 24.24 / 219 | 16.44 / 135 | 16.69 / 210 | 23.14 / 98 | 27.26 / 166 |
| small    | 11.48 / 90  | 12.76 / 127 | 7.71 / 96   | 7.88 / 122  | 12.87 / 64 | 14.39 / 90  |
| medium   | 8.45 / 51   | 9.27 / 65   | 5.34 / 54   | 5.17 / 65   | 7.88 / 36  | 8.32 / 46   |
| large-v3 | 9.73 / 35   | 7.53 / 41   | 4.07 / 37   | 4.49 / 41   | 5.93 / 25  | 6.36 / 28   |
| turbo    | 7.81 / 75   | 8.17 / 101  | 3.98 / 93   | 4.49 / 106  | 6.36 / 63  | 6.65 / 75   |
| **paraketto FP8** | **7.90 / 1256** | | **3.90 / 1490** | | **4.84 / 1158** | |
