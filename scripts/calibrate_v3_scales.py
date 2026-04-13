#!/usr/bin/env python3
"""Calibrate FP8 activation scales for V3 by running multiple utterances.

Each binary invocation produces 2 SCALES lines: one from silence warmup
(skip) and one from actual audio (keep). Takes element-wise max, but also
reports per-utterance scales for site 216 (sub_out) to identify outliers.

Usage: uv run python scripts/calibrate_v3_scales.py
"""

import subprocess
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "build" / "paraketto"
N_SITES = 218

# Calibration utterances: mix of languages
CALIBRATION_WAVS = []

# Add English reference utterance (same as V2 calibration)
en_wav = ROOT / "data" / "librispeech" / "1089-134686-0000.wav"
if en_wav.exists():
    CALIBRATION_WAVS.append(str(en_wav))

# Add first 5 clips from each language
for lang in ["german", "italian", "french"]:
    manifest_path = ROOT / "data" / lang / "manifest.json"
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        for entry in manifest[:5]:
            wav_path = ROOT / "data" / lang / entry["audio_path"]
            if wav_path.exists():
                CALIBRATION_WAVS.append(str(wav_path))

print(f"Calibrating on {len(CALIBRATION_WAVS)} utterances...")

all_scales = []
wav_labels = []
for i, wav in enumerate(CALIBRATION_WAVS):
    result = subprocess.run(
        [str(BINARY), "--model", "v3", wav],
        capture_output=True, text=True, timeout=60
    )
    scales_lines = []
    for line in result.stderr.split("\n"):
        if line.startswith("SCALES:"):
            vals = [float(x) for x in line.split()[1:]]
            assert len(vals) == N_SITES, f"Expected {N_SITES} scales, got {len(vals)}"
            scales_lines.append(vals)
    # Skip first (silence warmup), keep rest
    for s in scales_lines[1:]:
        all_scales.append(s)
        wav_labels.append(f"{Path(wav).parent.name}/{Path(wav).stem}")
    lang = Path(wav).parent.name
    name = Path(wav).stem
    print(f"  [{i+1}/{len(CALIBRATION_WAVS)}] {lang}/{name}")

print(f"\nCollected {len(all_scales)} scale sets")
arr = np.array(all_scales)

# Show per-utterance value at outlier sites
print("\nSite 216 (sub_out) per utterance:")
for j in range(len(all_scales)):
    print(f"  {wav_labels[j]:30s}  {arr[j, 216]:.6e}")

print("\nSite 217 (enc_proj) per utterance:")
for j in range(len(all_scales)):
    print(f"  {wav_labels[j]:30s}  {arr[j, 217]:.6e}")

# Strategy: element-wise max, but cap outliers using percentile
scales_max = np.max(arr, axis=0)
scales_p95 = np.percentile(arr, 95, axis=0)
scales_p99 = np.percentile(arr, 99, axis=0)

print(f"\n--- Element-wise max ---")
print(f"  site 216: {scales_max[216]:.6e}")
print(f"  site 217: {scales_max[217]:.6e}")
print(f"--- 95th percentile ---")
print(f"  site 216: {scales_p95[216]:.6e}")
print(f"  site 217: {scales_p95[217]:.6e}")

# Use max for the final output (conservative, prevents clipping)
scales = scales_max

print(f"\nstatic const float FP8_BAKED_ACT_SCALES_V3[{N_SITES}] = {{")
for i in range(0, N_SITES, 6):
    chunk = scales[i:i+6]
    parts = []
    for j, v in enumerate(chunk):
        idx = i + j
        suffix = "f," if idx < N_SITES - 1 else "f"
        parts.append(f"{v:.8e}{suffix}")
    line = "    " + " ".join(parts)
    blk = i // 9
    off = i % 9
    if off == 0 and blk < 24:
        line += f"  // blk {blk}"
    elif i >= 216:
        line += "  // sub_out, enc_proj"
    print(line)
print("};")
