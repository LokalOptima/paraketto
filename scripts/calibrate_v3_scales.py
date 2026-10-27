#!/usr/bin/env python3
"""Calibrate FP8 activation scales for V3 over a broad, diverse audio set.

Each binary invocation (with PARAKETTO_CALIB=1) emits 2 "SCALES:" lines: one
from the silence warmup (skipped) and one from the real audio (kept). We take
the element-wise max across all calibration utterances, then multiply by a
HEADROOM factor so real-world audio that exceeds the calibration set does not
saturate the E4M3 range (the cause of the V3-FP8 empty-output collapse).

Diversity matters: the previous calibration used a single reference utterance
(commit 18156a5), leaving many sites with zero headroom — Common Voice audio
drove activations up to 2.5x past the calibrated amax, saturating FP8 and
collapsing the TDT decoder to all-blank. Here we include Common Voice (the
triggering distribution), FLEURS, librispeech, and the difficult/earnings sets.

Usage: uv run python scripts/calibrate_v3_scales.py [HEADROOM]   (default 1.5)

For custom checkpoints, use calibrate_fp8.py to embed scales in their weight
files without replacing the built-in V3 defaults.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "paraketto.fp8"
N_SITES = 218
HEADROOM = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5

# Calibration corpus: broad coverage of recording conditions and languages.
# (dir, n_clips) — Common Voice first (the distribution that triggered the bug).
SOURCES = [
    ("cv_de", 25), ("cv_it", 25), ("cv_fr", 25), ("cv_en", 25),
    ("german", 10), ("italian", 10), ("french", 10),
    ("librispeech", 12), ("difficult", 12), ("earnings22", 8),
]

wavs = []
for name, n in SOURCES:
    manifest = ROOT / "data" / name / "manifest.json"
    if not manifest.exists():
        continue
    for entry in json.load(open(manifest))[:n]:
        p = ROOT / "data" / name / entry["audio_path"]
        if p.exists():
            wavs.append((name, str(p)))

print(f"Calibrating on {len(wavs)} utterances from "
      f"{len({n for n, _ in wavs})} datasets, headroom={HEADROOM}x")

all_scales = []
for i, (name, wav) in enumerate(wavs):
    r = subprocess.run([str(BINARY), "--model", "v3", wav],
                       capture_output=True, text=True, timeout=120,
                       env={**__import__("os").environ, "PARAKETTO_CALIB": "1"})
    lines = [l for l in r.stderr.split("\n") if l.startswith("SCALES:")]
    if len(lines) < 2:
        print(f"  WARN {name}: got {len(lines)} SCALES lines, skipping")
        continue
    # keep the audio pass (2nd line); skip the silence warmup (1st)
    all_scales.append([float(x) for x in lines[-1].split()[1:]])
    if (i + 1) % 25 == 0:
        print(f"  [{i+1}/{len(wavs)}]")

arr = np.array(all_scales)
assert arr.shape[1] == N_SITES, f"expected {N_SITES} sites, got {arr.shape[1]}"
print(f"Collected {arr.shape[0]} scale sets")

scales_max = arr.max(axis=0)
scales = scales_max * HEADROOM

# Report how tight the OLD single-ref calibration was vs the new max.
# sat = observed_max / old_scale; sat>1 means the old scale saturated.
print(f"\nPer-site max sat vs pre-headroom max: sites needing >1.0 headroom "
      f"are those the old calibration would clamp.")
print(f"  max scale (pre-headroom): {scales_max.max():.4e} at site {scales_max.argmax()}")
print(f"  applying uniform {HEADROOM}x headroom -> worst-case sat ~= {1.0/HEADROOM:.2f}")

out = ROOT / "scripts" / "v3_scales_new.txt"
with open(out, "w") as f:
    f.write(f"static const float FP8_BAKED_ACT_SCALES_V3[{N_SITES}] = {{\n")
    for i in range(0, N_SITES, 6):
        parts = []
        for j, v in enumerate(scales[i:i + 6]):
            idx = i + j
            suffix = "f," if idx < N_SITES - 1 else "f"
            parts.append(f"{v:.8e}{suffix}")
        line = "    " + " ".join(parts)
        blk, off = i // 9, i % 9
        if off == 0 and blk < 24:
            line += f"  // blk {blk}"
        elif i >= 216:
            line += "  // sub_out, enc_proj"
        f.write(line + "\n")
    f.write("};\n")
print(f"\nWrote new array to {out}")
