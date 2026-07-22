"""Download a small clean Common Voice sample for de/it/fr into data/cv_<lang>/.

Clips are filtered to up_votes>=2 and down_votes==0 (human-verified clean reads),
decoded from mp3, resampled to 16 kHz mono, written as wav + manifest.json in the
same format the paraketto benchmarks use.

Usage: uv run --with datasets==2.19.0 --with soundfile>=0.13 --with soxr --with numpy \
           scripts/make_cv_sample.py [N_PER_LANG]
"""
import io, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from datasets import Audio, load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
LANGS = {"de": "cv_de", "it": "cv_it", "fr": "cv_fr"}
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def is_clean(ex) -> bool:
    def i(x):
        try:
            return int(x)
        except Exception:
            return 0
    return i(ex.get("up_votes")) >= 2 and i(ex.get("down_votes")) == 0


for code, dirname in LANGS.items():
    out = DATA / dirname
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("fsicoli/common_voice_17_0", code, split="test",
                      streaming=True, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    manifest, kept = [], 0
    for ex in ds:
        if not is_clean(ex):
            continue
        b = ex["audio"]["bytes"] or open(ex["audio"]["path"], "rb").read()
        arr, sr = sf.read(io.BytesIO(b), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            arr = soxr.resample(arr, sr, 16000)
        cid = f"{code}-{kept:04d}"
        sf.write(out / f"{cid}.wav", arr, 16000, subtype="PCM_16")
        manifest.append(dict(id=cid, audio_path=f"{cid}.wav",
                             reference=ex["sentence"],
                             duration_s=round(len(arr) / 16000, 3)))
        kept += 1
        if kept >= N:
            break
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"{dirname}: wrote {kept} clips -> {out}")
