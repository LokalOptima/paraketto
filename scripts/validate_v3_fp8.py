"""Large-scale held-out validation of V3 FP8: stream the full clean Common Voice
test sets (de/it/fr), transcribe via the server, count empties + WER.

Skips the first 110 clean clips per language (used for calibration/earlier tests)
so this is genuinely held out. Streams+decodes in memory — no wav files written.

Usage: uv run --with datasets==2.19.0 --with soundfile>=0.13 --with soxr \
           --with jiwer --with num2words --with requests scripts/validate_v3_fp8.py [CAP]
"""
import io, sys, os
from pathlib import Path

import soundfile as sf
import soxr
import requests
from datasets import Audio, load_dataset
from jiwer import wer as compute_wer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from bench_common import ROOT, server_process
from test_v3_multilingual import multilingual_normalize

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
SKIP = 110  # clips used for calibration / earlier validation
LANGS = [("de", "de"), ("it", "it"), ("fr", "fr")]


def clean(ex):
    def i(x):
        try:
            return int(x)
        except Exception:
            return 0
    return i(ex.get("up_votes")) >= 2 and i(ex.get("down_votes")) == 0


def main():
    with server_process([str(ROOT / "paraketto.fp8"), "--model", "v3",
                         "--server", ":18120"], 18120) as url:
        def transcribe(buf):
            r = requests.post(f"{url}/transcribe",
                              files={"file": ("clip.wav", buf)})
            r.raise_for_status()
            return r.json()["text"]

        grand_empty = grand_n = 0
        for lang, code in LANGS:
            ds = load_dataset("fsicoli/common_voice_17_0", lang, split="test",
                              streaming=True, trust_remote_code=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            refs, hyps, empty, seen, n = [], [], 0, 0, 0
            for ex in ds:
                if not clean(ex):
                    continue
                seen += 1
                if seen <= SKIP:
                    continue
                b = ex["audio"]["bytes"] or open(ex["audio"]["path"], "rb").read()
                a, sr = sf.read(io.BytesIO(b), dtype="float32")
                if a.ndim > 1:
                    a = a.mean(axis=1)
                if sr != 16000:
                    a = soxr.resample(a, sr, 16000)
                buf = io.BytesIO()
                sf.write(buf, a, 16000, subtype="PCM_16", format="WAV")
                buf.seek(0)
                h = transcribe(buf)
                hyps.append(h)
                refs.append(ex["sentence"])
                if not multilingual_normalize(h, code).strip():
                    empty += 1
                n += 1
                if n % 500 == 0:
                    w = compute_wer([multilingual_normalize(r, code) for r in refs],
                                    [multilingual_normalize(x, code) for x in hyps]) * 100
                    print(f"[{lang}] n={n} empties={empty} WER={w:.2f}%", flush=True)
                if n >= CAP:
                    break
            w = compute_wer([multilingual_normalize(r, code) for r in refs],
                            [multilingual_normalize(x, code) for x in hyps]) * 100
            print(f"=== {lang}: n={n} empties={empty} "
                  f"({100*empty/max(n,1):.3f}%) WER={w:.2f}% ===", flush=True)
            grand_empty += empty
            grand_n += n
        print(f"\n### TOTAL: n={grand_n} empties={grand_empty} "
              f"({100*grand_empty/max(grand_n,1):.4f}%) ###", flush=True)


if __name__ == "__main__":
    main()
