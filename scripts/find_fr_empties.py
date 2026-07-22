"""Re-stream French CV, find the FP8 empties, save them, and check FP16.

Determines whether the 7 French empties are the scales bug (FP16 transcribes
them) or legitimate declines (FP16 also empty).
"""
import io, sys
from pathlib import Path

import soundfile as sf
import soxr
import requests
from datasets import Audio, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from bench_common import ROOT, server_process
from test_v3_multilingual import multilingual_normalize

SKIP = 110
OUT = ROOT / "data" / "fr_empties"
OUT.mkdir(exist_ok=True)


def is_clean(ex):
    def i(x):
        try:
            return int(x)
        except Exception:
            return 0
    return i(ex.get("up_votes")) >= 2 and i(ex.get("down_votes")) == 0


def txt_of(url, wavbytes):
    r = requests.post(f"{url}/transcribe", files={"file": ("c.wav", io.BytesIO(wavbytes))})
    r.raise_for_status()
    return r.json()["text"]


# Pass 1: FP8, find + save empties
empties = []
with server_process([str(ROOT / "paraketto.fp8"), "--model", "v3", "--server", ":18130"], 18130) as url:
    ds = load_dataset("fsicoli/common_voice_17_0", "fr", split="test",
                      streaming=True, trust_remote_code=True).cast_column("audio", Audio(decode=False))
    seen = n = 0
    for ex in ds:
        if not is_clean(ex):
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
        buf = io.BytesIO(); sf.write(buf, a, 16000, subtype="PCM_16", format="WAV")
        wavbytes = buf.getvalue()
        h = txt_of(url, wavbytes)
        if not multilingual_normalize(h, "fr").strip():
            import numpy as np
            rms = float(np.sqrt((a ** 2).mean())); pk = float(np.abs(a).max())
            fn = OUT / f"fr_empty_{len(empties)}.wav"
            open(fn, "wb").write(wavbytes)
            empties.append((fn, ex["sentence"], a.size / 16000, rms, pk))
            print(f"EMPTY #{len(empties)} at n={n}: dur={a.size/16000:.1f}s rms={rms:.4f} "
                  f"pk={pk:.3f} ref={ex['sentence'][:60]!r}", flush=True)
        n += 1
    print(f"\nScanned {n} French clips, found {len(empties)} FP8 empties", flush=True)

# Pass 2: FP16 on the saved empties
print("\n--- FP16 on the same clips ---", flush=True)
with server_process([str(ROOT / "paraketto.cuda"), "--model", "v3", "--server", ":18131"], 18131) as url:
    fp16_empty = 0
    for fn, ref, dur, rms, pk in empties:
        h = txt_of(url, open(fn, "rb").read())
        empty16 = not multilingual_normalize(h, "fr").strip()
        fp16_empty += empty16
        print(f"{fn.name}: FP16={'EMPTY' if empty16 else repr(h)}  (rms={rms:.4f} pk={pk:.3f})  ref={ref[:50]!r}", flush=True)
    print(f"\nOf {len(empties)} FP8 empties, {fp16_empty} are ALSO empty on FP16 "
          f"(legit declines), {len(empties)-fp16_empty} are FP8-only (scales bug)", flush=True)
