"""Score the Common Voice sample (data/cv_{de,it,fr}) with paraketto V3.

Reuses the identical multilingual normalizer + server harness as the FLEURS
bench (test_v3_multilingual), so numbers are directly comparable.

Usage: uv run python tests/test_cv_sample.py paraketto.cuda
"""
import sys
from pathlib import Path

import requests
from jiwer import wer as compute_wer

from bench_common import ROOT, DATA_DIR, load_manifest, print_results, server_process
from test_v3_multilingual import multilingual_normalize

DATASETS = ["cv_de", "cv_it", "cv_fr"]
LANG_CODES = {"cv_de": "de", "cv_it": "it", "cv_fr": "fr"}


def main(binary: Path, port: int = 18091) -> None:
    for name in DATASETS:
        if not (DATA_DIR / name / "manifest.json").exists():
            print(f"Missing {name}. Run scripts/make_cv_sample.py first.", file=sys.stderr)
            sys.exit(1)

    with server_process([str(binary), "--model", "v3", "--server", f":{port}"], port) as url:
        def transcribe(path: str) -> dict:
            with open(path, "rb") as f:
                r = requests.post(f"{url}/transcribe", files={"file": f})
            r.raise_for_status()
            return r.json()

        transcribe(load_manifest(DATASETS[0])[0]["audio_path"])  # warmup

        rows = []
        for name in DATASETS:
            manifest = load_manifest(name)
            audio_s = sum(e["duration_s"] for e in manifest)
            inf_s, refs, hyps = 0.0, [], []
            for e in manifest:
                res = transcribe(e["audio_path"])
                hyps.append(res["text"])
                refs.append(e["reference"])
                inf_s += res["inference_time_s"]
            lang = LANG_CODES[name]
            wer = compute_wer([multilingual_normalize(r, lang) for r in refs],
                              [multilingual_normalize(h, lang) for h in hyps]) * 100
            rows.append(dict(name=name, wer=wer, rtfx=audio_s / inf_s if inf_s else 0,
                             utts=len(manifest), audio_s=audio_s, inference_s=inf_s))
        print_results(rows)


if __name__ == "__main__":
    binary = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "paraketto.cuda")
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        sys.exit(1)
    main(binary)
