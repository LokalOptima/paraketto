r"""Compare V3-compatible weight files/backends on identical local audio.

Usage:
  uv run python tests/bench_weights.py --weights paraketto-ultra-fp16.bin \
      --baseline ~/.cache/paraketto/paraketto-v3-fp16.bin \
      --output scratch/ultra-benchmark.json

Reuses the project's English and multilingual WER normalizers. Timing is a
single warmed pass per model, useful for context, not a performance regression
test. The JSON retains references and hypotheses for inspection.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from jiwer import wer

from bench_common import ROOT, load_manifest, print_results, server_process, whisper_normalize
from test_v3_multilingual import multilingual_normalize

LANGUAGES = {
    "german": "de", "italian": "it", "french": "fr",
    "fleurs_german": "de", "fleurs_italian": "it", "fleurs_french": "fr",
    "cv_de_heldout": "de", "cv_it_heldout": "it", "cv_fr_heldout": "fr",
}
DEFAULT_DATASETS = ["librispeech", "earnings22", "long", "difficult", "german", "italian", "french"]


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark(binary, weights, datasets, port):
    result = {"weights": str(weights), "sha256": checksum(weights), "datasets": [],
              "binary": str(binary), "binary_sha256": checksum(binary)}
    cmd = [str(binary), "--model", "v3", "--weights", str(weights),
           "--server", f"127.0.0.1:{port}"]
    with server_process(cmd, port) as url, requests.Session() as session:
        response = session.get(f"{url}/info", timeout=10)
        response.raise_for_status()
        result["server"] = response.json()
        if result["server"].get("weights") != str(weights):
            raise RuntimeError("Benchmark connected to a server with different weights")

        def transcribe(path):
            with open(path, "rb") as f:
                response = session.post(f"{url}/transcribe", files={"file": f}, timeout=120)
            response.raise_for_status()
            return response.json()

        transcribe(next(iter(datasets.values()))[0]["audio_path"])
        for name, manifest in datasets.items():
            lang = LANGUAGES.get(name)
            normalize = (lambda s: multilingual_normalize(s, lang)) if lang else whisper_normalize
            clips = []
            for entry in manifest:
                response = transcribe(entry["audio_path"])
                clips.append({
                    "audio_path": entry["audio_path"], "reference": entry["reference"],
                    "hypothesis": response["text"], "audio_s": entry["duration_s"],
                    "inference_s": response["inference_time_s"],
                })
            references = [normalize(c["reference"]) for c in clips]
            hypotheses = [normalize(c["hypothesis"]) for c in clips]
            inference = sum(c["inference_s"] for c in clips)
            audio = sum(c["audio_s"] for c in clips)
            row = dict(name=name, wer=wer(references, hypotheses) * 100,
                       rtfx=audio / inference if inference else 0, utts=len(clips),
                       audio_s=audio, inference_s=inference,
                       empty=sum(bool(r) and not h for r, h in zip(references, hypotheses)),
                       clips=clips)
            result["datasets"].append(row)
            print(f"  {name}: WER {row['wer']:.2f}%, {row['rtfx']:.0f}x RTFx, "
                  f"{row['empty']} empty / {len(clips)} clips", flush=True)
    print_results(result["datasets"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=ROOT / "paraketto.cuda")
    parser.add_argument("--baseline-binary", type=Path, help="Baseline backend (defaults to --binary)")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18140)
    args = parser.parse_args()
    binary = args.binary.resolve()
    baseline_binary = args.baseline_binary.resolve() if args.baseline_binary else binary
    for path in (binary, baseline_binary, args.weights, args.baseline):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    datasets = {name: load_manifest(name) for name in args.datasets}
    for name, manifest in datasets.items():
        if not manifest or not all(Path(e["audio_path"]).is_file() for e in manifest):
            parser.error(f"Empty manifest or missing audio in {name}")
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "binary": str(binary), "binary_sha256": checksum(binary),
              "timing": "Single warmed pass per model; no statistical speed comparison."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, weights, backend in (("baseline", args.baseline, baseline_binary),
                                    ("candidate", args.weights, binary)):
        print(f"\n{label}: {weights}", flush=True)
        report[label] = benchmark(backend, weights.resolve(), datasets, args.port)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved comparison and transcripts to {args.output}")


if __name__ == "__main__":
    main()
