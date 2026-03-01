"""Benchmark the parakeet.cuda binary: WER and RTFx.

Starts the parakeet.cuda HTTP server, sends all utterances from
data/{librispeech,earnings22,long}/manifest.json via /transcribe,
reports per-dataset WER and RTFx.

Usage:
    uv run python tests/bench_cuda.py
"""

import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from jiwer import wer

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "parakeet.cuda"
SERVER_URL = "http://localhost:18080"

from bench_common import DATASETS, load_manifest, normalize, print_results


def transcribe(path: str) -> dict:
    with open(path, "rb") as f:
        r = requests.post(f"{SERVER_URL}/transcribe", files={"file": f})
    r.raise_for_status()
    return r.json()


def wait_for_server(timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=1)
            if r.ok:
                return
        except requests.ConnectionError:
            pass
        time.sleep(0.1)
    print("Server failed to start", file=sys.stderr)
    sys.exit(1)


def main():
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}", file=sys.stderr)
        print("Run 'make parakeet.cuda' first.", file=sys.stderr)
        sys.exit(1)

    # Start server
    server = subprocess.Popen(
        [str(BINARY), "--server", ":18080"],
        stderr=subprocess.PIPE,
    )
    try:
        wait_for_server()

        # Warmup
        manifest = load_manifest(DATASETS[0])
        transcribe(manifest[0]["audio_path"])

        # Bench each dataset
        rows = []
        for name in DATASETS:
            manifest = load_manifest(name)
            ds_audio = sum(e["duration_s"] for e in manifest)
            ds_inference = 0.0
            references = []
            hypotheses = []

            for entry in manifest:
                result = transcribe(entry["audio_path"])
                hypotheses.append(result["text"])
                references.append(entry["reference"])
                ds_inference += result["inference_time_s"]

            wer_pct = wer(
                references, hypotheses,
                reference_transform=normalize,
                hypothesis_transform=normalize,
            ) * 100
            rtfx = ds_audio / ds_inference if ds_inference > 0 else 0
            rows.append(dict(
                name=name, wer=wer_pct, rtfx=rtfx,
                utts=len(manifest), audio_s=ds_audio, inference_s=ds_inference,
            ))

        print_results(rows)

    finally:
        server.send_signal(signal.SIGINT)
        server.wait(timeout=5)


if __name__ == "__main__":
    main()
