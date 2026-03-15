"""Benchmark V3 multilingual transcription on de/it/fr test sets.

Uses curated FLEURS clips in data/{german,italian,french}/ (same format
as English benchmarks). Download with: make download-data-v3

Usage:
    uv run python tests/test_v3_multilingual.py paraketto.fp8
"""

import json
import re
import signal
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import requests
from jiwer import wer as compute_wer

from bench_common import ROOT, DATA_DIR, load_manifest, print_results

DATASETS_V3 = ["german", "italian", "french"]


# HuggingFace Open ASR Leaderboard multilingual normalizer
# (vendored from github.com/huggingface/open_asr_leaderboard/normalizer/normalizer.py)
def _remove_symbols_and_diacritics(s: str) -> str:
    return "".join(
        "" if unicodedata.category(c) == "Mn"
        else " " if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )

def multilingual_normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
    s = re.sub(r"\(([^)]+?)\)", "", s)
    s = _remove_symbols_and_diacritics(s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def bench_v3_server(binary: Path, port: int = 18090) -> None:
    """Start V3 server, benchmark de/it/fr, print results."""
    for name in DATASETS_V3:
        if not (DATA_DIR / name / "manifest.json").exists():
            print(f"Benchmark data not found. Run 'make download-data-v3' first.",
                  file=sys.stderr)
            sys.exit(1)

    server = subprocess.Popen(
        [str(binary), "--model", "v3", "--server", f":{port}"],
        stderr=subprocess.PIPE,
    )

    def transcribe(path: str) -> dict:
        with open(path, "rb") as f:
            r = requests.post(f"http://localhost:{port}/transcribe", files={"file": f})
        r.raise_for_status()
        return r.json()

    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if server.poll() is not None:
                err = server.stderr.read().decode() if server.stderr else ""
                print(f"Server exited: {err}", file=sys.stderr)
                sys.exit(1)
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=1)
                if r.ok:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.1)
        else:
            print("Server timeout", file=sys.stderr)
            sys.exit(1)

        # Warmup
        manifest = load_manifest(DATASETS_V3[0])
        transcribe(manifest[0]["audio_path"])

        # Bench each language
        rows = []
        for name in DATASETS_V3:
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

            wer_pct = compute_wer(
                [multilingual_normalize(r) for r in references],
                [multilingual_normalize(h) for h in hypotheses],
            ) * 100
            rtfx = ds_audio / ds_inference if ds_inference > 0 else 0
            rows.append(dict(
                name=name, wer=wer_pct, rtfx=rtfx,
                utts=len(manifest), audio_s=ds_audio, inference_s=ds_inference,
            ))

        print_results(rows)

    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: test_v3_multilingual.py <binary>", file=sys.stderr)
        sys.exit(1)
    binary = ROOT / sys.argv[1]
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        sys.exit(1)
    bench_v3_server(binary)
