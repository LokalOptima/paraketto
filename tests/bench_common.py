"""Shared utilities for benchmark scripts."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from jiwer import Compose, ReduceToListOfListOfWords, RemovePunctuation, ToLowerCase
from jiwer import wer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DATASETS = ["librispeech", "earnings22", "long", "difficult"]

normalize = Compose([ToLowerCase(), RemovePunctuation(), ReduceToListOfListOfWords()])


def load_manifest(name: str) -> list[dict]:
    manifest_path = DATA_DIR / name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        entry["audio_path"] = str(DATA_DIR / name / entry["audio_path"])
    return manifest


def print_results(rows: list[dict]) -> None:
    """Print benchmark results as a Unicode box table.

    Each row dict has: name, wer, rtfx, utts, audio_s, inference_s.
    """
    grand_audio = sum(r["audio_s"] for r in rows)
    grand_inference = sum(r["inference_s"] for r in rows)
    grand_rtfx = grand_audio / grand_inference if grand_inference > 0 else 0

    # Format inference time
    def fmt_time(s):
        return f"{s * 1000:.0f}ms" if s < 1 else f"{s:.2f}s"

    # Column widths
    W_NAME = max(len(r["name"]) for r in rows)
    W_NAME = max(W_NAME, len("Total"), len("Dataset"))
    W_WER = 8
    W_RTFX = 7
    W_UTTS = 6
    W_AUDIO = 7
    W_TIME = 8

    def sep(left, mid, right, fill="─"):
        return (f"{left}{fill * (W_NAME + 2)}{mid}{fill * (W_WER + 2)}"
                f"{mid}{fill * (W_RTFX + 2)}{mid}{fill * (W_UTTS + 2)}"
                f"{mid}{fill * (W_AUDIO + 2)}{mid}{fill * (W_TIME + 2)}{right}")

    def row(name, wer_s, rtfx_s, utts_s, audio_s, time_s):
        return (f"│ {name:<{W_NAME}} │ {wer_s:>{W_WER}} │ {rtfx_s:>{W_RTFX}} │"
                f" {utts_s:>{W_UTTS}} │ {audio_s:>{W_AUDIO}} │ {time_s:>{W_TIME}} │")

    print(sep("┌", "┬", "┐"))
    print(row("Dataset", "WER", "RTFx", "Utts", "Audio", "Time"))
    print(sep("├", "┼", "┤"))
    for r in rows:
        print(row(
            r["name"],
            f"{r['wer']:.2f}%",
            f"{r['rtfx']:.0f}x",
            str(r["utts"]),
            f"{r['audio_s']:.0f}s",
            fmt_time(r["inference_s"]),
        ))
    print(sep("├", "┼", "┤"))
    print(row("Total", "", f"{grand_rtfx:.0f}x", str(sum(r["utts"] for r in rows)),
              f"{grand_audio:.0f}s", fmt_time(grand_inference)))
    print(sep("└", "┴", "┘"))


def bench_server(binary: Path, binary_name: str, port: int = 18080) -> None:
    """Start a server binary, run benchmarks, and print results."""
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        print(f"Run 'make {binary_name}' first.", file=sys.stderr)
        sys.exit(1)

    server_url = f"http://localhost:{port}"

    def transcribe(path: str) -> dict:
        with open(path, "rb") as f:
            r = requests.post(f"{server_url}/transcribe", files={"file": f})
        r.raise_for_status()
        return r.json()

    def wait_for_server(timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ret = server.poll()
            if ret is not None:
                err = server.stderr.read().decode() if server.stderr else ""
                print(f"Server exited with code {ret}", file=sys.stderr)
                if err:
                    print(err, file=sys.stderr, end="")
                sys.exit(1)
            try:
                r = requests.get(f"{server_url}/health", timeout=1)
                if r.ok:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(0.1)
        print("Server failed to start (timeout)", file=sys.stderr)
        sys.exit(1)

    server = subprocess.Popen(
        [str(binary), "--server", f":{port}"],
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
