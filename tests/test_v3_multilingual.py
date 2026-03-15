"""Test V3 multilingual transcription on FLEURS test set.

Downloads clips from Google FLEURS via HF hub, converts to 16kHz WAV,
transcribes with paraketto V3, and computes WER.

Usage:
    uv run python tests/test_v3_multilingual.py paraketto.fp8
"""

import json
import signal
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from jiwer import wer as compute_wer
from whisper_normalizer.basic import BasicTextNormalizer

whisper_normalize = BasicTextNormalizer()

ROOT = Path(__file__).resolve().parent.parent
N_CLIPS = 50

# FLEURS stores audio in tar.gz shards. The TSV has wav filenames + transcriptions.
# We download the TSV index + first audio shard per language.
FLEURS_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main"
LANGUAGES = {
    "de_de": "German",
    "it_it": "Italian",
    "fr_fr": "French",
}


def start_server(binary: Path, port: int = 18090) -> subprocess.Popen:
    server = subprocess.Popen(
        [str(binary), "--model", "v3", "--server", f":{port}"],
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server.poll() is not None:
            stderr = server.stderr.read().decode() if server.stderr else ""
            print(f"Server exited: {stderr}", file=sys.stderr)
            sys.exit(1)
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=1)
            if r.ok:
                return server
        except requests.ConnectionError:
            pass
        time.sleep(0.1)
    print("Server timeout", file=sys.stderr)
    sys.exit(1)


def stop_server(server: subprocess.Popen):
    server.send_signal(signal.SIGINT)
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def transcribe(wav_path: str, port: int = 18090) -> str:
    with open(wav_path, "rb") as f:
        r = requests.post(f"http://localhost:{port}/transcribe", files={"file": f})
    r.raise_for_status()
    return r.json()["text"]


def download_fleurs_clips(lang_code: str, n_clips: int, tmpdir: Path) -> list[dict]:
    """Download FLEURS test clips: TSV index + audio tarball."""

    # Download TSV index
    tsv_url = f"{FLEURS_BASE}/data/{lang_code}/test.tsv"
    r = requests.get(tsv_url, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().split("\n")

    # Parse TSV: id, filename, raw_transcription, transcription, ...
    entries = []
    seen_texts = set()
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        wav_name = parts[1]
        reference = parts[2]  # raw transcription (with punctuation)
        # Deduplicate (FLEURS has multiple recordings per sentence)
        if reference in seen_texts:
            continue
        seen_texts.add(reference)
        entries.append({"wav_name": wav_name, "reference": reference})
        if len(entries) >= n_clips:
            break

    # Download audio tar.gz (contains all test WAVs)
    audio_url = f"{FLEURS_BASE}/data/{lang_code}/audio/test.tar.gz"
    print(f"  Downloading audio from {audio_url}...", file=sys.stderr)
    r = requests.get(audio_url, timeout=120, stream=True)
    r.raise_for_status()

    tar_path = tmpdir / f"{lang_code}_test.tar.gz"
    with open(tar_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    # Extract needed WAVs
    needed = {e["wav_name"] for e in entries}
    wav_dir = tmpdir / lang_code
    wav_dir.mkdir(exist_ok=True)

    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            basename = Path(member.name).name
            if basename in needed:
                member.name = basename  # flatten
                tar.extract(member, wav_dir)

    # Build clip list with resolved paths
    clips = []
    for entry in entries:
        wav_path = wav_dir / entry["wav_name"]
        if wav_path.exists():
            # Resample to 16kHz if needed
            audio, sr = sf.read(wav_path)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
                sf.write(wav_path, audio.astype(np.float32), 16000)
            clips.append({"wav_path": str(wav_path), "reference": entry["reference"]})

    return clips


def main():
    if len(sys.argv) < 2:
        print("Usage: test_v3_multilingual.py <binary>", file=sys.stderr)
        sys.exit(1)

    binary = ROOT / sys.argv[1]
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    print("Starting server...", file=sys.stderr)
    server = start_server(binary)

    try:
        for lang_code, lang_name in LANGUAGES.items():
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"  {lang_name} ({lang_code}) — {N_CLIPS} clips from FLEURS test",
                  file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            with tempfile.TemporaryDirectory() as tmpdir:
                clips = download_fleurs_clips(lang_code, N_CLIPS, Path(tmpdir))
                print(f"  Got {len(clips)} clips", file=sys.stderr)

                references = []
                hypotheses = []

                for i, clip in enumerate(clips):
                    hyp = transcribe(clip["wav_path"])
                    ref = clip["reference"]
                    references.append(ref)
                    hypotheses.append(hyp)

                    print(f"\n  [{i+1}/{len(clips)}]", file=sys.stderr)
                    print(f"  REF: {ref}", file=sys.stderr)
                    print(f"  HYP: {hyp}", file=sys.stderr)

            if references:
                wer_pct = compute_wer(
                    [whisper_normalize(r) for r in references],
                    [whisper_normalize(h) for h in hypotheses],
                ) * 100
                print(f"\n  {lang_name} WER: {wer_pct:.1f}% ({len(clips)} clips)",
                      file=sys.stderr)

    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
