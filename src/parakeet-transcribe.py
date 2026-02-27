#!/usr/bin/env python3
"""Chunked ASR transcription using Parakeet V3 via onnx-asr.

Reads raw PCM audio (S16_LE, 16kHz, mono) from stdin, transcribes in
fixed-size chunks, and prints results to stdout.

Usage:
    arecord -f S16_LE -r 16000 -c 1 -t raw - | python src/parakeet-transcribe.py
    ffmpeg -i audio.wav -f s16le -ar 16000 -ac 1 - | python src/parakeet-transcribe.py
"""

import argparse
import sys
import tempfile
import wave

import numpy as np
import onnx_asr

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # S16_LE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parakeet V3 chunked transcription")
    p.add_argument(
        "--model",
        default="nemo-parakeet-tdt-0.6b-v2",
        help="onnx-asr model name (default: nemo-parakeet-tdt-0.6b-v2)",
    )
    p.add_argument(
        "--quantization",
        default="int8",
        choices=["int8", "fp16", "fp32"],
        help="Model quantization (default: int8)",
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=15.0,
        help="Audio chunk duration in seconds (default: 15)",
    )
    p.add_argument("--cuda", action="store_true", help="Use CUDA execution provider")
    p.add_argument(
        "--tensorrt", action="store_true", help="Use TensorRT execution provider"
    )
    return p.parse_args()


def build_providers(args: argparse.Namespace) -> list | None:
    if args.tensorrt:
        import tensorrt_libs  # noqa: F401 — needed to register TRT libs

        return [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_max_workspace_size": 6 * 1024**3,
                    "trt_fp16_enable": True,
                },
            )
        ]
    if args.cuda:
        return [("CUDAExecutionProvider", {})]
    return None


def pcm_to_wav_path(pcm: bytes) -> str:
    """Write raw PCM bytes to a temporary WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return tmp.name


def main() -> None:
    args = parse_args()
    chunk_bytes = int(args.chunk_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)

    providers = build_providers(args)
    kwargs = {}
    if providers is not None:
        kwargs["providers"] = providers

    print(f"Loading {args.model} (quantization={args.quantization})...", file=sys.stderr)
    model = onnx_asr.load_model(
        args.model, quantization=args.quantization, **kwargs
    )
    print("Model loaded, listening...", file=sys.stderr)

    buf = b""
    try:
        while True:
            data = sys.stdin.buffer.read(chunk_bytes - len(buf))
            if not data:
                break
            buf += data
            if len(buf) < chunk_bytes:
                continue

            wav_path = pcm_to_wav_path(buf)
            text = model.recognize(wav_path)
            buf = b""

            if text and text.strip():
                print(text.strip(), flush=True)
    except KeyboardInterrupt:
        pass

    # Flush remaining audio
    if len(buf) >= SAMPLE_RATE * BYTES_PER_SAMPLE:  # at least 1 second
        wav_path = pcm_to_wav_path(buf)
        text = model.recognize(wav_path)
        if text and text.strip():
            print(text.strip(), flush=True)


if __name__ == "__main__":
    main()
