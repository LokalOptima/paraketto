"""Parakeet TDT 0.6B speech-to-text with TensorRT acceleration.

Handles model loading and transcription. TensorRT engines are compiled on first
inference and cached at PARAKEET_TRT_CACHE_DIR (default ~/.cache/parakeet-trt-cache/).
Set PARAKEET_MODEL_PATH to use a local model directory instead of downloading from HF.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger("parakeet_trt")

# Suppress ONNX Runtime warnings (memcpy / EP assignment noise)
os.environ["ORT_LOG_LEVEL"] = "3"

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v2"
MODEL_PATH = os.environ.get("PARAKEET_MODEL_PATH")
TRT_CACHE_DIR = Path(os.environ.get("PARAKEET_TRT_CACHE_DIR", Path.home() / ".cache" / "parakeet-trt-cache"))

_model = None


def load_model():
    """Load the Parakeet TDT model with TensorRT EP.

    First run compiles TRT engines (~47s). Subsequent runs load from cache (~3s).
    """
    global _model

    import onnx_asr
    from onnx_asr.onnx import TensorRtOptions

    # batch=1 (no batching), up to 120s audio segments.
    # Default batch=16 wastes ~5GB VRAM on pre-allocated activation buffers.
    TensorRtOptions.profile_max_shapes = {"batch": 1, "waveform_len_ms": 120_000}
    TensorRtOptions.profile_opt_shapes = {"batch": 1, "waveform_len_ms": 20_000}

    TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    providers = [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(TRT_CACHE_DIR),
            "trt_builder_optimization_level": "5",
            "trt_auxiliary_streams": "0",
        }),
        ("CUDAExecutionProvider", {}),
    ]

    logger.info("Loading %s with TensorRT...", MODEL_NAME)
    t0 = time.monotonic()
    # Suppress C-level ONNX Runtime/TensorRT warnings written directly to stderr
    # (Int64 bindings, engine plan files, EP node assignment noise)
    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        _model = onnx_asr.load_model(MODEL_NAME, path=MODEL_PATH, providers=providers)
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
    elapsed = time.monotonic() - t0
    logger.info("Model loaded in %.1fs", elapsed)


def transcribe(audio: np.ndarray, sample_rate: int) -> str:
    """Transcribe a numpy audio array to text."""
    if _model is None:
        raise RuntimeError("STT model not loaded — call load_model() first")

    # Mono conversion
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return (_model.recognize(audio, sample_rate=sample_rate) or "").strip()


def transcribe_file(path: str | Path) -> str:
    """Transcribe a WAV/FLAC/OGG file to text."""
    audio, sr = sf.read(str(path), dtype="float32")
    return transcribe(audio, sr)
