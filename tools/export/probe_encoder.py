"""Compare the Hugging Face ParakeetForTDT encoder (fp32, CPU) with FluidAudio's int8 Core ML
encoder on a real mel spectrogram, to establish the parity baseline for our own encoder export.

coremltools and torch crash when mixed in one process here, so this runs in two steps:

    uv run python probe_encoder.py coreml path/to/clip.wav   # mel + FluidAudio encoder output -> cache/encoder_probe_<clip>.npz
    uv run python probe_encoder.py hf <clip>                 # Hugging Face encoder on the same mel, compare
    uv run python probe_encoder.py ours <clip> <Encoder.mlpackage> [function]   # our export vs FluidAudio and HF
"""

import sys
import time
from pathlib import Path

import numpy as np

MODELS = Path.home() / "Library/Application Support/FluidAudio/Models/parakeet-ultra"
WINDOW = 240_000


def cache_path(clip: str) -> Path:
    return Path(f"cache/encoder_probe_{Path(clip).stem}.npz")


def step_coreml(path: Path):
    import coremltools as ct
    import soundfile

    samples, rate = soundfile.read(path, dtype="float32")
    assert rate == 16_000
    samples = samples[:WINDOW]
    frames = -(-len(samples) // 1280)
    declared = min(frames * 1280, WINDOW)
    padded = np.zeros((1, WINDOW), dtype=np.float32)
    padded[0, : len(samples)] = samples

    preprocessor = ct.models.CompiledMLModel(str(MODELS / "Preprocessor.mlmodelc"), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = preprocessor.predict({"audio_signal": padded, "audio_length": np.array([declared], dtype=np.int32)})
    mel, mel_length = out["mel"], int(out["mel_length"][0])
    print(f"mel {mel.shape}, mel_length {mel_length}, encoder frames {frames}")

    encoder_cm = ct.models.CompiledMLModel(str(MODELS / "Encoder.mlmodelc"), compute_units=ct.ComputeUnit.CPU_AND_NE)
    t = time.perf_counter()
    cm = encoder_cm.predict({"mel": mel, "mel_length": np.array([mel_length], dtype=np.int32)})
    print(f"coreml encoder: {(time.perf_counter() - t) * 1000:.0f} ms, output {cm['encoder'].shape}, length {int(cm['encoder_length'][0])}")
    cache = cache_path(path.name)
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, mel=mel, mel_length=mel_length, frames=frames, encoder=cm["encoder"])
    print(f"saved {cache}")


def compare(name: str, a: np.ndarray, b: np.ndarray, frames: int):
    """Per-frame agreement of two [T, 1024] encoder outputs over the valid frames."""
    n = min(len(a), len(b), frames)
    a, b = a[:n], b[:n]
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    print(f"{name}: {n} frames, mean |diff| {np.abs(a - b).mean():.4f}, max |diff| {np.abs(a - b).max():.3f}, "
          f"cosine min/mean {cos.min():.4f}/{cos.mean():.4f}")


def frames_first(encoder: np.ndarray) -> np.ndarray:
    e = encoder[0]
    return e.T if e.shape[0] == 1024 else e


def step_hf(clip: str):
    import torch
    from transformers import ParakeetForTDT

    saved = np.load(cache_path(clip))
    mel, mel_length, frames, encoder = saved["mel"], int(saved["mel_length"]), int(saved["frames"]), saved["encoder"]
    cm = {"encoder": encoder}
    model = ParakeetForTDT.from_pretrained("moondream/parakeet-ultra", dtype=torch.float32).eval()
    print("hf model loaded; encoder type:", type(model.encoder).__name__)
    print("has vad head:", hasattr(model, "vad_head"))
    # HF expects input_features [B, T_mel, 128] over the valid frames only.
    features = torch.from_numpy(mel[0, :, :mel_length].T.copy()).unsqueeze(0)
    with torch.no_grad():
        t = time.perf_counter()
        hf = model.encoder(input_features=features).last_hidden_state[0].numpy()  # [T', 1024]
        print(f"hf encoder (fp32 cpu): {(time.perf_counter() - t) * 1000:.0f} ms, output {hf.shape}")

    np.save(cache_path(clip).with_suffix(".hf.npy"), hf)
    compare("hf fp32 vs fluidaudio int8", hf, frames_first(cm["encoder"]), frames)


def step_ours(clip: str, package: str, function: str | None):
    import coremltools as ct

    saved = np.load(cache_path(clip))
    mel, mel_length, frames = saved["mel"], int(saved["mel_length"]), int(saved["frames"])
    fluid = frames_first(saved["encoder"])
    hf_path = cache_path(clip).with_suffix(".hf.npy")
    hf = np.load(hf_path) if hf_path.exists() else None
    kwargs = {"function_name": function} if function else {}
    model = ct.models.MLModel(package, compute_units=ct.ComputeUnit.CPU_AND_NE, **kwargs)
    description = model.get_spec().description
    inputs = description.input or next(f.input for f in description.functions if f.name == (function or description.defaultFunctionName))
    window = next(i for i in inputs if i.name == "mel").type.multiArrayType.shape[2]
    assert mel_length <= window, f"clip needs {mel_length} mel frames, function has {window}"
    mel_in = mel[:, :, :window].astype(np.float32)
    for _ in range(2):
        t = time.perf_counter()
        out = model.predict({"mel": mel_in, "mel_length": np.array([mel_length], dtype=np.int32)})
        elapsed = (time.perf_counter() - t) * 1000
    ours = frames_first(out["encoder"])
    print(f"ours ({function or 'default'}, {window} mel frames): {elapsed:.0f} ms, output {out['encoder'].shape}, length {int(out['encoder_length'][0])}")
    compare("ours vs fluidaudio int8", ours, fluid, frames)
    if hf is not None:
        compare("ours vs hf fp32", ours, hf, frames)


if __name__ == "__main__":
    if sys.argv[1] == "coreml":
        step_coreml(Path(sys.argv[2]))
    elif sys.argv[1] == "hf":
        step_hf(sys.argv[2])
    else:
        step_ours(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
