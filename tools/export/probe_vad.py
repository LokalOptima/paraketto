"""Run the exported VAD head on WAV files and print per-frame speech probabilities.

Feeds each file through FluidAudio's compiled Preprocessor model (the same mel the app
computes), then through VadHead.mlpackage, and prints an 80 ms-per-character strip:
'#' for p >= 0.5, '.' below. Useful to check that leading/trailing silence and pauses
read as pauses on real speech.

Usage:
    uv run python probe_vad.py --vad ../../Resources/VadHead.mlpackage file1.wav [file2.wav ...]
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile

PREPROCESSOR = Path.home() / "Library/Application Support/FluidAudio/Models/parakeet-ultra/Preprocessor.mlmodelc"
WINDOW_SAMPLES = 240_000
SAMPLES_PER_FRAME = 1280


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vad", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    preprocessor = ct.models.CompiledMLModel(str(PREPROCESSOR), compute_units=ct.ComputeUnit.CPU_ONLY)
    vad = ct.models.MLModel(str(args.vad), compute_units=ct.ComputeUnit.ALL)

    for path in args.files:
        samples, rate = soundfile.read(path, dtype="float32")
        assert rate == 16_000 and samples.ndim == 1, f"{path}: need 16 kHz mono"
        samples = samples[:WINDOW_SAMPLES]
        padded = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
        padded[0, : len(samples)] = samples
        mel = preprocessor.predict(
            {"audio_signal": padded, "audio_length": np.array([len(samples)], dtype=np.int32)}
        )["mel"]
        probabilities = vad.predict({"mel": mel.astype(np.float32)})["speech_probability"][0]
        frames = len(samples) // SAMPLES_PER_FRAME
        p = probabilities[:frames]
        strip = "".join("#" if v >= args.threshold else "." for v in p)
        print(f"{path.name}: {len(samples) / 16000:.1f}s, {frames} frames, "
              f"speech {np.mean(p >= args.threshold) * 100:.0f}%, p[min/median/max] "
              f"{p.min():.2f}/{np.median(p):.2f}/{p.max():.2f}")
        print(f"  {strip}")


if __name__ == "__main__":
    main()
