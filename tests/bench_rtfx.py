"""RTFx benchmark for Parakeet TDT TensorRT.

Measures real-time factor (RTFx = audio_duration / inference_time).
Target: ~500x RTFx on a modern NVIDIA GPU.

Usage:
    uv run python tests/bench_rtfx.py [--wav PATH] [--runs N]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="Parakeet TRT RTFx benchmark")
    parser.add_argument("--wav", default=str(ROOT / "data" / "sample.wav"),
                        help="Audio file to benchmark (default: data/sample.wav)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of inference runs (default: 10)")
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"Error: {wav_path} not found", file=sys.stderr)
        sys.exit(1)

    audio, sr = sf.read(str(wav_path), dtype="float32")
    audio_duration = len(audio) / sr
    print(f"Audio: {wav_path.name} ({audio_duration:.2f}s, {sr}Hz)")

    import parakeet_trt

    print("Loading model (first run compiles TRT engines, may take ~47s)...")
    t0 = time.monotonic()
    parakeet_trt.load_model()
    load_time = time.monotonic() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Warmup run
    print("Warmup run...")
    parakeet_trt.transcribe(audio, sr)

    # Benchmark runs
    times = []
    print(f"\nRunning {args.runs} inference passes...")
    for i in range(args.runs):
        t0 = time.perf_counter()
        text = parakeet_trt.transcribe(audio, sr)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        rtfx = audio_duration / elapsed
        print(f"  Run {i+1:2d}: {elapsed*1000:7.2f}ms  RTFx={rtfx:7.1f}x  text={text!r:.60}")

    times_arr = np.array(times)
    mean_time = times_arr.mean()
    std_time = times_arr.std()
    mean_rtfx = audio_duration / mean_time
    min_rtfx = audio_duration / times_arr.max()
    max_rtfx = audio_duration / times_arr.min()

    print(f"\n{'='*60}")
    print(f"Audio duration:  {audio_duration:.2f}s")
    print(f"Inference time:  {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"RTFx:            {mean_rtfx:.1f}x (min={min_rtfx:.1f}x, max={max_rtfx:.1f}x)")
    print(f"{'='*60}")

    if mean_rtfx < 400:
        print(f"\nWARNING: RTFx {mean_rtfx:.1f}x is below the 500x target", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nPASS: RTFx {mean_rtfx:.1f}x meets the ~500x target")


if __name__ == "__main__":
    main()
