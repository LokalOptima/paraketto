"""Benchmark the C++ parakeet binary (TensorRT): WER and RTFx.

Usage:
    uv run python tests/bench_cpp.py
"""

from pathlib import Path

from bench_common import bench_server

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    bench_server(ROOT / "parakeet", "parakeet")
