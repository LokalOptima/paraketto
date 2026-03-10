"""Benchmark paraketto with LLM corrector: WER and RTFx.

Usage:
    uv run python tests/bench_corrector.py paraketto.fp8
"""

import sys
from pathlib import Path
from bench_common import bench_server

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bench_corrector.py <binary>", file=sys.stderr)
        sys.exit(1)
    binary = ROOT / sys.argv[1]
    bench_server(binary, sys.argv[1], use_corrected=True)
