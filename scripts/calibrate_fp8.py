#!/usr/bin/env python3
r"""Measure FP8 activation ranges and embed checkpoint-specific static scales.

Example:
  uv run python scripts/calibrate_fp8.py --weights paraketto-ultra-fp8.bin \
      --output paraketto-ultra-fp8-calibrated.bin \
      --report scratch/ultra-fp8-calibration.json

Uses dynamic per-GEMM activation scaling during calibration, then takes the
maximum at each of 218 sites over a diverse corpus and adds 1.5x headroom.
Format-3 output contains the original weight payload plus 218 FP32 scales.
The report records audio-content hashes so benchmarks can exclude these clips,
even when the same recording occurs under a different filename.
"""

import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
N_SITES = 218
UNCALIBRATED_SIZES = {2: 634_386_324, 3: 657_338_260}
SOURCES = [
    ("cv_de", 25), ("cv_it", 25), ("cv_fr", 25), ("cv_en", 25),
    ("german", 10), ("italian", 10), ("french", 10),
    ("librispeech", 12), ("difficult", 12), ("earnings22", 8),
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audio_sha256(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    digest = hashlib.sha256(struct.pack("<II", rate, samples.shape[1]))
    digest.update(np.ascontiguousarray(samples, dtype="<f4").tobytes())
    return digest.hexdigest()


def read_header(path):
    with Path(path).open("rb") as f:
        header = f.read(16)
    if len(header) != 16:
        raise ValueError("Truncated FP8 header")
    magic, version, model = struct.unpack("<8sII", header)
    if magic != b"PRKTFP8\0" or version not in (2, 3) or model not in (2, 3):
        raise ValueError("Expected PRKTFP8 format 2/3, model V2/V3")
    payload_size = UNCALIBRATED_SIZES[model]
    if Path(path).stat().st_size != payload_size + (N_SITES * 4 if version == 3 else 0):
        raise ValueError("FP8 file size does not match its header")
    return model, payload_size


def parse_scales(stderr):
    lines = [line for line in stderr.splitlines() if line.startswith("SCALES:")]
    if len(lines) < 2:
        raise ValueError("Expected warmup and audio calibration output")
    # Every real-audio chunk counts; the first line is the silence warmup.
    values = np.array([[float(v) for v in line.split()[1:]] for line in lines[1:]], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != N_SITES:
        raise ValueError(f"Expected {N_SITES} scales per audio chunk")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Calibration returned non-finite or negative scales")
    return values.max(axis=0)


def aggregate_scales(values, headroom):
    if not np.isfinite(headroom) or headroom < 1.0:
        raise ValueError("Headroom must be finite and at least 1.0")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != N_SITES or not len(values):
        raise ValueError("No complete calibration observations")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid calibration observations")
    scales = values.max(axis=0) * np.float32(headroom)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("All activation sites must have a finite, positive calibrated range")
    return scales


def embed_scales(source, output, scales):
    model, payload_size = read_header(source)
    scales = np.asarray(scales, dtype="<f4")
    if scales.shape != (N_SITES,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Expected 218 finite, positive activation scales")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with Path(source).open("rb") as src, tempfile.NamedTemporaryFile(
                dir=output.parent, prefix=output.name + ".", delete=False) as dst:
            temporary = Path(dst.name)
            dst.write(struct.pack("<8sII", b"PRKTFP8\0", 3, model))
            src.seek(16)
            remaining = payload_size - 16
            while remaining:
                chunk = src.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    raise ValueError("Truncated weight payload")
                dst.write(chunk)
                remaining -= len(chunk)
            dst.write(scales.tobytes())
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "paraketto.fp8")
    parser.add_argument("--report", type=Path, default=ROOT / "scratch/fp8-calibration.json")
    parser.add_argument("--headroom", type=float, default=1.5)
    args = parser.parse_args()
    if not np.isfinite(args.headroom) or args.headroom < 1:
        parser.error("--headroom must be finite and at least 1.0")
    model, _ = read_header(args.weights)
    wavs = []
    seen = set()
    for name, count in SOURCES:
        manifest = ROOT / "data" / name / "manifest.json"
        if not manifest.exists():
            parser.error(f"Missing calibration corpus: {manifest}")
        entries = json.loads(manifest.read_text())[:count]
        if len(entries) != count:
            parser.error(f"Expected at least {count} clips in {manifest}")
        for entry in entries:
            path = manifest.parent / entry["audio_path"]
            digest = audio_sha256(path)
            if digest not in seen:
                wavs.append({"dataset": name, "audio_path": str(path), "audio_sha256": digest})
                seen.add(digest)
    print(f"Calibrating {args.weights} on {len(wavs)} clips, headroom={args.headroom}x", flush=True)
    all_scales = []
    env = {**os.environ, "PARAKETTO_CALIB": "1"}
    for i, entry in enumerate(wavs):
        cmd = [str(args.binary.resolve()), "--model", f"v{model}", "--weights",
               str(args.weights.resolve()), entry["audio_path"]]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        if run.returncode:
            raise RuntimeError(f"Calibration failed on {entry['audio_path']}:\n{run.stderr}")
        all_scales.append(parse_scales(run.stderr))
        if (i + 1) % 25 == 0 or i + 1 == len(wavs):
            print(f"  [{i+1}/{len(wavs)}]", flush=True)
    scales = aggregate_scales(all_scales, args.headroom)
    source_hash = file_sha256(args.weights)
    embed_scales(args.weights, args.output, scales)
    report = {
        "weights": str(args.weights.resolve()), "weights_sha256": source_hash,
        "binary_sha256": file_sha256(args.binary), "mode": "dynamic",
        "headroom": args.headroom, "n_sites": N_SITES,
        "observed_max_scales": np.max(all_scales, axis=0).tolist(),
        "activation_scales": scales.tolist(), "samples": wavs,
        "output": str(args.output.resolve()), "output_sha256": file_sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes) and {args.report}")


if __name__ == "__main__":
    main()
