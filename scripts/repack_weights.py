#!/usr/bin/env python3
"""Convert existing weight files to the current format.

paraketto-fp16.bin: old PRKT v1 (with text header) → new PRKT v2 (headerless, fixed order)

Run once after pulling this update. Does not require ONNX re-export.

Usage:
  uv run python scripts/repack_weights.py              # repack cached V2 weights in-place
  uv run python scripts/repack_weights.py paraketto-fp16.bin  # repack only paraketto-fp16.bin
"""

import os
import struct
import sys
from pathlib import Path

from weight_format import MAGIC, TENSOR_ALIGN, align_up, tensor_order


def repack_fp16(src: Path, dst: Path) -> None:
    print(f"Repacking {src} → {dst}")
    with src.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        sys.exit(f"Truncated PRKT header: {src}")
    magic, version = struct.unpack("<II", header)
    if magic != 0x544B5250:
        print(f"  ERROR: not a PRKT file (magic=0x{magic:08x})")
        sys.exit(1)
    if version in (2, 3):
        print(f"  Already version {version}, nothing to do.")
        return
    if version != 1:
        print(f"  ERROR: unknown version {version}")
        sys.exit(1)

    data = src.read_bytes()
    header_len, = struct.unpack_from("<Q", data, 8)
    header_text = data[16:16 + header_len].decode("utf-8")
    data_start = align_up(16 + header_len, 4096)
    tensor_data = data[data_start:]

    # Parse tensor index
    tensors: dict[str, tuple[int, int, str, tuple]] = {}
    for line in header_text.strip().split("\n"):
        parts = line.split()
        name, offset, size_bytes, dtype = parts[0], int(parts[1]), int(parts[2]), parts[3]
        tensors[name] = (offset, size_bytes, dtype, tuple(int(d) for d in parts[4:]))

    # Load all tensors into memory
    loaded: dict[str, bytes] = {}
    for name, (offset, size_bytes, dtype, shape) in tensors.items():
        loaded[name] = tensor_data[offset:offset + size_bytes]

    ordered = tensor_order()
    missing = [n for n in ordered if n not in loaded]
    if missing:
        print(f"  ERROR: {len(missing)} tensors not found:")
        for n in missing[:10]:
            print(f"    {n}")
        sys.exit(1)

    # Write new format
    tmp = dst.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", 2))
        offset = 0
        for name in ordered:
            raw = loaded[name]
            pad = align_up(offset, TENSOR_ALIGN) - offset
            if pad:
                f.write(b"\x00" * pad)
                offset += pad
            f.write(raw)
            offset += len(raw)
        pad = align_up(offset, TENSOR_ALIGN) - offset
        if pad:
            f.write(b"\x00" * pad)

    tmp.replace(dst)
    size = dst.stat().st_size
    print(f"  Done: {size:,} bytes ({size / 1e6:.1f} MB)")


def main() -> None:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "paraketto"
    targets = sys.argv[1:] or [str(cache / "paraketto-fp16.bin")]

    for name in targets:
        path = Path(name)
        if not path.exists():
            print(f"Skipping {name} (not found)")
            continue
        repack_fp16(path, path)

    print("\nDone. Rebuild all binaries.")


if __name__ == "__main__":
    main()
