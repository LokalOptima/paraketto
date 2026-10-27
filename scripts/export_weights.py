#!/usr/bin/env python3
"""Export ONNX model weights to paraketto-fp16.bin for the CUDA backend.

paraketto-fp16.bin format:
  uint32  magic   = 0x544B5250  ("PRKT" little-endian)
  uint32  version = 2 (V2 English) or 3 (V3 multilingual)
  [raw FP16 tensor data, 256-byte aligned, in fixed order matching source layout]

The tensor order in weight_format.py matches assign_weight_pointers() in
weights.cpp exactly; there is no separate index in the binary.

Usage:
  uv run python scripts/export_weights.py         # V2 English
  uv run python scripts/export_weights.py --v3    # V3 multilingual
"""

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx

from weight_format import MAGIC, TENSOR_ALIGN, align_up, tensor_order

REPOS = {
    2: "istupakov/parakeet-tdt-0.6b-v2-onnx",
    3: "istupakov/parakeet-tdt-0.6b-v3-onnx",
}


def resolve_names(model: onnx.ModelProto) -> dict[str, str]:
    """Map ONNX initializer names to semantic names by tracing the graph."""
    init_names = {init.name for init in model.graph.initializer}
    name_map: dict[str, str] = {}

    for node in model.graph.node:
        init_inputs = [inp for inp in node.input if inp in init_names]
        if not init_inputs:
            continue

        output_name = node.output[0] if node.output else ""
        path = output_name.strip("/")
        parts = path.split("/")
        if parts and "_output_" in parts[-1]:
            last = parts[-1]
            parts = parts[:-1]
            if last.startswith("LSTM_") and not last.startswith("LSTM_output"):
                idx_str = last.split("_")[1]
                parts.append(idx_str)
        semantic_base = ".".join(parts)

        for iname in init_inputs:
            if not iname.startswith("onnx::"):
                name_map[iname] = iname
                continue

            input_list = list(node.input)
            input_idx = input_list.index(iname)

            if node.op_type == "MatMul":
                name_map[iname] = f"{semantic_base}.weight"
            elif node.op_type == "Conv":
                if input_idx == 1:
                    name_map[iname] = f"{semantic_base}.weight"
                elif input_idx == 2:
                    name_map[iname] = f"{semantic_base}.bias"
            elif node.op_type == "LSTM":
                if input_idx == 1:
                    name_map[iname] = f"{semantic_base}.weight_ih"
                elif input_idx == 2:
                    name_map[iname] = f"{semantic_base}.weight_hh"
                elif input_idx == 3:
                    name_map[iname] = f"{semantic_base}.bias"
            elif node.op_type == "Add":
                name_map[iname] = f"{semantic_base}.bias"
            elif node.op_type == "Gemm":
                if input_idx == 1:
                    name_map[iname] = f"{semantic_base}.weight"
                elif input_idx == 2:
                    name_map[iname] = f"{semantic_base}.bias"
            else:
                name_map[iname] = iname

    for init in model.graph.initializer:
        if init.name not in name_map:
            name_map[init.name] = init.name

    return name_map


def extract_tensors(onnx_path: Path, prefix: str) -> dict[str, np.ndarray]:
    model = onnx.load(str(onnx_path))
    name_map = resolve_names(model)
    tensors = {}
    for init in model.graph.initializer:
        np_dtype = {
            onnx.TensorProto.FLOAT:   np.float32,
            onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.INT32:   np.int32,
            onnx.TensorProto.INT64:   np.int64,
        }.get(init.data_type)
        if np_dtype is None:
            continue
        arr = np.frombuffer(init.raw_data, dtype=np_dtype) if init.raw_data else \
              np.array(init.float_data or init.int32_data or init.int64_data, dtype=np_dtype)
        arr = arr.reshape(list(init.dims))
        tensors[f"{prefix}/{name_map.get(init.name, init.name)}"] = arr
    return tensors


def export_weights(output_path: Path, version: int = 2) -> dict[str, np.ndarray]:
    repo_id = REPOS[version]
    print(f"Downloading ONNX models from {repo_id}...")
    result = subprocess.run(
        ["uvx", "hf", "download", repo_id],
        capture_output=True, text=True, check=True,
    )
    onnx_dir = Path(result.stdout.strip())

    all_tensors: dict[str, np.ndarray] = {}
    for name, prefix in [("encoder-model.onnx", "encoder"),
                          ("decoder_joint-model.onnx", "decoder")]:
        path = onnx_dir / name
        if path.exists():
            print(f"Loading {name}...")
            all_tensors.update(extract_tensors(path, prefix))

    # Convert FP32 → FP16, drop int tensors
    for name in list(all_tensors):
        arr = all_tensors[name]
        if arr.dtype in (np.int64, np.int32, np.int8, np.uint8):
            del all_tensors[name]
        elif arr.dtype in (np.float32, np.float64):
            all_tensors[name] = arr.astype(np.float16)

    ordered = tensor_order()

    # Validate all expected tensors are present
    missing = [n for n in ordered if n not in all_tensors]
    if missing:
        print(f"ERROR: {len(missing)} tensors not found in ONNX:")
        for n in missing:
            print(f"  {n}")
        sys.exit(1)

    print(f"\nWriting {output_path} ({len(ordered)} tensors, version={version})...")
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", version))
        offset = 0
        for name in ordered:
            arr = all_tensors[name]
            raw = arr.astype(np.float16).tobytes()
            # 256-byte align
            pad = align_up(offset, TENSOR_ALIGN) - offset
            if pad:
                f.write(b"\x00" * pad)
                offset += pad
            f.write(raw)
            offset += len(raw)
        # Final alignment
        pad = align_up(offset, TENSOR_ALIGN) - offset
        if pad:
            f.write(b"\x00" * pad)

    size = output_path.stat().st_size
    print(f"Written: {size:,} bytes ({size / 1e6:.1f} MB)")
    return all_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX weights to paraketto-fp16.bin")
    parser.add_argument("--v3", action="store_true", help="Export V3 multilingual model")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.v3:
        output_path = root / "paraketto-v3-fp16.bin"
        export_weights(output_path, version=3)
    else:
        output_path = root / "paraketto-fp16.bin"
        export_weights(output_path, version=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
