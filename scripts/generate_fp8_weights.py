#!/usr/bin/env python3
"""Generate paraketto-fp8.bin from paraketto-fp16.bin.

Reads the FP16 weight file, quantizes GEMM matrices to FP8 E4M3 (absmax),
and writes the FP8 weight file in the exact binary layout expected by
conformer_fp8.cpp's fp8_load.

Usage:
  uv run python scripts/generate_fp8_weights.py                          # default paths
  uv run python scripts/generate_fp8_weights.py input.bin output.bin     # explicit paths
"""

import struct
import sys
from pathlib import Path

import numpy as np

try:
    from ml_dtypes import float8_e4m3fn
except ImportError:
    print("error: ml_dtypes required.  uv pip install ml_dtypes", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model constants (must match model_defs.h)
# ---------------------------------------------------------------------------
D_MODEL = 1024
D_FF = 4096
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS  # 128
N_BLOCKS = 24
D_CONV_PW = 2048
CONV_K = 9
SUB_CHANNELS = 256
D_PRED = 640
D_JOINT = 640

N_FP8_SCALES = N_BLOCKS * 9 + 6  # 222

FP8_E4M3_MAX = np.float32(448.0)
ALIGN = 256

WEIGHTS_HEADER = 8   # fp16: magic(4) + version(4)
FP8_HEADER = 16      # fp8: magic(8) + version(4) + model_version(4)
FP8_WEIGHTS_VERSION = 2


def align_up(x: int) -> int:
    return (x + ALIGN - 1) & ~(ALIGN - 1)


# ---------------------------------------------------------------------------
# FP16 tensor reader — walks the fp16.bin layout matching weights.cpp
# ---------------------------------------------------------------------------
class FP16Reader:
    def __init__(self, data: bytes, n_vocab: int, d_output: int):
        self.data = data
        self.base = WEIGHTS_HEADER  # file offset where GPU data starts
        self.off = 0  # GPU allocation offset (matches assign_weight_pointers)
        self.n_vocab = n_vocab
        self.d_output = d_output

    def take(self, n: int) -> np.ndarray:
        self.off = align_up(self.off)
        file_off = self.base + self.off
        arr = np.frombuffer(self.data, dtype=np.float16, count=n, offset=file_off)
        self.off += n * 2
        return arr

    def read_all(self):
        """Read all tensors in the exact order of assign_weight_pointers."""
        t = {}

        # Subsampling
        for i in [0, 2, 3, 5, 6]:
            wn = SUB_CHANNELS * SUB_CHANNELS if i in (3, 6) else SUB_CHANNELS * 9
            t[f"sub_conv.{i}.weight"] = self.take(wn)
            t[f"sub_conv.{i}.bias"] = self.take(SUB_CHANNELS)
        t["sub_out_w"] = self.take(SUB_CHANNELS * 16 * D_MODEL)
        t["sub_out_b"] = self.take(D_MODEL)

        # 24 conformer blocks
        for b in range(N_BLOCKS):
            p = f"block.{b}"
            t[f"{p}.ff1_ln_w"] = self.take(D_MODEL)
            t[f"{p}.ff1_ln_b"] = self.take(D_MODEL)
            t[f"{p}.ff1_w1"] = self.take(D_MODEL * D_FF)
            t[f"{p}.ff1_w2"] = self.take(D_FF * D_MODEL)
            t[f"{p}.mhsa_ln_w"] = self.take(D_MODEL)
            t[f"{p}.mhsa_ln_b"] = self.take(D_MODEL)
            t[f"{p}.q_w"] = self.take(D_MODEL * D_MODEL)
            t[f"{p}.k_w"] = self.take(D_MODEL * D_MODEL)
            t[f"{p}.v_w"] = self.take(D_MODEL * D_MODEL)
            t[f"{p}.pos_w"] = self.take(D_MODEL * D_MODEL)
            t[f"{p}.pos_bias_u"] = self.take(N_HEADS * HEAD_DIM)
            t[f"{p}.pos_bias_v"] = self.take(N_HEADS * HEAD_DIM)
            t[f"{p}.out_w"] = self.take(D_MODEL * D_MODEL)
            t[f"{p}.conv_ln_w"] = self.take(D_MODEL)
            t[f"{p}.conv_ln_b"] = self.take(D_MODEL)
            t[f"{p}.conv_pw1_w"] = self.take(D_CONV_PW * D_MODEL)
            t[f"{p}.conv_dw_w"] = self.take(D_MODEL * CONV_K)
            t[f"{p}.conv_dw_b"] = self.take(D_MODEL)
            t[f"{p}.conv_pw2_w"] = self.take(D_MODEL * D_MODEL)
            t[f"{p}.ff2_ln_w"] = self.take(D_MODEL)
            t[f"{p}.ff2_ln_b"] = self.take(D_MODEL)
            t[f"{p}.ff2_w1"] = self.take(D_MODEL * D_FF)
            t[f"{p}.ff2_w2"] = self.take(D_FF * D_MODEL)
            t[f"{p}.final_ln_w"] = self.take(D_MODEL)
            t[f"{p}.final_ln_b"] = self.take(D_MODEL)

        # Decoder
        t["embed_w"] = self.take(self.n_vocab * D_PRED)
        t["lstm0_w_ih"] = self.take(4 * D_PRED * D_PRED)
        t["lstm0_w_hh"] = self.take(4 * D_PRED * D_PRED)
        t["lstm0_bias"] = self.take(8 * D_PRED)  # b_ih || b_hh
        t["lstm1_w_ih"] = self.take(4 * D_PRED * D_PRED)
        t["lstm1_w_hh"] = self.take(4 * D_PRED * D_PRED)
        t["lstm1_bias"] = self.take(8 * D_PRED)

        # Joint network
        t["enc_proj_w"] = self.take(D_MODEL * D_JOINT)
        t["enc_proj_b"] = self.take(D_JOINT)
        t["dec_proj_w"] = self.take(D_PRED * D_JOINT)
        t["dec_proj_b"] = self.take(D_JOINT)
        t["out_proj_w"] = self.take(D_JOINT * self.d_output)
        t["out_proj_b"] = self.take(self.d_output)

        return t


# ---------------------------------------------------------------------------
# FP8 E4M3 absmax quantization (matches kernels_fp8.cu)
# ---------------------------------------------------------------------------
def quantize_absmax(tensor: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """Quantize fp16 tensor to fp8 E4M3 with absmax scaling.

    Returns (quantized_bytes, scale) where:
      scale = amax / 448.0
      quantized = cast_to_e4m3(tensor / scale)  with saturation
    """
    fp32 = tensor.astype(np.float32)
    amax = np.max(np.abs(fp32))
    scale = np.float32(amax / FP8_E4M3_MAX) if amax > 0 else np.float32(1.0)
    inv_scale = np.float32(1.0) / scale  # match CUDA: multiply by reciprocal
    scaled = fp32 * inv_scale
    # ml_dtypes float8_e4m3fn saturates on cast (matches __NV_SATFINITE)
    quantized = scaled.astype(float8_e4m3fn)
    return quantized.view(np.uint8), scale


# ---------------------------------------------------------------------------
# Combine LSTM weights: interleave w_ih and w_hh into [4*D_PRED, 2*D_PRED]
# ---------------------------------------------------------------------------
def combine_lstm_w(w_ih: np.ndarray, w_hh: np.ndarray) -> np.ndarray:
    """Interleave w_ih [4*D_PRED, D_PRED] and w_hh [4*D_PRED, D_PRED]
    into combined [4*D_PRED, 2*D_PRED] (row-interleaved).

    Matches the cudaMemcpy2D interleaving in the old C++ quantization path:
      dst_pitch = 2*D_PRED, src_pitch = D_PRED, width = D_PRED, height = 4*D_PRED
    """
    ih = w_ih.reshape(4 * D_PRED, D_PRED)
    hh = w_hh.reshape(4 * D_PRED, D_PRED)
    combined = np.empty((4 * D_PRED, 2 * D_PRED), dtype=np.float16)
    combined[:, :D_PRED] = ih
    combined[:, D_PRED:] = hh
    return combined.ravel()


def combine_lstm_bias(bias: np.ndarray) -> np.ndarray:
    """Combine b_ih + b_hh into single [4*D_PRED] bias.

    The fp16.bin stores [8*D_PRED] = b_ih || b_hh concatenated.
    The C++ code does: residual_add_fp16(bias, bias + 4*D_PRED, out, 4*D_PRED, 1.0, stream)
    which adds bias[0:4*D_PRED] + bias[4*D_PRED:8*D_PRED].
    """
    b_ih = bias[:4 * D_PRED].astype(np.float32)
    b_hh = bias[4 * D_PRED:].astype(np.float32)
    return (b_ih + b_hh).astype(np.float16)


# ---------------------------------------------------------------------------
# Concatenate Q, K, V into QKV [D_MODEL, 3*D_MODEL]
# ---------------------------------------------------------------------------
def concat_qkv(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Interleave Q, K, V weight rows into [D_MODEL, 3*D_MODEL].

    Matches the cudaMemcpy2D in the old C++ code:
      dst_pitch = 3*D_MODEL, src_pitch = D_MODEL, width = D_MODEL, height = D_MODEL
    """
    q2 = q.reshape(D_MODEL, D_MODEL)
    k2 = k.reshape(D_MODEL, D_MODEL)
    v2 = v.reshape(D_MODEL, D_MODEL)
    qkv = np.empty((D_MODEL, 3 * D_MODEL), dtype=np.float16)
    qkv[:, :D_MODEL] = q2
    qkv[:, D_MODEL:2*D_MODEL] = k2
    qkv[:, 2*D_MODEL:] = v2
    return qkv.ravel()


# ---------------------------------------------------------------------------
# Build FP8 pool blob (matches conformer_fp8.cpp pool layout)
# ---------------------------------------------------------------------------
class PoolBuilder:
    """Builds the fp8 pool blob with 256-byte aligned tensors."""

    def __init__(self):
        self.buf = bytearray()
        self.scales: list[np.float32] = []

    def _align(self):
        pad = align_up(len(self.buf)) - len(self.buf)
        if pad:
            self.buf += b"\x00" * pad

    def add_fp8(self, tensor_fp16: np.ndarray) -> None:
        """Quantize and append an FP8 weight + its scale."""
        self._align()
        quantized, scale = quantize_absmax(tensor_fp16)
        self.buf += quantized.tobytes()
        self.scales.append(scale)

    def finalize(self) -> bytes:
        """Append scales array (256-aligned) and return the complete pool blob."""
        self._align()
        scales_bytes = np.array(self.scales, dtype=np.float32).tobytes()
        self.buf += scales_bytes
        assert len(self.scales) == N_FP8_SCALES, \
            f"expected {N_FP8_SCALES} scales, got {len(self.scales)}"
        return bytes(self.buf)


# ---------------------------------------------------------------------------
# Build non-GEMM FP16 blob (matches fp8_load order in conformer_fp8.cpp)
# ---------------------------------------------------------------------------
def build_fp16_blob(t: dict, n_vocab: int, d_output: int,
                    lstm_combined_w: list[np.ndarray],
                    lstm_combined_b: list[np.ndarray]) -> bytes:
    """Pack non-GEMM FP16 weights in the order expected by fp8_load."""
    parts: list[bytes] = []

    def add(arr: np.ndarray):
        parts.append(arr.astype(np.float16).tobytes())

    # Sub-conv weights
    for i in [0, 2, 3, 5, 6]:
        add(t[f"sub_conv.{i}.weight"])
        add(t[f"sub_conv.{i}.bias"])
    add(t["sub_out_b"])

    # Per-block non-GEMM
    for b in range(N_BLOCKS):
        p = f"block.{b}"
        add(t[f"{p}.ff1_ln_w"]);   add(t[f"{p}.ff1_ln_b"])
        add(t[f"{p}.mhsa_ln_w"]);  add(t[f"{p}.mhsa_ln_b"])
        add(t[f"{p}.pos_bias_u"]); add(t[f"{p}.pos_bias_v"])
        add(t[f"{p}.conv_ln_w"]);  add(t[f"{p}.conv_ln_b"])
        add(t[f"{p}.conv_dw_w"]);  add(t[f"{p}.conv_dw_b"])
        add(t[f"{p}.ff2_ln_w"]);   add(t[f"{p}.ff2_ln_b"])
        add(t[f"{p}.final_ln_w"]); add(t[f"{p}.final_ln_b"])

    # Decoder
    add(t["embed_w"])

    # LSTM combined weights + biases (pre-computed by caller)
    for w in lstm_combined_w:
        add(w)
    for b in lstm_combined_b:
        add(b)

    # Joint
    add(t["dec_proj_w"])
    add(t["out_proj_w"])
    add(t["enc_proj_b"])
    add(t["dec_proj_b"])
    add(t["out_proj_b"])

    return b"".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) >= 3:
        fp16_path = Path(sys.argv[1])
        fp8_path = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        fp16_path = Path(sys.argv[1])
        fp8_path = fp16_path.parent / fp16_path.name.replace("-fp16.", "-fp8.")
    else:
        cache = Path.home() / ".cache" / "paraketto"
        fp16_path = cache / "paraketto-fp16.bin"
        fp8_path = cache / "paraketto-fp8.bin"

    if not fp16_path.exists():
        print(f"error: {fp16_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {fp16_path} ({fp16_path.stat().st_size / 1e6:.1f} MB)")
    data = fp16_path.read_bytes()

    # Validate header
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != 0x544B5250:  # "PRKT"
        print(f"error: bad magic 0x{magic:08x}", file=sys.stderr)
        sys.exit(1)
    if version not in (2, 3):
        print(f"error: unsupported version {version}", file=sys.stderr)
        sys.exit(1)

    model_version = version
    if model_version == 3:
        n_vocab = 8193
        d_output = 8198
    else:
        n_vocab = 1025
        d_output = 1030

    print(f"  model version: {model_version}, vocab: {n_vocab}, d_output: {d_output}")

    # Read all FP16 tensors
    reader = FP16Reader(data, n_vocab, d_output)
    t = reader.read_all()
    print(f"  read {len(t)} tensors ({reader.off:,} bytes consumed)")

    # Build FP8 pool blob
    print("Quantizing GEMM weights to FP8 E4M3...")
    pool = PoolBuilder()

    for b in range(N_BLOCKS):
        p = f"block.{b}"
        qkv = concat_qkv(t[f"{p}.q_w"], t[f"{p}.k_w"], t[f"{p}.v_w"])
        pool.add_fp8(qkv)
        pool.add_fp8(t[f"{p}.ff1_w1"])
        pool.add_fp8(t[f"{p}.ff1_w2"])
        pool.add_fp8(t[f"{p}.ff2_w1"])
        pool.add_fp8(t[f"{p}.ff2_w2"])
        pool.add_fp8(t[f"{p}.pos_w"])
        pool.add_fp8(t[f"{p}.out_w"])
        pool.add_fp8(t[f"{p}.conv_pw1_w"])
        pool.add_fp8(t[f"{p}.conv_pw2_w"])

    pool.add_fp8(t["sub_out_w"])
    pool.add_fp8(t["enc_proj_w"])

    # Pre-combine LSTM weights (used in both pool and fp16 blob)
    lstm_combined_w = [
        combine_lstm_w(t[f"lstm{layer}_w_ih"], t[f"lstm{layer}_w_hh"])
        for layer in [0, 1]
    ]
    lstm_combined_b = [
        combine_lstm_bias(t[f"lstm{layer}_bias"])
        for layer in [0, 1]
    ]

    for w in lstm_combined_w:
        pool.add_fp8(w)

    pool.add_fp8(t["dec_proj_w"])
    pool.add_fp8(t["out_proj_w"])

    pool_blob = pool.finalize()
    print(f"  pool blob: {len(pool_blob):,} bytes, {len(pool.scales)} scales")

    # Build non-GEMM FP16 blob
    fp16_blob = build_fp16_blob(t, n_vocab, d_output, lstm_combined_w, lstm_combined_b)
    print(f"  fp16 blob: {len(fp16_blob):,} bytes")

    # Write fp8.bin
    tmp = fp8_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(b"PRKTFP8\x00")
        f.write(struct.pack("<I", FP8_WEIGHTS_VERSION))
        f.write(struct.pack("<I", model_version))
        f.write(pool_blob)
        f.write(fp16_blob)
    tmp.replace(fp8_path)

    size = fp8_path.stat().st_size
    print(f"Wrote {fp8_path} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
