#!/usr/bin/env python3
"""Convert moondream/parakeet-ultra Safetensors to the existing PRKT V3 format.

Usage:
  uvx hf download moondream/parakeet-ultra --local-dir scratch/parakeet-ultra
  uv run python scripts/export_ultra_weights.py scratch/parakeet-ultra
  ./paraketto.cuda --model v3 --weights paraketto-ultra-fp16.bin audio.wav

No PyTorch or NeMo is needed. Linear weights are transposed to our ONNX-derived
layout, convolution BatchNorm is folded, and PyTorch LSTM gates (ifgo) become
ONNX gates (iofg). The auxiliary VAD head is omitted; runtime chunking and
decoding remain those of paraketto. This exports FP16 only.
"""

import argparse
import ast
import json
import re
import struct
import tempfile
from itertools import zip_longest
from pathlib import Path

import numpy as np
from safetensors import safe_open

from weight_format import MAGIC, TENSOR_ALIGN, align_up, tensor_order

ROOT = Path(__file__).resolve().parent.parent
BATCH_NORM_EPS = 1e-5  # Transformers ParakeetEncoderConvolutionModule / BatchNorm1d
V3_FILE_BYTES = 1_253_967_368  # src/weights.cpp, including header and alignment


def validate_metadata(source: Path) -> None:
    config = json.loads((source / "config.json").read_text())
    expected = {
        "model_type": "parakeet_tdt", "vocab_size": 8193,
        "blank_token_id": 8192, "decoder_hidden_size": 640,
        "num_decoder_layers": 2, "durations": [0, 1, 2, 3, 4],
        "hidden_act": "relu",
    }
    encoder = {
        "hidden_size": 1024, "intermediate_size": 4096,
        "num_hidden_layers": 24, "num_attention_heads": 8,
        "num_key_value_heads": 8, "num_mel_bins": 128,
        "conv_kernel_size": 9, "subsampling_conv_channels": 256,
        "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2,
        "subsampling_factor": 8, "attention_bias": False,
        "convolution_bias": False, "scale_input": False, "hidden_act": "silu",
    }
    for section, values, actual in [
        ("config", expected, config),
        ("encoder_config", encoder, config.get("encoder_config", {})),
    ]:
        for key, value in values.items():
            if actual.get(key) != value:
                raise ValueError(f"Unsupported {section}.{key}: {actual.get(key)!r}; expected {value!r}")

    tokenizer = json.loads((source / "tokenizer.json").read_text())
    vocab = tokenizer["model"]["vocab"]
    cpp = (ROOT / "src/vocab_v3.h").read_text().split("{", 1)[1].rsplit("}", 1)[0]
    # C++ stores UTF-8 bytes as \xNN escapes; decode those before comparing.
    local = [ast.literal_eval(s).encode("latin1").decode("utf-8")
             for s in re.findall(r'"(?:\\.|[^"\\])*"', cpp)]
    if not isinstance(vocab, dict) or set(vocab.values()) != set(range(8192)):
        raise ValueError("Expected the V3 tokenizer with 8192 non-blank tokens")
    for token, index in vocab.items():
        if token.replace("▁", " ") != local[index]:
            raise ValueError(f"Tokenizer differs from V3 at token {index}")
    if not any(t["id"] == 8192 and t["content"] == "<blank>"
               for t in tokenizer["added_tokens"]):
        raise ValueError("Expected blank token at ID 8192")


def fold_batch_norm(weight, gamma, beta, mean, variance):
    """Fold eval-mode BatchNorm after a bias-free depthwise convolution in FP32."""
    scale = gamma.astype(np.float32) / np.sqrt(variance.astype(np.float32) + BATCH_NORM_EPS)
    folded_weight = weight.astype(np.float32) * scale.reshape((-1,) + (1,) * (weight.ndim - 1))
    folded_bias = beta.astype(np.float32) - mean.astype(np.float32) * scale
    return folded_weight, folded_bias


def reorder_lstm_gates(value):
    """PyTorch [input, forget, cell, output] -> our [input, output, forget, cell]."""
    i, f, g, o = np.split(value, 4, axis=0)
    return np.concatenate((i, o, f, g), axis=0)


def converted_tensors(reader):
    """Yield one tensor at a time in the layout defined by export_weights.py."""
    used = set()

    def get(name, shape):
        value = reader.get_tensor(name)
        if value.shape != tuple(shape):
            raise ValueError(f"{name}: shape {value.shape}, expected {shape}")
        if value.dtype not in (np.float16, np.float32) or not np.isfinite(value).all():
            raise ValueError(f"{name}: expected finite FP16/FP32 weights")
        used.add(name)
        return value

    for i in (0, 2, 3, 5, 6):
        shape = (256, 256, 1, 1) if i in (3, 6) else (256, 1, 3, 3)
        yield f"encoder/pre_encode.conv.{i}.weight", get(f"encoder.subsampling.layers.{i}.weight", shape)
        yield f"encoder/pre_encode.conv.{i}.bias", get(f"encoder.subsampling.layers.{i}.bias", (256,))
    yield "encoder/pre_encode.out.weight", get("encoder.subsampling.linear.weight", (1024, 4096)).T
    yield "encoder/pre_encode.out.bias", get("encoder.subsampling.linear.bias", (1024,))

    for i in range(24):
        src, dst = f"encoder.layers.{i}.", f"encoder/layers.{i}."

        def norm(name):
            for suffix in ("weight", "bias"):
                key = f"{name}.{suffix}"
                yield dst + key, get(src + key, (1024,))

        def feed_forward(name):
            for linear, shape in (("linear1", (4096, 1024)), ("linear2", (1024, 4096))):
                key = f"{name}.{linear}.weight"
                yield dst + key, get(src + key, shape).T

        yield from norm("norm_feed_forward1")
        yield from feed_forward("feed_forward1")
        yield from norm("norm_self_att")
        for target, origin in (("linear_q", "q_proj"), ("linear_k", "k_proj"),
                               ("linear_v", "v_proj"), ("linear_pos", "relative_k_proj")):
            yield dst + f"self_attn.{target}.weight", get(src + f"self_attn.{origin}.weight", (1024, 1024)).T
        for suffix in ("u", "v"):
            yield dst + f"self_attn.pos_bias_{suffix}", get(src + f"self_attn.bias_{suffix}", (8, 128))
        yield dst + "self_attn.linear_out.weight", get(src + "self_attn.o_proj.weight", (1024, 1024)).T
        yield from norm("norm_conv")
        yield dst + "conv.pointwise_conv1.weight", get(src + "conv.pointwise_conv1.weight", (2048, 1024, 1))
        weight, bias = fold_batch_norm(
            get(src + "conv.depthwise_conv.weight", (1024, 1, 9)),
            get(src + "conv.norm.weight", (1024,)), get(src + "conv.norm.bias", (1024,)),
            get(src + "conv.norm.running_mean", (1024,)), get(src + "conv.norm.running_var", (1024,)),
        )
        yield dst + "conv.depthwise_conv.weight", weight
        yield dst + "conv.depthwise_conv.bias", bias
        yield dst + "conv.pointwise_conv2.weight", get(src + "conv.pointwise_conv2.weight", (1024, 1024, 1))
        yield from norm("norm_feed_forward2")
        yield from feed_forward("feed_forward2")
        yield from norm("norm_out")

    yield "decoder/decoder.prediction.embed.weight", get("decoder.embedding.weight", (8193, 640))
    for layer in range(2):
        dst = "decoder/decoder.dec_rnn.lstm" + (".1" if layer else "")
        for name in ("weight_ih", "weight_hh"):
            yield f"{dst}.{name}", reorder_lstm_gates(get(f"decoder.lstm.{name}_l{layer}", (2560, 640)))
        biases = [reorder_lstm_gates(get(f"decoder.lstm.bias_{name}_l{layer}", (2560,)))
                  for name in ("ih", "hh")]
        yield dst + ".bias", np.concatenate(biases)

    for dst, src, shape in (
        ("decoder/joint.enc", "encoder_projector", (640, 1024)),
        ("decoder/joint.pred", "decoder.decoder_projector", (640, 640)),
    ):
        yield dst + ".weight", get(src + ".weight", shape).T
        yield dst + ".bias", get(src + ".bias", (640,))
    yield "decoder/joint.joint_net.joint_net.2.weight", get("joint.head.weight", (8198, 640)).T
    yield "decoder/joint.joint_net.2.bias", get("joint.head.bias", (8198,))

    ignored = {f"encoder.layers.{i}.conv.norm.num_batches_tracked" for i in range(24)}
    ignored |= {f"vad_head.{layer}.{param}" for layer in ("ctx", "out", "proj")
                for param in ("weight", "bias")}
    unexpected = set(reader.keys()) - used - ignored
    if unexpected:
        raise ValueError(f"Unmapped tensors: {sorted(unexpected)}")


def export_weights(source: Path, output: Path) -> None:
    validate_metadata(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with safe_open(source / "model.safetensors", framework="numpy") as reader, \
                tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", delete=False) as f:
            temporary = Path(f.name)
            f.write(MAGIC + struct.pack("<I", 3))
            offset = 0
            for expected, item in zip_longest(tensor_order(), converted_tensors(reader)):
                if item is None or item[0] != expected:
                    raise ValueError(f"Tensor layout mismatch: expected {expected}, got {None if item is None else item[0]}")
                name, value = item
                with np.errstate(over="raise", invalid="raise"):
                    value = np.ascontiguousarray(value, dtype="<f2")
                if not np.isfinite(value).all():
                    raise ValueError(f"{name}: non-finite converted weights")
                padding = align_up(offset, TENSOR_ALIGN) - offset
                f.write(b"\0" * padding)
                f.write(value.tobytes())
                offset += padding + value.nbytes
            f.write(b"\0" * (align_up(offset, TENSOR_ALIGN) - offset))
            if f.tell() != V3_FILE_BYTES:
                raise ValueError(f"Expected {V3_FILE_BYTES} output bytes, got {f.tell()}")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Wrote {output}: {output.stat().st_size:,} bytes, PRKT V3 FP16")
    print("V3 vocabulary verified; auxiliary VAD omitted. FP16 needs no calibration.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Directory containing model.safetensors, config.json and tokenizer.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paraketto-ultra-fp16.bin")
    args = parser.parse_args()
    export_weights(args.source, args.output)


if __name__ == "__main__":
    main()
