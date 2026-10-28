"""Export the Parakeet Ultra encoder to Core ML as a multi-function model with 5 s, 10 s and
15 s windows, so short utterances do not pay for a full 15 s encoder pass.

Contract per function (same as FluidAudio's Encoder.mlmodelc, which only has the 15 s one):
    mel        fp32 [1, 128, T_mel]   T_mel = 501 / 1001 / 1501 (10 ms hop, plus one)
    mel_length int32 [1]
  -> encoder        fp32 [1, 1024, T']  T' = T_mel subsampled 8x (63 / 126 / 188)
     encoder_length int32 [1]

The encoder comes from Hugging Face transformers (`ParakeetForTDT`), traced per window length.
Weights are quantized to int8 per output channel (data-free), like FluidAudio's export.

Usage:
    uv run python export_encoder.py --out ~/Library/Application\\ Support/Paraketto/models/parakeet-ultra-encoder/Encoder.mlpackage
    uv run python export_encoder.py --seconds 5 10 15 --no-quantize --out ...   # fp16 only, for parity checks
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

MEL_BINS = 128
HOP = 160
SAMPLE_RATE = 16_000


def mel_frames(seconds: int) -> int:
    return seconds * SAMPLE_RATE // HOP + 1


def subsampled(length: torch.Tensor) -> torch.Tensor:
    """Output frames of three stride-2, kernel-3, padding-1 convolutions."""
    for _ in range(3):
        length = (length + 2 - 3) // 2 + 1
    return length


class EncoderWindow(nn.Module):
    """Wraps the HF encoder with FluidAudio's input/output layout and a length-derived mask."""

    def __init__(self, encoder: nn.Module, frames: int):
        super().__init__()
        self.encoder = encoder
        self.frames = frames

    def forward(self, mel: torch.Tensor, mel_length: torch.Tensor):
        features = mel.transpose(1, 2)  # [1, T_mel, 128]
        positions = torch.arange(self.frames, dtype=torch.int32).unsqueeze(0)
        attention_mask = positions < mel_length.unsqueeze(1)
        hidden = self.encoder(input_features=features, attention_mask=attention_mask).last_hidden_state
        encoder = hidden.transpose(1, 2)  # [1, 1024, T']
        return encoder, subsampled(mel_length)


def export_function(encoder: nn.Module, seconds: int, quantize: bool, workdir: Path) -> Path:
    frames = mel_frames(seconds)
    module = EncoderWindow(encoder, frames).eval()
    mel = torch.randn(1, MEL_BINS, frames)
    length = torch.tensor([frames - 7], dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(module, (mel, length), check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=(1, MEL_BINS, frames), dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="encoder", dtype=np.float32), ct.TensorType(name="encoder_length", dtype=np.int32)],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    if quantize:
        from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
        )
        mlmodel = linear_quantize_weights(mlmodel, config=config)
    path = workdir / f"encoder_{seconds}s.mlpackage"
    mlmodel.save(str(path))
    print(f"saved {path} ({frames} mel frames)")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Output multi-function .mlpackage")
    parser.add_argument("--seconds", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--no-quantize", action="store_true", help="Keep fp16 weights")
    parser.add_argument("--workdir", type=Path, default=Path("cache/encoder-export"))
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    from transformers import ParakeetForTDT

    model = ParakeetForTDT.from_pretrained("moondream/parakeet-ultra", dtype=torch.float32).eval()
    encoder = model.encoder
    del model

    paths = {seconds: export_function(encoder, seconds, not args.no_quantize, args.workdir) for seconds in args.seconds}

    descriptor = ct.utils.MultiFunctionDescriptor()
    for seconds, path in paths.items():
        descriptor.add_function(str(path), src_function_name="main", target_function_name=f"encoder_{seconds}s")
    descriptor.default_function_name = f"encoder_{max(paths)}s"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ct.utils.save_multifunction(descriptor, str(args.out))
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"saved {args.out}: functions {list(paths)} s, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
