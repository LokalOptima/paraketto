"""Export Parakeet Ultra's voice-activity head to Core ML.

The head sits on the encoder's subsampler, so the exported model is the subsampler
(NeMo "dw_striding" ConvSubsampling, 8x in time) followed by the head:

    proj: Conv1d 1024 -> 128, k=1, SiLU
    ctx:  Conv1d 128 -> 128, k=5, same padding, SiLU
    out:  Conv1d 128 -> 1, k=1, sigmoid

Input is the mel spectrogram the FluidAudio Preprocessor model produces ([1, 128, 1501],
15 s at 10 ms hop); output is one speech probability per 80 ms encoder frame ([1, 188]).

Moondream does not document the head's activations. The wiring follows parakeet.cpp's
published analysis (docs/ternary.md): subsampler tap, SiLU after proj and ctx, no residual.

Only the 18 needed tensors (about 9 MB) are fetched from the Hugging Face safetensors file
with HTTP range requests; the full 1.2 GB checkpoint is never downloaded.

Usage:
    uv run python export_vad_head.py --out ../../Resources/VadHead.mlpackage
"""

import argparse
import json
import struct
import urllib.request
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

REPO = "moondream/parakeet-ultra"
SAFETENSORS_URL = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors"
NEEDED_PREFIXES = ("encoder.subsampling.", "vad_head.")
MEL_BINS = 128
MEL_FRAMES = 1501  # FluidAudio Preprocessor output for 15 s
SUB_CHANNELS = 256
D_MODEL = 1024


def fetch_tensors(cache: Path) -> dict[str, torch.Tensor]:
    """Range-fetches the subsampler and VAD head tensors, caching them as a small safetensors file."""
    if cache.exists():
        return load_safetensors(cache)
    with urllib.request.urlopen(urllib.request.Request(SAFETENSORS_URL, headers={"Range": "bytes=0-7"})) as r:
        header_len = struct.unpack("<Q", r.read(8))[0]
    with urllib.request.urlopen(
        urllib.request.Request(SAFETENSORS_URL, headers={"Range": f"bytes=8-{7 + header_len}"})
    ) as r:
        header = json.loads(r.read())
    base = 8 + header_len
    wanted = {n: h for n, h in header.items() if n != "__metadata__" and n.startswith(NEEDED_PREFIXES)}
    blobs, out_header, offset = [], {}, 0
    for name, info in wanted.items():
        start, end = info["data_offsets"]
        req = urllib.request.Request(SAFETENSORS_URL, headers={"Range": f"bytes={base + start}-{base + end - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        assert len(data) == end - start, name
        out_header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    encoded = json.dumps(out_header).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        f.writelines(blobs)
    print(f"fetched {len(wanted)} tensors ({offset / 1e6:.1f} MB) into {cache}")
    return load_safetensors(cache)


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return {k: v.float() for k, v in load_file(str(path)).items()}


class Subsampler(nn.Module):
    """NeMo ConvSubsampling, subsampling_factor 8, dw_striding, 256 channels."""

    def __init__(self):
        super().__init__()
        c = SUB_CHANNELS
        self.layers = nn.Sequential(
            nn.Conv2d(1, c, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c),
            nn.Conv2d(c, c, 1),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c),
            nn.Conv2d(c, c, 1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(c * (MEL_BINS // 8), D_MODEL)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 128, T] -> [B, 1, T, 128]
        x = mel.transpose(1, 2).unsqueeze(1)
        x = self.layers(x)  # [B, C, T', F']
        x = torch.flatten(x.transpose(1, 2), start_dim=2)  # [B, T', C*F'], channel-major as NeMo does
        return self.linear(x)  # [B, T', 1024]


class VadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(D_MODEL, 128, 1)
        self.ctx = nn.Conv1d(128, 128, 5, padding=2)
        self.out = nn.Conv1d(128, 1, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.transpose(1, 2)  # [B, 1024, T']
        x = torch.nn.functional.silu(self.proj(x))
        x = torch.nn.functional.silu(self.ctx(x))
        return torch.sigmoid(self.out(x)).squeeze(1)  # [B, T']


class SubsamplerVad(nn.Module):
    def __init__(self, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.subsampler = Subsampler()
        self.vad = VadHead()
        state = {}
        for name, tensor in weights.items():
            if name.startswith("encoder.subsampling."):
                state["subsampler." + name[len("encoder.subsampling."):]] = tensor
            elif name.startswith("vad_head."):
                state["vad." + name[len("vad_head."):]] = tensor
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not missing and not unexpected, (missing, unexpected)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.vad(self.subsampler(mel))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Output .mlpackage path")
    parser.add_argument("--cache", type=Path, default=Path("cache/vad_subsampler.safetensors"))
    args = parser.parse_args()

    model = SubsamplerVad(fetch_tensors(args.cache)).eval()
    example = torch.randn(1, MEL_BINS, MEL_FRAMES)
    with torch.no_grad():
        reference = model(example).numpy()
        traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=(1, MEL_BINS, MEL_FRAMES), dtype=np.float32)],
        outputs=[ct.TensorType(name="speech_probability", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.short_description = "Parakeet Ultra voice-activity head (subsampler + VAD), one speech probability per 80 ms"
    mlmodel.author = "Weights: moondream/parakeet-ultra (CC-BY-4.0). Export: paraketto-ultra tools/vad-export"
    mlmodel.save(str(args.out))

    predicted = mlmodel.predict({"mel": example.numpy()})["speech_probability"]
    print(f"output shape {predicted.shape}, max |coreml - torch| = {np.abs(predicted - reference).max():.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
