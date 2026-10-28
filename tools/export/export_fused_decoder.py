"""Export one Core ML model that does a whole TDT token step: advance the prediction network
with the emitted token, then evaluate the joint on the next encoder frame.

FluidAudio ships Decoder and JointDecision as separate models, so a token step costs two
framework calls (and blank steps one). Fusing halves the calls on token steps; blank steps
keep using the joint-only model.

    inputs:  token int32 [1, 1], h_in fp32 [2, 1, 640], c_in fp32 [2, 1, 640], encoder_step fp32 [1, 1024, 1]
    outputs: h_out, c_out [2, 1, 640], decoder fp32 [1, 640, 1] (LSTM output, for later joint-only calls),
             token_id int32 [1], duration int32 [1] (duration bin), token_logit fp32 [1]

Weights come from moondream/parakeet-ultra (the same tensors the CPU decoder uses).

Usage:
    uv run python export_fused_decoder.py --out "~/Library/Application Support/Paraketto/models/parakeet-ultra-decoder/DecoderJoint.mlpackage"
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

from probe_decoder import CACHE, fetch, lstm_step

VOCAB = 8193
HIDDEN = 640
ENCODER = 1024


class FusedStep(nn.Module):
    def __init__(self, w: dict[str, np.ndarray]):
        super().__init__()
        t = {k: torch.from_numpy(v) for k, v in w.items()}
        self.embedding = nn.Embedding(VOCAB, HIDDEN)
        self.embedding.weight.data = t["decoder.embedding.weight"]
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, num_layers=2)
        for layer in range(2):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(self.lstm, f"{name}_l{layer}").data = t[f"decoder.lstm.{name}_l{layer}"]
        self.pred_proj = nn.Linear(HIDDEN, HIDDEN)
        self.pred_proj.weight.data = t["decoder.decoder_projector.weight"]
        self.pred_proj.bias.data = t["decoder.decoder_projector.bias"]
        self.enc_proj = nn.Linear(ENCODER, HIDDEN)
        self.enc_proj.weight.data = t["encoder_projector.weight"]
        self.enc_proj.bias.data = t["encoder_projector.bias"]
        self.head = nn.Linear(HIDDEN, VOCAB + 5)
        self.head.weight.data = t["joint.head.weight"]
        self.head.bias.data = t["joint.head.bias"]

    def forward(self, token, h_in, c_in, encoder_step):
        x = self.embedding(token.to(torch.long)).transpose(0, 1)  # [seq=1, batch=1, 640]
        out, (h, c) = self.lstm(x, (h_in, c_in))
        decoder = out[0]  # [1, 640]
        f = self.enc_proj(encoder_step[:, :, 0])  # [1, 640]
        g = self.pred_proj(decoder)
        logits = self.head(torch.relu(f + g))  # [1, 8198]
        token_logits = logits[:, :VOCAB]
        token_id = torch.argmax(token_logits, dim=1).to(torch.int32)
        duration = torch.argmax(logits[:, VOCAB:], dim=1).to(torch.int32)
        return h, c, decoder.reshape(1, HIDDEN, 1), token_id, duration, torch.amax(token_logits, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    w = fetch(CACHE)
    module = FusedStep(w).eval()
    example = (
        torch.tensor([[1234]], dtype=torch.int32),
        torch.zeros(2, 1, HIDDEN), torch.zeros(2, 1, HIDDEN),
        torch.randn(1, ENCODER, 1),
    )
    with torch.no_grad():
        traced = torch.jit.trace(module, example)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="h_in", shape=(2, 1, HIDDEN), dtype=np.float32),
            ct.TensorType(name="c_in", shape=(2, 1, HIDDEN), dtype=np.float32),
            ct.TensorType(name="encoder_step", shape=(1, ENCODER, 1), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="h_out", dtype=np.float32), ct.TensorType(name="c_out", dtype=np.float32),
            ct.TensorType(name="decoder", dtype=np.float32),
            ct.TensorType(name="token_id", dtype=np.int32), ct.TensorType(name="duration", dtype=np.int32),
            ct.TensorType(name="token_logit", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    mlmodel.short_description = "Parakeet TDT fused step: LSTM decoder update + joint decision for one encoder frame"
    args.out = args.out.expanduser()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))

    # Check against the NumPy reference used to validate the CPU decoder.
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(20):
        token = int(rng.integers(0, VOCAB))
        h0 = (rng.standard_normal((2, 1, HIDDEN)) * 0.1).astype(np.float32)
        c0 = (rng.standard_normal((2, 1, HIDDEN)) * 0.1).astype(np.float32)
        enc = rng.standard_normal(ENCODER).astype(np.float32)
        out = mlmodel.predict({"token": np.array([[token]], dtype=np.int32), "h_in": h0, "c_in": c0, "encoder_step": enc.reshape(1, ENCODER, 1)})
        x = w["decoder.embedding.weight"][token]
        for layer in range(2):
            x, _ = lstm_step(x, h0[layer, 0], c0[layer, 0], w[f"decoder.lstm.weight_ih_l{layer}"], w[f"decoder.lstm.weight_hh_l{layer}"],
                             w[f"decoder.lstm.bias_ih_l{layer}"], w[f"decoder.lstm.bias_hh_l{layer}"])
        f = w["encoder_projector.weight"] @ enc + w["encoder_projector.bias"]
        g = w["decoder.decoder_projector.weight"] @ x + w["decoder.decoder_projector.bias"]
        logits = w["joint.head.weight"] @ np.maximum(f + g, 0) + w["joint.head.bias"]
        ref_token, ref_duration = int(np.argmax(logits[:VOCAB])), int(np.argmax(logits[VOCAB:]))
        if (int(out["token_id"][0]), int(out["duration"][0])) != (ref_token, ref_duration):
            mismatches += 1
    print(f"saved {args.out}; decisions differing from fp32 reference on 20 random states: {mismatches}")


if __name__ == "__main__":
    main()
