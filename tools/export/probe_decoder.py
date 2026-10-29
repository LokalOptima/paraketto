"""Check a NumPy re-implementation of Parakeet's prediction network and joint against the
FluidAudio Core ML Decoder and JointDecisionv3 models, to pin down the exact semantics
(what the Core ML "decoder" output is, activations, logit layout) before porting to Swift.

Fetches only the decoder/joint tensors (about 36 MB) from the Hugging Face safetensors file.

Usage:
    uv run python probe_decoder.py
"""

import json
import struct
import urllib.request
from pathlib import Path

import coremltools as ct
import numpy as np
from safetensors.numpy import load_file

REPO = "moondream/parakeet-ultra"
URL = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors"
PREFIXES = ("decoder.", "joint.", "encoder_projector")
MODELS = Path.home() / "Library/Application Support/FluidAudio/Models/parakeet-ultra"
CACHE = Path("cache/decoder_joint.safetensors")


def fetch(cache: Path) -> dict[str, np.ndarray]:
    if not cache.exists():
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": "bytes=0-7"})) as r:
            n = struct.unpack("<Q", r.read(8))[0]
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": f"bytes=8-{7 + n}"})) as r:
            header = json.loads(r.read())
        base = 8 + n
        wanted = {k: v for k, v in header.items() if k != "__metadata__" and k.startswith(PREFIXES)}
        blobs, out, off = [], {}, 0
        for name, info in wanted.items():
            s, e = info["data_offsets"]
            with urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": f"bytes={base + s}-{base + e - 1}"})) as r:
                data = r.read()
            out[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [off, off + len(data)]}
            off += len(data)
            blobs.append(data)
        enc = json.dumps(out).encode()
        enc += b" " * ((8 - len(enc) % 8) % 8)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            f.write(struct.pack("<Q", len(enc)))
            f.write(enc)
            f.writelines(blobs)
        print(f"fetched {off / 1e6:.1f} MB")
    return {k: v.astype(np.float32) for k, v in load_file(str(cache)).items()}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lstm_step(x, h, c, w_ih, w_hh, b_ih, b_hh):
    gates = w_ih @ x + b_ih + w_hh @ h + b_hh
    i, f, g, o = np.split(gates, 4)
    c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
    h = sigmoid(o) * np.tanh(c)
    return h, c


def main():
    w = fetch(CACHE)
    print("tensors:", sorted(w))
    decoder = ct.models.CompiledMLModel(str(MODELS / "Decoder.mlmodelc"), compute_units=ct.ComputeUnit.CPU_ONLY)
    joint = ct.models.CompiledMLModel(str(MODELS / "JointDecisionv3.mlmodelc"), compute_units=ct.ComputeUnit.CPU_ONLY)

    rng = np.random.default_rng(0)
    token = 1234
    h0 = rng.standard_normal((2, 1, 640)).astype(np.float32) * 0.1
    c0 = rng.standard_normal((2, 1, 640)).astype(np.float32) * 0.1

    out = decoder.predict({
        "targets": np.array([[token]], dtype=np.int32),
        "target_length": np.array([1], dtype=np.int32),
        "h_in": h0, "c_in": c0,
    })
    print("coreml decoder outputs:", {k: v.shape for k, v in out.items()})

    # NumPy prediction network: embedding -> 2-layer LSTM -> (maybe) projection.
    emb = w["decoder.embedding.weight"][token]
    x = emb
    h_new, c_new = [], []
    for layer in range(2):
        h, c = lstm_step(
            x, h0[layer, 0], c0[layer, 0],
            w[f"decoder.lstm.weight_ih_l{layer}"], w[f"decoder.lstm.weight_hh_l{layer}"],
            w[f"decoder.lstm.bias_ih_l{layer}"], w[f"decoder.lstm.bias_hh_l{layer}"],
        )
        h_new.append(h); c_new.append(c)
        x = h
    lstm_out = x
    proj_out = w["decoder.decoder_projector.weight"] @ lstm_out + w["decoder.decoder_projector.bias"]

    cm_h = out["h_out"].reshape(2, 640); cm_c = out["c_out"].reshape(2, 640); cm_dec = out["decoder"].reshape(-1)
    print(f"h_out  max|diff| {np.abs(cm_h - np.stack(h_new)).max():.2e}")
    print(f"c_out  max|diff| {np.abs(cm_c - np.stack(c_new)).max():.2e}")
    print(f"decoder vs lstm_out  {np.abs(cm_dec - lstm_out).max():.2e}   vs projected {np.abs(cm_dec - proj_out).max():.2e}")

    # Joint: enc_proj(encoder_step) + pred_proj(decoder_step) -> relu -> head -> [8193 tokens | 5 durations]
    enc = rng.standard_normal(1024).astype(np.float32)
    jo = joint.predict({"encoder_step": enc.reshape(1, 1024, 1), "decoder_step": cm_dec.reshape(1, 640, 1)})
    print("coreml joint outputs:", {k: (v.shape, v.ravel()[:3]) for k, v in jo.items()})
    f = w["encoder_projector.weight"] @ enc + w["encoder_projector.bias"]
    for name, g in (("lstm_out as decoder_step", w["decoder.decoder_projector.weight"] @ cm_dec + w["decoder.decoder_projector.bias"]),
                    ("decoder_step already projected", cm_dec)):
        hdn = np.maximum(f + g, 0)
        logits = w["joint.head.weight"] @ hdn + w["joint.head.bias"]
        tok = int(np.argmax(logits[:8193])); dur = int(np.argmax(logits[8193:]))
        prob = float(np.exp(logits[:8193] - logits[:8193].max()).max() / np.exp(logits[:8193] - logits[:8193].max()).sum())
        print(f"  numpy ({name}): token {tok} duration_bin {dur} prob {prob:.3f}")
    print(f"  coreml: token {int(jo['token_id'].ravel()[0])} duration_bin {int(jo['duration'].ravel()[0])} prob {float(jo['token_prob'].ravel()[0]):.3f}")


if __name__ == "__main__":
    main()
