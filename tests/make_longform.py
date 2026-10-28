"""Build long-form test audio by joining the LibriSpeech clips with short gaps.

The reference transcript of each file is the concatenation of FluidAudio's per-clip transcripts
(tests/fixtures/librispeech_fluidaudio.tsv), so a long recording can be scored against a known
answer. Gaps are low-level noise rather than digital silence, and each file ends with half a
second of it, like the moment between a speaker's last word and the stop key.

Usage (from tools/export, which has numpy and soundfile):
    uv run python ../../tests/make_longform.py --clips ../../../paraketto/data/librispeech --out ../../.build/longform
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16_000
GROUPS = [  # (name, clip indices into the sorted fixture list, gap seconds)
    ("gap06_a", range(0, 5), 0.6),
    ("gap03_b", range(5, 11), 0.3),
    ("gap10_c", range(11, 17), 1.0),
    ("gap06_all", range(0, 17), 0.6),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixtures" / "librispeech_fluidaudio.tsv")
    args = parser.parse_args()

    clips = []
    for line in args.fixture.read_text().splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        name, text = line.split("\t", 1)
        clips.append((name.strip(), text.strip()))
    clips.sort()
    rng = np.random.default_rng(0)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, indices, gap in GROUPS:
        pieces, texts = [rng.normal(0, 1e-4, int(0.2 * RATE)).astype(np.float32)], []
        for i in indices:
            audio, rate = sf.read(args.clips / clips[i][0], dtype="float32")
            assert rate == RATE and audio.ndim == 1, clips[i][0]
            pieces += [audio, rng.normal(0, 1e-4, int(gap * RATE)).astype(np.float32)]
            texts.append(clips[i][1])
        pieces.append(rng.normal(0, 1e-4, int(0.5 * RATE)).astype(np.float32))
        audio = np.concatenate(pieces)
        sf.write(args.out / f"{name}.wav", audio, RATE, subtype="PCM_16")
        rows.append(f"{name}.wav\t{' '.join(texts)}")
        print(f"{name}.wav  {len(audio) / RATE:6.1f} s  {len(texts)} clips, {gap} s gaps")
    (args.out / "reference.tsv").write_text("# concatenated FluidAudio transcripts (file<TAB>text)\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
