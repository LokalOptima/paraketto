#!/usr/bin/env python3
"""Freeze probabilities from the separately built, pinned Jarvis CPU program."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import wave

SOURCE_REVISION = '7df37bab2381a28346c875de18b0e2decca4539a'
MODEL_SHA256 = '0776449541b2b34cea5479bda89f6028e304ba0728fd4a1f053e17169f74a3eb'
GGML_REVISION = 'a0ed91a442ea6b013bd42ebc3887a81792eaefa1'
REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_audio():
    noise = 0x12345678
    samples = []
    for index in range(64000):
        noise = (noise * 1664525 + 1013904223) & 0xffffffff
        region = index // 16000
        sample = 0
        if region == 1:
            sample = ((noise >> 16) & 0xffff) % 101 - 50
        elif region == 2:
            sample = ((index % 160) - 80) * 80 + ((noise >> 16) & 0xffff) % 401 - 200
        samples.append(sample)
    return struct.pack('<64000h', *samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jarvis', type=Path, required=True, help='Clean checkout at the pinned Jarvis revision')
    parser.add_argument('--reference', type=Path, help='Built vad_test; defaults to JARVIS/build/vad_test')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('reference.json'))
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.jarvis), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.jarvis), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != SOURCE_REVISION or dirty:
        raise SystemExit('The reference checkout must be clean at the pinned Jarvis revision')
    if digest(args.weights) != MODEL_SHA256:
        raise SystemExit('Pinned VAD model checksum mismatch')
    executable = (args.reference or args.jarvis / 'build/vad_test').resolve()
    result = {'jarvis_revision': SOURCE_REVISION, 'reference_executable_sha256': digest(executable),
              'model_sha256': MODEL_SHA256, 'adapted_ggml_revision': GGML_REVISION,
              'onnx_reference': 'Not run: no local ONNX reference model was available', 'cases': []}
    with tempfile.TemporaryDirectory(prefix='paraketto-vad-reference-') as directory:
        synthetic = Path(directory) / 'deterministic.wav'
        with wave.open(str(synthetic), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(deterministic_audio())
        inputs = [('deterministic-v1', synthetic, None)]
        speech = Path('data/librispeech/1089-134686-0000.wav')
        if (REPO / speech).is_file():
            inputs.append(('librispeech-1089-134686-0000', REPO / speech, str(speech)))
        for name, path, stored_path in inputs:
            completed = subprocess.run([str(executable), str(args.weights.resolve()), str(path)],
                                       check=True, capture_output=True, text=True)
            probabilities = [float(line) for line in completed.stdout.splitlines()]
            with wave.open(str(path), 'rb') as source:
                assert source.getframerate() == 16000 and source.getnchannels() == 1
                assert len(probabilities) == source.getnframes() // 512
            assert all(0 <= probability <= 1 for probability in probabilities)
            result['cases'].append({'name': name, 'path': stored_path, 'wav_sha256': digest(path),
                                    'probabilities': probabilities})
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {sum(len(case["probabilities"]) for case in result["cases"])} reference probabilities to {args.output}')


if __name__ == '__main__':
    main()
