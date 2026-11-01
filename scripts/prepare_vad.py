#!/usr/bin/env python3
"""Prepare the pinned Silero v5 CPU weights; never called during recording."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

MODEL_SHA256 = '0776449541b2b34cea5479bda89f6028e304ba0728fd4a1f053e17169f74a3eb'
MODEL_URL = 'https://github.com/LokalOptima/jarvis/releases/download/models-v1/silero_vad.bin'
SOURCE_REVISION = '7df37bab2381a28346c875de18b0e2decca4539a'
REPO = Path(__file__).resolve().parent.parent


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(128 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    parser.add_argument('--directory', type=Path, default=cache / 'paraketto/vad')
    parser.add_argument('--source', type=Path, default=cache / 'jarvis/silero_vad.bin', help='Copy existing verified weights when available')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    destination = args.directory / 'silero_vad.bin'
    valid = destination.is_file() and digest(destination) == MODEL_SHA256
    if args.verify_only:
        if not valid:
            raise SystemExit(f'Missing or incorrect pinned weights: {destination}')
        print(f'Verified Silero v5: {destination}')
        return
    args.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not valid:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.directory, prefix='.silero-', delete=False) as output:
                temporary = Path(output.name)
                if args.source.is_file() and digest(args.source) == MODEL_SHA256:
                    with args.source.open('rb') as source:
                        shutil.copyfileobj(source, output)
                else:
                    print('Downloading pinned Silero v5 weights', flush=True)
                    with urlopen(Request(MODEL_URL, headers={'User-Agent': 'Paraketto model preparation'}), timeout=60) as source:
                        shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            if digest(temporary) != MODEL_SHA256:
                raise SystemExit('Silero checksum mismatch; existing weights were not replaced')
            os.replace(temporary, destination)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    for name in ('LICENSE', 'LICENSE_SILERO', 'NOTICE'):
        shutil.copyfile(REPO / 'src/service/vad' / name, args.directory / name)
    manifest = {'model': 'Silero v5 16 kHz', 'sha256': MODEL_SHA256, 'url': MODEL_URL,
                'source_revision': SOURCE_REVISION, 'sample_rate': 16000, 'frame_samples': 512}
    (args.directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    directory_fd = os.open(args.directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f'Verified Silero v5: {destination}')


if __name__ == '__main__':
    main()
