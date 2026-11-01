#!/usr/bin/env python3
"""Compare filtered and unfiltered ASR for explicit WAV inputs on an idle GPU.

Uses a private service and data directory. Never opens a
microphone or contacts the user's running service. Input filenames and generated
text are saved in the requested local report for inspection.
"""
import argparse
import json
from pathlib import Path
import select
import subprocess
import tempfile
import time
import uuid
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', nargs='+', type=Path)
    parser.add_argument('--weights', type=Path, default=REPO / 'paraketto-ultra-fp16.bin')
    parser.add_argument('--output', type=Path, default=REPO / 'scratch/vad-quality-results.json')
    args = parser.parse_args()
    subprocess.run(['nvidia-smi'], check=True)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise SystemExit('GPU compute work is active; retry when it is idle')
    results = []
    with tempfile.TemporaryDirectory(prefix='paraketto-vad-gpu-') as temporary:
        data = Path(temporary)
        (data / 'preferences.ini').write_text('[Model]\nkeep-loaded=true\n')
        with (data / 'service.log').open('w') as log:
            service = subprocess.Popen([str(REPO / 'paraketto-service'), '--port', '0', '--data-dir', temporary,
                '--weights', str(args.weights.resolve()), '--worker-binary', str(REPO / 'paraketto.cuda')],
                stdout=subprocess.PIPE, stderr=log, text=True)
            try:
                assert select.select([service.stdout], [], [], 15)[0]
                endpoint = json.loads(service.stdout.readline())
                url = f'http://127.0.0.1:{endpoint["port"]}'

                def request(path, payload=None):
                    body = None if payload is None else json.dumps(payload).encode()
                    with urlopen(Request(url + path, body, {'Content-Type': 'application/json'}), timeout=15) as response:
                        return json.loads(response.read())

                for audio in args.audio:
                    case = {'path': str(audio.resolve())}
                    for filtered in (False, True):
                        session = request('/sessions/import', {'request_id': str(uuid.uuid4()),
                            'path': str(audio.resolve()), 'silence_filter': filtered})
                        identity = session['id']
                        deadline = time.monotonic() + 90
                        while not session.get('publication') and time.monotonic() < deadline:
                            time.sleep(.05)
                            session = request(f'/sessions/{identity}')
                        assert session.get('publication'), session
                        text = request(f'/sessions/{identity}/publication')['text']
                        updates = request(f'/sessions/{identity}/updates')['segments']
                        case['filtered' if filtered else 'unfiltered'] = {'text': text,
                            'asr_seconds': sum((span['end']-span['start'])/16000 for segment in updates for span in segment['source_spans'])}
                    results.append(case)
                    print(json.dumps(case), flush=True)
            finally:
                service.terminate(); service.wait(timeout=15); service.stdout.close()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({'cases': results, 'service_log': (data/'service.log').read_text()}, indent=2))


if __name__ == '__main__':
    main()
