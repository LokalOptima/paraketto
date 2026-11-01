#!/usr/bin/env python3
"""Measure a live audio chunk through warm ASR and durable publication.

Explicit GPU benchmark with public LibriSpeech audio, private service/data and
real-time GStreamer pacing. No microphone or clipboard is used.
"""
import argparse
import json
from pathlib import Path
import select
import sqlite3
import subprocess
import tempfile
import time
import uuid
from urllib.request import Request, urlopen
import wave

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--output', type=Path, default=REPO / 'scratch/pipeline-speed.json')
    args = parser.parse_args()
    assert 1 <= args.seconds <= 90
    subprocess.run(['nvidia-smi'], check=True)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise SystemExit('GPU compute work is active; retry when it is idle')
    result = {'requested_audio_seconds': args.seconds, 'clips': []}
    with tempfile.TemporaryDirectory(prefix='paraketto-pipeline-speed-') as temporary:
        data = Path(temporary)
        (data / 'preferences.ini').write_text('[Model]\nkeep-loaded=true\n')
        audio = data / 'speech.wav'
        remaining = args.seconds * 16000
        with wave.open(str(audio), 'wb') as out:
            out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            for clip in sorted((REPO / 'data/librispeech').glob('*.wav')):
                with wave.open(str(clip), 'rb') as source:
                    assert (source.getframerate(), source.getsampwidth(), source.getnchannels()) == (16000, 2, 1)
                    chunk = source.readframes(remaining)
                out.writeframes(chunk); remaining -= len(chunk)//2
                result['clips'].append(clip.name)
                if remaining <= 0:
                    break
        assert remaining == 0, 'Not enough public speech in the selected passage'
        service = None
        with (data / 'service.log').open('w') as log:
            try:
                service = subprocess.Popen([str(REPO / 'paraketto-service'), '--port', '0', '--data-dir', temporary,
                    '--weights', str(REPO / 'paraketto-ultra-fp16.bin')], stdout=subprocess.PIPE, stderr=log, text=True)
                assert select.select([service.stdout], [], [], 15)[0]
                endpoint = json.loads(service.stdout.readline())
                url = f'http://127.0.0.1:{endpoint["port"]}'

                def request(path, payload=None):
                    with urlopen(Request(url+path, None if payload is None else json.dumps(payload).encode(),
                                         {'Content-Type': 'application/json'}), timeout=15) as response:
                        return json.loads(response.read())

                def wait(check, timeout=120):
                    deadline = time.monotonic() + timeout
                    while time.monotonic() < deadline:
                        value = check()
                        if value:
                            return value
                        time.sleep(.01)
                    raise AssertionError('Pipeline stage timed out')

                started = time.monotonic()
                request('/models/asr/load', {})
                wait(lambda: request('/status')['models']['asr']['status'] == 'ready')
                result['asr_load_ms'] = (time.monotonic()-started)*1000
                identity = request('/sessions', {'request_id': str(uuid.uuid4()), 'silence_filter': False,
                    'audio_source': f'filesrc location="{audio}" ! wavparse ! identity sync=true'})['id']
                print(f'Recording {args.seconds}s of public speech in real time with ASR warm…', flush=True)
                wait(lambda: request(f'/sessions/{identity}/levels')['captured_samples'] >= args.seconds*16000)
                stopped_at = time.monotonic()
                state = request(f'/sessions/{identity}/stop', {})
                result['audio_seconds'] = state['durable_samples']/16000
                result['flush_ms'] = (time.monotonic()-stopped_at)*1000
                wait(lambda: request(f'/sessions/{identity}').get('publication'))
                result['stop_to_publication_ms'] = (time.monotonic()-stopped_at)*1000
                result['session'] = request(f'/sessions/{identity}')
                result['segments'] = request(f'/sessions/{identity}/updates')['segments']
                with sqlite3.connect(f'file:{data / "sessions.sqlite3"}?mode=ro', uri=True) as database:
                    result['asr_inference_ms'] = database.execute('SELECT sum(r.inference_ms) FROM transcript_revisions r JOIN segments s ON s.id=r.segment_id WHERE s.session_id=?', (identity,)).fetchone()[0]
                result['complete'] = True
            finally:
                if service:
                    service.terminate(); service.wait(timeout=15); service.stdout.close()
                result['service_log'] = (data / 'service.log').read_text()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key,value in result.items() if key not in ('segments','session','service_log')}, indent=2))


if __name__ == '__main__':
    main()
