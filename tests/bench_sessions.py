#!/usr/bin/env python3
"""Explicit GPU integration benchmark using isolated data and public speech.

Does not open a microphone or contact the user's service. Measures cold import,
warm Stop-to-publication latency, ASR queueing and bounded chunks.
Clipboard behavior is covered separately in the private desktop lifecycle suite.
"""
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import select
import sqlite3
import subprocess
import tempfile
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

REPO = Path(__file__).resolve().parent.parent


def wait(check, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(.025)
    raise AssertionError('Timed out waiting for session processing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=REPO / 'paraketto-ultra-fp16.bin')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--output', type=Path, default=REPO / 'scratch/session-gpu-results.json')
    args = parser.parse_args()
    assert args.samples > 0
    subprocess.run(['nvidia-smi'], check=True)
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise SystemExit('GPU compute work is active; run this benchmark when it is idle')
    clip = next((REPO / 'data/librispeech').glob('*.wav')).resolve()
    with wave.open(str(clip), 'rb') as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (16000, 1, 2)
        speech = source.readframes(source.getnframes())
    results = {'audio': clip.name, 'warm_stop_ms': [], 'cold': {}, 'bounded': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='paraketto-session-gpu-') as temporary:
        directory = Path(temporary)
        data = directory / 'service'
        data.mkdir()
        (data / 'preferences.ini').write_text('[Model]\nidle-seconds=300\nkeep-loaded=true\n')
        service = None
        with (directory / 'service.log').open('w') as log:
            try:
                service = subprocess.Popen([str(REPO / 'paraketto-service'), '--port', '0', '--data-dir', str(data),
                    '--weights', str(args.weights.resolve()), '--worker-binary', str(REPO / 'paraketto.cuda')],
                    stdout=subprocess.PIPE, stderr=log, text=True)
                assert select.select([service.stdout], [], [], 15)[0], 'Service startup timed out'
                endpoint = json.loads(service.stdout.readline())
                url = f'http://127.0.0.1:{endpoint["port"]}'

                def request(path, method='GET', payload=None, content_type='application/json'):
                    body = json.dumps(payload).encode() if isinstance(payload, dict) else payload
                    try:
                        response = urlopen(Request(url + path, body, {'Content-Type': content_type}, method=method), timeout=120)
                    except HTTPError as error:
                        response = error
                    with response:
                        raw = response.read()
                        result = json.loads(raw) if 'application/json' in response.headers.get('Content-Type', '') else raw.decode()
                        assert response.status == 200 or response.status == 202, (path, response.status, result)
                        return result

                def session(identity):
                    return request(f'/sessions/{identity}')

                def publication(identity):
                    wait(lambda: session(identity).get('publication'))
                    return request(f'/sessions/{identity}/publication')

                def legacy(audio):
                    boundary = 'paraketto-gpu-boundary'
                    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
                            'Content-Type: audio/wav\r\n\r\n').encode() + audio.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
                    return request('/transcribe', 'POST', body, f'multipart/form-data; boundary={boundary}')

                cold_start = time.monotonic()
                first = request('/sessions/import', 'POST', {'request_id': str(uuid.uuid4()), 'path': str(clip), 'silence_filter': False})['id']
                first_publication = publication(first)
                results['cold']['publication_ms'] = (time.monotonic() - cold_start) * 1000
                raw = request(f'/sessions/{first}/transcript?view=original')
                baseline = legacy(clip)
                assert raw == baseline['text'], (raw, baseline['text'])
                results['cold']['raw_matches_legacy'] = True
                results['cold']['publication'] = first_publication
                results['cold']['updates'] = request(f'/sessions/{first}/updates')

                for index in range(args.samples):
                    identity = request('/sessions', 'POST', {'request_id': str(uuid.uuid4()), 'silence_filter': False,
                        'audio_source': f'filesrc location="{clip}" ! wavparse ! identity sync=true'})['id']
                    time.sleep(3.2)
                    started = time.monotonic()
                    stopped = request(f'/sessions/{identity}/stop', 'POST', {})
                    flushed = time.monotonic()
                    snapshot = publication(identity)
                    duration = (time.monotonic() - started) * 1000
                    assert snapshot['complete'] and stopped['capture_state'] == 'stopped'
                    with sqlite3.connect(f'file:{data / "sessions.sqlite3"}?mode=ro', uri=True) as database:
                        inference, commit_us, published_us = database.execute('''
                            SELECT r.inference_ms,r.created_us,p.created_us FROM segments s
                            JOIN transcript_revisions r ON r.id=s.revision_id
                            JOIN publications p ON p.session_id=s.session_id AND p.generation=s.generation
                            WHERE s.session_id=? ORDER BY s.sequence DESC LIMIT 1''', (identity,)).fetchone()
                    results['warm_stop_ms'].append({'total_ms': duration, 'flush_ms': (flushed-started)*1000,
                        'asr_inference_ms': inference, 'commit_to_publication_ms': (published_us-commit_us)/1000})
                    print(f'Warm Stop {index+1}/{args.samples}: {duration:.1f} ms', flush=True)

                # Continuous speech crosses both ordinary and forced source
                # boundaries. Compare final words with the existing long-file path.
                long_audio = directory / 'continuous.wav'
                continuous = (speech * (125*32000 // len(speech) + 1))[:125*32000]
                with wave.open(str(long_audio), 'wb') as out:
                    out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    out.writeframes(continuous)
                long_id = request('/sessions/import', 'POST', {'request_id': str(uuid.uuid4()), 'path': str(long_audio), 'silence_filter': True})['id']
                # Both long-form sessions and legacy requests share the actual
                # ASR worker. Keep a legacy request in flight during processing.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    concurrent_legacy = pool.submit(legacy, clip)
                    publication(long_id)
                    response = concurrent_legacy.result(timeout=120)
                    assert response['text'] == baseline['text']
                    results['concurrent_legacy_succeeded'] = True
                segments = request(f'/sessions/{long_id}/updates')['segments']
                results['bounded']['updates'] = segments
                results['bounded']['original'] = request(f'/sessions/{long_id}/transcript?view=original')
                results['bounded']['legacy'] = legacy(long_audio)['text']
                import re
                results['bounded']['normalized_words_match'] = re.findall(r'\w+', results['bounded']['original'].lower()) == re.findall(r'\w+', results['bounded']['legacy'].lower())
                assert len(segments) > 1, 'Long speech was not split into bounded groups'
                results['bounded']['state'] = session(long_id)

                values = sorted(row['total_ms'] for row in results['warm_stop_ms'])
                results['warm_p95_ms'] = values[max(0, math.ceil(.95*len(values))-1)]
                results['warm_target_under_1000ms'] = results['warm_p95_ms'] < 1000
                results['complete'] = True
            finally:
                if service is not None:
                    service.terminate()
                    service.wait(timeout=15)
                    service.stdout.close()
                results['service_log'] = (directory / 'service.log').read_text()
                for filename in ('server.log',):
                    path = data / filename
                    if path.exists():
                        # Persist logs only for this public test recording.
                        (args.output.parent / f'{args.output.stem}-{filename}').write_text(path.read_text())
                args.output.write_text(json.dumps(results, indent=2))
    print(f'PASS: actual ASR sessions; warm Stop p95 {results["warm_p95_ms"]:.1f} ms', flush=True)


if __name__ == '__main__':
    main()
