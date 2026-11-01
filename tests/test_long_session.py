"""Opt-in real-time capture soak; isolated data, public speech fixtures, no GPU.

python3 tests/test_long_session.py --seconds 3720 --report scratch/long-session.json
The default duration exceeds an hour. Never opens the user's microphone.
"""
import argparse
import json
from pathlib import Path
import time
import uuid
import wave

from service_test_support import Fixture, REPO, wait


def resident_kb(pid):
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])
    raise AssertionError('Service exited during capture')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=3720)
    parser.add_argument('--report', type=Path, default=REPO / 'scratch/long-session.json')
    args = parser.parse_args()
    assert args.seconds > 0
    with Fixture() as fixture:
        (fixture.directory / 'release-transcription').write_text('continue')
        # Alternate speech and forty-second pauses. This exercises silence
        # skipping and continued recording without a physical microphone.
        clip = next((REPO / 'data/librispeech').glob('*.wav'))
        with wave.open(str(clip), 'rb') as source:
            assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (16000, 1, 2)
            speech = source.readframes(min(source.getnframes(), 20 * 16000))
        spoken = (speech * (640000 // len(speech) + 1))[:640000]
        minute = spoken + b'\0' * (40 * 32000)
        source_path = fixture.directory / 'soak.wav'
        with wave.open(str(source_path), 'wb') as out:
            out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            remaining = args.seconds * 32000
            while remaining:
                part = minute[:min(remaining, len(minute))]
                out.writeframesraw(part)
                remaining -= len(part)
        fixture.start()
        code, session = fixture.request('/sessions', 'POST', {
            'request_id': str(uuid.uuid4()),
            'audio_source': f'filesrc location="{source_path}" ! wavparse ! identity sync=true',
            'silence_filter': True,
        })
        assert code == 200, session
        identity = session['id']
        started = time.monotonic()
        samples = []
        args.report.parent.mkdir(parents=True, exist_ok=True)
        while time.monotonic() - started < args.seconds + 10:
            time.sleep(min(15, max(1, args.seconds)))
            code, state = fixture.request(f'/sessions/{identity}')
            assert code == 200, state
            code, levels = fixture.request(f'/sessions/{identity}/levels')
            assert code == 200 and not levels['error'], levels
            row = {'wall_seconds': round(time.monotonic() - started, 2),
                   'rss_kb': resident_kb(fixture.process.pid),
                   'durable_samples': levels['durable_samples'],
                   'capture_state': state['capture_state'],
                   'transcription_state': state['transcription_state']}
            samples.append(row)
            args.report.write_text(json.dumps({'complete': False, 'seconds': args.seconds, 'samples': samples}, indent=2))
            print(json.dumps(row), flush=True)
            assert state['capture_state'] != 'interrupted', state
            if state['capture_state'] != 'recording':
                break
        fixture.request(f'/sessions/{identity}/stop', 'POST', {})
        wait(lambda: fixture.request(f'/sessions/{identity}')[1]['transcription_state'] == 'completed', timeout=30)
        state = fixture.request(f'/sessions/{identity}')[1]
        assert state['durable_samples'] == args.seconds * 16000, state
        steady = [row['rss_kb'] for row in samples if row['wall_seconds'] >= min(60, args.seconds / 2)]
        assert max(steady) - min(steady) < 48 * 1024, 'Capture memory grew beyond bounded working buffers'
        report = {'complete': True, 'seconds': args.seconds, 'samples': samples,
                  'final_session': state, 'steady_rss_growth_kb': max(steady) - min(steady)}
        args.report.write_text(json.dumps(report, indent=2))
        print(f'PASS: {args.seconds}s real-time recording, exact saved frames, bounded memory', flush=True)


if __name__ == '__main__':
    main()
