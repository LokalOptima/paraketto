"""Private CPU-only end-to-end checks for durable recording sessions.

Uses audiotestsrc and the existing HTTP worker fixture, never the user's
microphone, active service, display, clipboard, or GPU inference.
"""
from pathlib import Path
import io
import concurrent.futures
import json
import hashlib
import os
import signal
import sqlite3
import time
import uuid
import wave
from service_test_support import Fixture, wait


def expect(fixture, path, method='GET', value=None, code=200):
    actual, body = fixture.request(path, method, value)
    assert actual == code, (path, actual, body)
    return body


def session(fixture, identity):
    return expect(fixture, f'/sessions/{identity}')


def complete(fixture, identity, timeout=12):
    return wait(lambda: (state := session(fixture, identity)).get('publication') and state, timeout)


def start(fixture, key=None):
    key = key or str(uuid.uuid4())
    payload = {'request_id': key, 'audio_source': 'audiotestsrc is-live=true wave=sine volume=0.02',
               'silence_filter': False}
    result = expect(fixture, '/sessions', 'POST', payload)
    assert result['capture_state'] == 'recording'
    assert result['delivery_epoch'] == 0
    return result['id'], payload


def release(fixture):
    (fixture.directory / 'release-transcription').write_text('continue')


def make_wav(path, seconds):
    # The service must ingest incrementally; the test also writes bounded buffers.
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 16000, seconds * 16000, 'NONE', 'not compressed'))
        for _ in range(seconds):
            audio.writeframesraw(b'\0\0' * 16000)


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    status = fixture.status()
    assert status['capabilities']['sessions'] is True
    assert set(status['models']) == {'asr'}
    assert set(status['queues']) == {'asr'}
    assert 'cleanup_enabled' not in status
    for operation in ('load', 'unload'):
        expect(fixture, f'/models/cleanup/{operation}', 'POST', {}, 404)
    with sqlite3.connect(fixture.data / 'sessions.sqlite3') as database:
        tables = {name for (name,) in database.execute("select name from sqlite_schema where type='table'")}
        assert not tables.intersection({'cleanup_runs', 'retired_cleanup_runs'})
    dictionary = expect(fixture, '/dictionary', 'PUT', {'entries': [
        {'canonical': 'Session', 'aliases': ['Lifecycle']}]})
    assert dictionary['revision'] > 1
    expect(fixture, '/dictionary', 'PUT', {'entries': [
        {'canonical': 'A', 'aliases': ['ambiguous']},
        {'canonical': 'B', 'aliases': ['AMBIGUOUS']}]}, 400)
    extended = expect(fixture, '/dictionary/entries', 'POST', {'canonical': 'Paraketto', 'aliases': ['paraketo']})
    assert {e['canonical'] for e in extended['entries']} == {'Session', 'Paraketto'}
    assert expect(fixture, '/dictionary/entries', 'POST', {'canonical': 'Paraketto', 'aliases': ['paraketo']})['revision'] == extended['revision']
    expect(fixture, '/dictionary', 'PUT', {'entries': [], 'expected_revision': dictionary['revision']}, 409)
    expect(fixture, '/dictionary/entries', 'POST', {'canonical': 'Wrong', 'aliases': ['PARAKETO']}, 400)
    assert expect(fixture, '/dictionary')['revision'] == extended['revision']
    index = next(i for i, entry in enumerate(extended['entries']) if entry['canonical'] == 'Paraketto')
    path = f'/dictionary/entries/{index}'
    expect(fixture, path, 'PUT', {'canonical': 'ParaKit', 'aliases': ['paraketo']}, 400)
    changed = expect(fixture, path, 'PUT', {'canonical': 'ParaKit', 'aliases': ['paraketo', 'para; kit'],
        'expected_revision': extended['revision']})
    assert changed['entries'][index] == {'canonical': 'ParaKit', 'aliases': ['paraketo', 'para; kit']}
    assert changed['entries'][0] == dictionary['entries'][0]
    expect(fixture, path, 'DELETE', {'expected_revision': extended['revision']}, 409)
    expect(fixture, path, 'PUT', {'canonical': 'Wrong', 'aliases': ['Lifecycle'],
        'expected_revision': changed['revision']}, 400)
    assert expect(fixture, '/dictionary')['revision'] == changed['revision']
    removed = expect(fixture, path, 'DELETE', {'expected_revision': changed['revision']})
    assert removed['entries'] == dictionary['entries']
    expect(fixture, path, 'DELETE', {'expected_revision': removed['revision']}, 404)
    expect(fixture, '/dictionary/entries', 'POST', {'canonical': 'Paraketto', 'aliases': ['paraketo']})

    identity, request = start(fixture)
    assert expect(fixture, '/sessions', 'POST', request)['id'] == identity
    expect(fixture, '/sessions', 'POST', {**request, 'request_id': str(uuid.uuid4())}, 409)
    wait(lambda: session(fixture, identity)['durable_samples'] >= 16000)
    levels = expect(fixture, f'/sessions/{identity}/levels')
    assert levels['captured_samples'] >= levels['durable_samples'] >= 16000
    assert levels['rms'] > 0
    expect(fixture, f'/sessions/{identity}/stop', 'POST', {})
    state = complete(fixture, identity)
    assert state['capture_state'] == 'stopped' and state['transcription_state'] == 'completed'
    publication = expect(fixture, f'/sessions/{identity}/publication')
    assert publication['text'] == 'Session fixture transcript'
    assert fixture.request(f'/sessions/{identity}/transcript?view=original')[1] == b'Lifecycle fixture transcript'
    assert fixture.request(f'/sessions/{identity}/transcript?view=edited')[1] == b'Session fixture transcript'
    assert expect(fixture, f'/sessions/{identity}/stop', 'POST', {})['publication']['id'] == publication['id']
    code, audio = fixture.request(f'/sessions/{identity}/audio?start=0&end=16000')
    assert code == 200 and len(audio) == 32044
    with wave.open(io.BytesIO(audio), 'rb') as decoded:
        assert decoded.getnframes() == 16000 and decoded.getframerate() == 16000
    changes = expect(fixture, f'/sessions/{identity}/updates')
    revision = session(fixture, identity)['revision']
    disabled = expect(fixture, f'/sessions/{identity}/replacements', 'PATCH', {
        'enabled': False, 'expected_revision': revision})
    assert disabled['replacements_enabled'] is False
    assert disabled['delivery_epoch'] == state['delivery_epoch']
    expect(fixture, f'/sessions/{identity}/replacements', 'PATCH', {
        'enabled': False, 'expected_revision': revision})  # Idempotent despite the old revision.
    expect(fixture, f'/sessions/{identity}/replacements', 'PATCH', {
        'enabled': True, 'expected_revision': revision}, 409)
    for invalid in ({}, {'enabled': 'false'}, {'enabled': False, 'expected_revision': -1},
                    {'enabled': False, 'expected_revision': 0.5}):
        expect(fixture, f'/sessions/{identity}/replacements', 'PATCH', invalid, 400)
    expect(fixture, f'/sessions/{identity}/edits', 'PATCH', {'enabled': True}, 404)
    assert fixture.request(f'/sessions/{identity}/transcript?view=edited')[1] == b'Lifecycle fixture transcript'
    assert expect(fixture, f'/sessions/{identity}/publication')['text'] == 'Session fixture transcript'
    later = expect(fixture, f'/sessions/{identity}/updates?after={changes["cursor"]}')
    assert later['segments'] and all(not edit['enabled'] for segment in later['segments'] for edit in segment['edits'])
    enabled = expect(fixture, f'/sessions/{identity}/replacements', 'PATCH', {'enabled': True})
    assert enabled['replacements_enabled'] is True
    assert fixture.request(f'/sessions/{identity}/transcript?view=edited')[1] == b'Session fixture transcript'

    original_generation = state['generation']
    changed = expect(fixture, f'/sessions/{identity}/reprocess', 'POST', {'audio': True, 'silence_filter': False})
    assert changed['generation'] == original_generation + 1
    assert changed['delivery_epoch'] > state['delivery_epoch']
    complete(fixture, identity)
    reset = expect(fixture, f'/sessions/{identity}/updates?after={later["cursor"]}&generation={original_generation}')
    assert reset['reset'] and reset['generation'] == original_generation + 1
    # Publication copies are immutable and are never replayed by the service;
    # the desktop separately decides whether a new Stop completion is eligible.
    assert expect(fixture, f'/sessions/{identity}/publication')['id'] != publication['id']
    history = expect(fixture, '/sessions?limit=1')
    assert len(history['sessions']) == 1 and history['stats']['words'] == 3

    cancelled, _ = start(fixture)
    wait(lambda: session(fixture, cancelled)['durable_samples'] > 0)
    expect(fixture, f'/sessions/{cancelled}/cancel', 'POST', {})
    assert session(fixture, cancelled)['capture_state'] == 'cancelled'
    expect(fixture, f'/sessions/{cancelled}/publication', code=409)
    expect(fixture, f'/sessions/{cancelled}', 'DELETE')
    wait(lambda: fixture.request(f'/sessions/{cancelled}')[0] == 404)
    wait(lambda: not (fixture.data / 'sessions' / cancelled).exists())
    expect(fixture, f'/sessions/{identity}', 'DELETE')
    expect(fixture, '/sessions', 'POST', request, 410)
print('PASS: session capture, durable audio, idempotence, dictionary edits, snapshots, reprocessing, cancellation and deletion')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    identity, request = start(fixture)
    initial = session(fixture, identity)
    wait(lambda: expect(fixture, f'/sessions/{identity}/levels')['captured_samples'] >= 2048)
    paused = expect(fixture, f'/sessions/{identity}/pause', 'POST', {})
    assert paused['capture_state'] == 'paused' and paused['publication'] is None
    assert paused['durable_samples'] == paused['planned_samples'] > 0
    assert paused['delivery_epoch'] == initial['delivery_epoch']
    assert fixture.status()['active_session_id'] == identity
    levels = expect(fixture, f'/sessions/{identity}/levels')
    assert levels['paused'] and not levels['speech_active'] and levels['rms'] == 0
    assert levels['captured_samples'] == levels['written_samples'] == levels['durable_samples'] == paused['durable_samples']
    assert expect(fixture, '/sessions', 'POST', request)['id'] == identity
    expect(fixture, '/sessions', 'POST', {**request, 'request_id': str(uuid.uuid4())}, 409)
    expect(fixture, f'/sessions/{identity}/reprocess', 'POST', {'audio': True}, 409)
    expect(fixture, f'/sessions/{identity}/publication', code=409)
    wait(lambda: expect(fixture, f'/sessions/{identity}/updates')['segments'][0]['raw_text'] == 'Lifecycle fixture transcript')
    again = expect(fixture, f'/sessions/{identity}/pause', 'POST', {})
    assert again['durable_samples'] == paused['durable_samples']
    assert expect(fixture, f'/sessions/{identity}/levels')['captured_samples'] == levels['captured_samples']
    resumed = expect(fixture, f'/sessions/{identity}/resume', 'POST', {})
    assert resumed['capture_state'] == 'recording' and resumed['delivery_epoch'] == initial['delivery_epoch']
    assert expect(fixture, f'/sessions/{identity}/resume', 'POST', {})['capture_state'] == 'recording'
    wait(lambda: expect(fixture, f'/sessions/{identity}/levels')['captured_samples'] >= levels['captured_samples'] + 2048)
    second = expect(fixture, f'/sessions/{identity}/pause', 'POST', {})
    assert second['durable_samples'] == second['planned_samples'] > paused['durable_samples']
    assert not second['gaps'] and second['publication'] is None
    stopped = expect(fixture, f'/sessions/{identity}/stop', 'POST', {})
    assert stopped['capture_state'] == 'stopped' and stopped['durable_samples'] == second['durable_samples']
    finished = complete(fixture, identity)
    assert finished['transcription_state'] == 'completed' and not finished['gaps']
    segments = expect(fixture, f'/sessions/{identity}/updates')['segments']
    assert len(segments) == 2
    with sqlite3.connect(fixture.data / 'sessions.sqlite3') as database:
        coverage = database.execute('select coverage_start,coverage_end from segments where session_id=? order by sequence', (identity,)).fetchall()
    assert coverage == [(0, paused['durable_samples']), (paused['durable_samples'], second['durable_samples'])]
    expect(fixture, f'/sessions/{identity}/resume', 'POST', {}, 409)
    expect(fixture, f'/sessions/{identity}/pause', 'POST', {}, 409)
    cancelled, _ = start(fixture)
    expect(fixture, f'/sessions/{cancelled}/pause', 'POST', {})
    expect(fixture, f'/sessions/{cancelled}/cancel', 'POST', {})
    assert session(fixture, cancelled)['capture_state'] == 'cancelled'
    expect(fixture, f'/sessions/{cancelled}/publication', code=409)
print('PASS: recording pause flushes speech without publication, Resume preserves one timeline, Stop and Cancel work while paused')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    identity, _ = start(fixture)
    wait(lambda: expect(fixture, f'/sessions/{identity}/levels')['captured_samples'] >= 2048)
    paused = expect(fixture, f'/sessions/{identity}/pause', 'POST', {})
    fixture.process.kill()
    fixture.stop()
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    recovered = complete(fixture, identity)
    assert recovered['capture_state'] == 'interrupted'
    assert recovered['durable_samples'] == recovered['planned_samples'] == paused['durable_samples']
    assert recovered['delivery_epoch'] > paused['delivery_epoch']
    assert fixture.status()['active_session_id'] == '' and not recovered['gaps']
    expect(fixture, f'/sessions/{identity}/resume', 'POST', {}, 409)
print('PASS: an interrupted paused session recovers its exact checkpoint without reopening capture')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    identity, _ = start(fixture)
    wait(lambda: expect(fixture, f'/sessions/{identity}/levels')['captured_samples'] >= 2048)
    with sqlite3.connect(fixture.data / 'sessions.sqlite3') as database:
        database.execute("create trigger pause_boundary_failure before insert on segments begin select raise(abort,'pause boundary fixture'); end")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pausing = pool.submit(fixture.request, f'/sessions/{identity}/pause', 'POST', {}, timeout=15)
        wait(lambda: session(fixture, identity)['capture_state'] == 'paused')
        wait(lambda: 'pause boundary fixture' in fixture.status()['session_error'])
        assert not pausing.done(), 'Pause must not acknowledge a rejected source boundary'
        resuming = pool.submit(fixture.request, f'/sessions/{identity}/resume', 'POST', {}, timeout=15)
        paused_levels = expect(fixture, f'/sessions/{identity}/levels')
        assert paused_levels['paused'] and not resuming.done()
        assert session(fixture, identity)['planned_samples'] < paused_levels['durable_samples']
        with sqlite3.connect(fixture.data / 'sessions.sqlite3') as database:
            database.execute('drop trigger pause_boundary_failure')
        code, paused = pausing.result(timeout=15)
        assert code == 200 and paused['planned_samples'] == paused_levels['durable_samples']
        code, resumed = resuming.result(timeout=15)
        assert code == 200 and resumed['capture_state'] == 'recording'
    wait(lambda: expect(fixture, f'/sessions/{identity}/levels')['captured_samples'] > paused_levels['captured_samples'])
    with sqlite3.connect(fixture.data / 'sessions.sqlite3') as database:
        database.execute("create trigger pause_state_failure before update of capture_state on sessions when NEW.capture_state='paused' begin select raise(abort,'pause state fixture'); end")
    expect(fixture, f'/sessions/{identity}/pause', 'POST', {}, 503)
    paused_levels = expect(fixture, f'/sessions/{identity}/levels')
    assert paused_levels['paused'] and session(fixture, identity)['capture_state'] == 'recording'
    with sqlite3.connect(fixture.data / 'sessions.sqlite3') as database:
        database.execute('drop trigger pause_state_failure')
    expect(fixture, f'/sessions/{identity}/resume', 'POST', {})
    wait(lambda: expect(fixture, f'/sessions/{identity}/levels')['captured_samples'] > paused_levels['captured_samples'])
    expect(fixture, f'/sessions/{identity}/stop', 'POST', {})
    assert complete(fixture, identity)['transcription_state'] == 'completed'
print('PASS: Pause and Resume preserve source boundaries across failed planning and capture-state writes')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    audio = fixture.directory / 'three-minutes.wav'
    make_wav(audio, 181)
    payload = {'request_id': str(uuid.uuid4()), 'path': str(audio), 'silence_filter': False}
    imported = expect(fixture, '/sessions/import', 'POST', payload)
    identity = imported['id']
    state = complete(fixture, identity, 20)
    assert state['durable_samples'] == 181 * 16000
    assert state['planned_samples'] == state['durable_samples']
    assert len(list((fixture.data / 'sessions' / identity / 'audio').glob('*.pcm'))) == 7
    sequence, cursor = [], 0
    while True:
        page = expect(fixture, f'/sessions/{identity}/updates?after={cursor}&limit=1')
        sequence.extend(item['sequence'] for item in page['segments'])
        assert page['cursor'] >= cursor
        cursor = page['cursor']
        if not page['has_more']:
            break
    assert len(sequence) >= 3 and len(sequence) == len(set(sequence))
    assert sequence == sorted(sequence)
    assert expect(fixture, '/sessions/import', 'POST', payload)['id'] == identity
    expect(fixture, '/sessions/import', 'POST', {'request_id': str(uuid.uuid4()), 'path': '/nonexistent'}, 400)
    assert fixture.status()['queues']['asr'] == 0
    expect(fixture, f'/sessions/{identity}', 'DELETE')
    wait(lambda: not (fixture.data / 'sessions' / identity).exists())
    assert audio.exists(), 'Deleting an imported session must preserve the caller original WAV'
print('PASS: bounded WAV ingestion, independent storage blocks, ordered long-file ASR and change pagination')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    identity, _ = start(fixture)
    saved = wait(lambda: (s := session(fixture, identity))['durable_samples'] >= 16000 and s)
    assert saved['planned_samples'] == 0, 'The fixture should crash before its first ASR group'
    fixture.process.kill()
    fixture.stop()
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    recovered = complete(fixture, identity)
    assert recovered['capture_state'] == 'interrupted'
    assert recovered['delivery_epoch'] > saved['delivery_epoch']
    assert recovered['durable_samples'] >= saved['durable_samples']
    assert recovered['planned_samples'] == recovered['durable_samples']
    assert fixture.status()['active_session_id'] == ''
    assert expect(fixture, f'/sessions/{identity}/publication')['text'] == 'Lifecycle fixture transcript'
    # The synced prefix and adopted active-file tail are all source-backed.
    database = sqlite3.connect(fixture.data / 'sessions.sqlite3')
    assert database.execute('select count(*) from segments where session_id=?', (identity,)).fetchone()[0] == 1
    database.close()
print('PASS: abrupt service death recovers unplanned audio and never reopens capture')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    fixture.start(missing=True, vad_model=fixture.directory / 'absent-vad')
    identity, _ = start(fixture)
    wait(lambda: session(fixture, identity)['durable_samples'] >= 16000)
    expect(fixture, f'/sessions/{identity}/stop', 'POST', {})
    wait(lambda: session(fixture, identity)['transcription_state'] == 'failed')
    failed_epoch = session(fixture, identity)['delivery_epoch']
    assert failed_epoch > 0
    expect(fixture, f'/sessions/{identity}/publication', code=409)
    assert session(fixture, identity)['durable_samples'] >= 16000
    (fixture.directory / 'missing-weights').write_text('fixed fixture')
    expect(fixture, f'/sessions/{identity}/retry', 'POST', {'stage': 'asr'})
    state = complete(fixture, identity)
    assert state['transcription_state'] == 'completed'
    assert state['delivery_epoch'] > failed_epoch
    database = sqlite3.connect(fixture.data / 'sessions.sqlite3')
    recorded = json.loads(database.execute('select r.model from transcript_revisions r join segments s on s.revision_id=r.id where s.session_id=?', (identity,)).fetchone()[0])
    database.close()
    assert recorded['weights_status'] == 'verified'
    assert recorded['runtime_status'] == 'verified'
    assert recorded['weights_sha256'] == hashlib.sha256((fixture.directory / 'missing-weights').read_bytes()).hexdigest()
    expect(fixture, f'/sessions/{identity}/retry', 'POST', {'stage': 'not-a-stage'}, 400)
print('PASS: failed ASR retains source audio and explicit retry completes without duplicate segments')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    request = {'request_id': str(uuid.uuid4()), 'path': str(fixture.audio), 'silence_filter': False}
    identity = expect(fixture, '/sessions/import', 'POST', request)['id']
    wait(lambda: fixture.status()['status'] == 'busy')
    expect(fixture, f'/sessions/{identity}', 'DELETE')
    release(fixture)
    wait(lambda: fixture.status()['status'] == 'sleeping')
    expect(fixture, f'/sessions/{identity}', code=404)
    expect(fixture, '/sessions/import', 'POST', request, 410)
    wait(lambda: not (fixture.data / 'sessions' / identity).exists())
    database = sqlite3.connect(fixture.data / 'sessions.sqlite3')
    assert database.execute('select count(*) from transcript_revisions').fetchone()[0] == 0
    database.close()
print('PASS: deletion during inference rejects late completion and removes durable transcript data')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    history = fixture.data / 'history'
    history.mkdir()
    metadata = history / '20250101T123000000000-old.json'
    metadata.write_text(json.dumps({'text': 'Legacy corrected text', 'duration': 1}))
    legacy_audio = metadata.with_suffix('.wav')
    legacy_audio.write_bytes(fixture.audio.read_bytes())
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    entries = expect(fixture, '/sessions')['sessions']
    assert len(entries) == 1 and entries[0]['source'] == 'legacy'
    assert entries[0]['created_at'].startswith('2025-01-01T12:30:00')
    identity = entries[0]['id']
    assert metadata.exists() and legacy_audio.exists(), 'Migration must retain source files'
    detail = expect(fixture, f'/sessions/{identity}/updates')['segments'][0]
    assert detail['provenance'] == 'legacy' and detail['raw_text'] == 'Legacy corrected text'
    expect(fixture, f'/sessions/{identity}', 'DELETE')
    wait(lambda: not metadata.exists() and not legacy_audio.exists())
print('PASS: legacy provenance and dates survive migration; explicit deletion removes owned legacy source files')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    request = {'request_id': str(uuid.uuid4()), 'silence_filter': False,
               'audio_source': 'audiotestsrc is-live=true num-buffers=200 samplesperbuffer=512 wave=sine ! identity drop-probability=0.5'}
    identity = expect(fixture, '/sessions', 'POST', request)['id']
    result = wait(lambda: (s := session(fixture, identity))['transcription_state'] == 'incomplete' and s, 12)
    assert result['gaps'] and result['error']
    expect(fixture, f'/sessions/{identity}/publication', code=409)
    text = fixture.request(f'/sessions/{identity}/transcript?view=original')[1]
    assert b'[Audio gap' in text and b'Lifecycle fixture transcript' in text
    detail = expect(fixture, f'/sessions/{identity}/updates')['segments'][0]
    assert detail['raw_text'] == 'Lifecycle fixture transcript', 'Gap rendering must not modify the immutable source revision'
print('PASS: device-side timestamp gaps terminate as incomplete and stay visible in partial exports')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    history = fixture.data / 'history'
    history.mkdir()
    metadata = history / 'unsafe.json'
    metadata.write_text(json.dumps({'text': 'Retained legacy text', 'duration': 1}))
    linked = metadata.with_suffix('.wav')
    linked.symlink_to(fixture.audio)
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    legacy = expect(fixture, '/sessions')['sessions'][0]['id']
    expect(fixture, f'/sessions/{legacy}', 'DELETE')
    wait(lambda: 'symlink' in str(fixture.status()).lower())
    identity, _ = start(fixture)
    wait(lambda: session(fixture, identity)['durable_samples'] >= 16000)
    expect(fixture, f'/sessions/{identity}/stop', 'POST', {})
    assert complete(fixture, identity)['transcription_state'] == 'completed'
    assert fixture.audio.exists() and linked.is_symlink()
    linked.unlink()  # Resolve only the private fixture's rejected owned path.
    wait(lambda: not metadata.exists())
print('PASS: rejected legacy deletion cannot follow a symlink or starve unrelated transcription')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    malformed = fixture.directory / 'malformed-response'
    malformed.write_text('invalid JSON')
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    identity = expect(fixture, '/sessions/import', 'POST', {
        'request_id': str(uuid.uuid4()), 'path': str(fixture.audio), 'silence_filter': False})['id']
    wait(lambda: session(fixture, identity)['transcription_state'] == 'failed')
    assert session(fixture, identity)['durable_samples'] == 16000
    malformed.unlink()
    expect(fixture, f'/sessions/{identity}/retry', 'POST', {'stage': 'asr'})
    assert complete(fixture, identity)['transcription_state'] == 'completed'
print('PASS: malformed ASR completion fails visibly and can be retried without stranding its future')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    release(fixture)
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    expect(fixture, '/settings', 'PATCH', {'keep_loaded': True})
    expect(fixture, '/model/load', 'POST', {}, 202)
    wait(lambda: fixture.status()['status'] == 'ready')
    resident = expect(fixture, '/info')['model_revision']
    assert resident['weights_sha256'] == hashlib.sha256(fixture.weights.read_bytes()).hexdigest()
    fixture.weights.write_text('replacement model contents')
    assert expect(fixture, '/info')['model_revision'] == resident, 'A live worker must retain the identity it actually loaded'
    request = {'request_id': str(uuid.uuid4()), 'path': str(fixture.audio), 'silence_filter': False}
    original = expect(fixture, '/sessions/import', 'POST', request)['id']
    complete(fixture, original)
    expect(fixture, '/model/unload', 'POST', {}, 202)
    wait(lambda: fixture.status()['status'] == 'sleeping')
    expect(fixture, '/model/load', 'POST', {}, 202)
    wait(lambda: fixture.status()['status'] == 'ready')
    reloaded = expect(fixture, '/info')['model_revision']
    assert reloaded['weights_sha256'] == hashlib.sha256(fixture.weights.read_bytes()).hexdigest()
    assert reloaded['weights_sha256'] != resident['weights_sha256']
    replacement = expect(fixture, '/sessions/import', 'POST', {**request, 'request_id': str(uuid.uuid4())})['id']
    complete(fixture, replacement)
    database = sqlite3.connect(fixture.data / 'sessions.sqlite3')
    def provenance(identity):
        return json.loads(database.execute('select r.model from transcript_revisions r join segments s on s.revision_id=r.id where s.session_id=? and s.generation=1', (identity,)).fetchone()[0])
    assert provenance(original) == resident
    assert provenance(replacement) == reloaded
    database.close()
print('PASS: ASR provenance stays attached to resident weights and refreshes after a verified reload')


with Fixture(PARAKETTO_FIXTURE_DELAY='0') as fixture:
    fixture.start(vad_model=fixture.directory / 'absent-vad')
    expect(fixture, '/settings', 'PATCH', {'keep_loaded': True})
    request = {'request_id': str(uuid.uuid4()), 'path': str(fixture.audio), 'silence_filter': False}
    identity = expect(fixture, '/sessions/import', 'POST', request)['id']
    wait(lambda: fixture.status()['status'] == 'busy')
    os.kill(fixture.status()['worker_pid'], signal.SIGKILL)
    wait(lambda: session(fixture, identity)['transcription_state'] == 'failed')
    fixture.weights.write_text('different model after failed request')
    release(fixture)
    expect(fixture, '/model/load', 'POST', {}, 202)
    wait(lambda: fixture.status()['status'] == 'ready')
    expect(fixture, f'/sessions/{identity}/retry', 'POST', {'stage': 'asr'})
    wait(lambda: session(fixture, identity)['transcription_state'] == 'failed')
    expect(fixture, f'/sessions/{identity}/publication', code=409)
    database = sqlite3.connect(fixture.data / 'sessions.sqlite3')
    assert database.execute('select count(*) from transcript_revisions').fetchone()[0] == 0
    database.close()
    expect(fixture, f'/sessions/{identity}/reprocess', 'POST', {'audio': True, 'silence_filter': False})
    assert complete(fixture, identity)['generation'] == 2
print('PASS: retries cannot silently change a known ASR model revision; explicit reprocessing can select the new loaded model')
