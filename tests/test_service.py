"""Exercise the public API, multiple clients, failure recovery, and process ownership."""
from concurrent.futures import ThreadPoolExecutor
import signal
import os
import time
import uuid
from service_test_support import Fixture, alive, wait

with Fixture() as fixture:
    fixture.start()
    pid = fixture.process.pid
    assert fixture.request('/health')[0] == 200
    assert b'<title>paraketto</title>' in fixture.request('/')[1]
    assert not (fixture.directory / 'spawned').exists(), 'Service startup loaded the model'
    assert fixture.request('/model/load', 'POST', {}, {'Origin': 'https://unrelated.example'})[0] == 403
    assert fixture.request('/settings', 'PATCH', {'idle_seconds': -1})[0] == 400
    assert fixture.request('/settings', 'PATCH', {'keep_loaded': 'yes'})[0] == 400
    assert fixture.send('/model/load', 'POST', b'{broken', {'Content-Type': 'application/json'})[0] == 400
    assert fixture.request('/model/load', 'POST', {'lease_id': 123})[0] == 400
    assert fixture.request('/model/load', 'POST', {'lease_id': ''})[0] == 400
    assert fixture.request('/transcribe', 'POST', {})[0] == 400
    assert fixture.request('/shutdown', 'POST', {})[0] == 404

    with ThreadPoolExecutor(max_workers=5) as requests:
        first = requests.submit(fixture.transcribe)
        wait(lambda: fixture.status()['status'] == 'busy')
        worker = fixture.status()['worker_pid']
        other = [requests.submit(fixture.transcribe) for _ in range(3)]
        wait(lambda: fixture.status()['pending_requests'] == 4)
        assert fixture.transcribe()[0] == 429
        time.sleep(1.2)
        assert fixture.request('/model/unload', 'POST', {})[0] == 409
        assert alive(worker), 'Idle timeout interrupted active or queued requests'
        assert fixture.status()['pid'] == pid, 'Status unavailable while transcription is blocked'
        (fixture.directory / 'release-transcription').write_text('continue')
        for result in [first, *other]:
            code, body = result.result(timeout=10)
            assert code == 200 and body['text'] == 'Lifecycle fixture transcript'
    wait(lambda: fixture.status()['status'] == 'sleeping')
    assert not alive(worker)
    assert fixture.status()['pid'] == pid, 'Model unload stopped the API'
    assert len(list((fixture.directory / 'spawned').iterdir())) == 1, 'Concurrent clients loaded duplicate models'

    (fixture.directory / 'release-transcription').unlink()
    with ThreadPoolExecutor(max_workers=2) as requests:
        first = requests.submit(fixture.transcribe)
        wait(lambda: fixture.status()['status'] == 'busy')
        other = requests.submit(fixture.transcribe)
        wait(lambda: fixture.status()['pending_requests'] == 2)
        os.kill(fixture.status()['worker_pid'], signal.SIGKILL)
        assert first.result(timeout=5)[0] in (502, 503)
        assert other.result(timeout=5)[0] == 503
    wait(lambda: fixture.status()['status'] == 'error')
    (fixture.directory / 'release-transcription').write_text('continue')
    assert fixture.transcribe()[0] == 200, 'A failed request left the service stuck'
    wait(lambda: fixture.status()['status'] == 'sleeping')

    lease = str(uuid.uuid4())
    fixture.request('/model/load', 'POST', {'lease_id': lease})
    wait(lambda: fixture.status()['status'] == 'ready')
    worker = fixture.status()['worker_pid']
    time.sleep(1.2)
    assert alive(worker)
    assert fixture.request('/model/unload', 'POST', {})[0] == 409
    fixture.request('/model/release', 'POST', {'lease_id': lease})
    wait(lambda: fixture.status()['status'] == 'sleeping')
    assert not alive(worker)

    fixture.request('/settings', 'PATCH', {'keep_loaded': True, 'idle_seconds': 1})
    fixture.request('/model/load', 'POST', {})
    wait(lambda: fixture.status()['status'] == 'ready')
    worker = fixture.status()['worker_pid']
    time.sleep(1.2)
    assert alive(worker)
    os.kill(worker, signal.SIGKILL)
    wait(lambda: fixture.status()['status'] == 'error')
    assert fixture.transcribe()[0] == 200, 'A new request did not recover from worker failure'
    fixture.request('/model/unload', 'POST', {})
    wait(lambda: fixture.status()['status'] == 'sleeping')
    fixture.stop()
    fixture.start(missing=True)
    assert fixture.status()['keep_loaded'] is True, 'Service preferences did not persist'
    assert fixture.transcribe()[0] == 503
    assert fixture.status()['load_failed'] is True
    assert fixture.request('/health')[0] == 200, 'Load failure killed the API'
    assert 'Weights not found' in (fixture.data / 'server.log').read_text()
    (fixture.directory / 'missing-weights').write_text('fixed')
    assert fixture.transcribe()[0] == 200, 'Failed model could not be retried'
print('PASS: stable API, lazy load, concurrent clients, bounded queue, idle/manual unload, recording leases, settings, worker crash, and failed-load recovery')

with Fixture(PARAKETTO_FIXTURE_DELAY='20', PARAKETTO_FIXTURE_IGNORE_TERM='1') as fixture:
    fixture.start(load_timeout=1)
    assert fixture.transcribe()[0] == 503
    wait(lambda: all(not alive(int(p.name)) for p in (fixture.directory / 'spawned').iterdir()))
    assert fixture.request('/health')[0] == 200
print('PASS: bounded loading and termination of an unresponsive worker')

with Fixture(PARAKETTO_FIXTURE_DELAY='20', PARAKETTO_FIXTURE_IGNORE_TERM='1') as fixture:
    fixture.start(load_timeout=10)
    fixture.request('/model/load', 'POST', {})
    wait(lambda: list((fixture.directory / 'spawned').glob('*')))
    assert fixture.request('/model/unload', 'POST', {})[0] == 202
    wait(lambda: fixture.status()['status'] == 'sleeping', timeout=4)
print('PASS: manual cancellation while a worker is still loading')

with Fixture(PARAKETTO_FIXTURE_IGNORE_TERM='1') as fixture:
    fixture.start()
    with ThreadPoolExecutor(max_workers=1) as requests:
        request = requests.submit(fixture.transcribe)
        wait(lambda: fixture.status()['status'] == 'busy')
        worker = fixture.status()['worker_pid']
        fixture.stop()
        assert request.result(timeout=3)[0] == 503
        assert not alive(worker)
print('PASS: service shutdown resolves pending requests and reaps the worker')

with Fixture() as fixture:
    fixture.start()
    fixture.request('/model/load', 'POST', {'lease_id': str(uuid.uuid4())})
    wait(lambda: fixture.status()['status'] == 'ready')
    worker = fixture.status()['worker_pid']
    wait(lambda: fixture.status()['status'] == 'sleeping', timeout=14)
    assert not alive(worker), 'An abandoned recording lease held the model forever'
print('PASS: crashed recording client leases expire')
