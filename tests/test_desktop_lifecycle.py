"""Exercise the real desktop as a client of a separately owned local service."""
from concurrent.futures import ThreadPoolExecutor
import json
import socket
import subprocess
import time
from service_test_support import Fixture, REPO, alive, wait

binary = str(REPO / 'paraketto-desktop')
with Fixture() as fixture:
    fixture.start()
    data = fixture.data
    app = None
    log = (fixture.directory / 'app.log').open('w')

    def command(*args):
        subprocess.run([binary, *args], check=True, capture_output=True, timeout=8, env=fixture.env)

    def action(name):
        # Controls reflect the client's 250 ms status poll plus a GTK frame.
        time.sleep(0.35)
        subprocess.run(['gdbus', 'call', '--session', '--dest', 'io.localoptima.Paraketto',
            '--object-path', '/io/localoptima/Paraketto', '--method', 'org.gtk.Actions.Activate',
            name, '[]', '{}'], check=True, capture_output=True, timeout=5, env=fixture.env)

    def start(explicit_url=False):
        service_args = ['--service-url', fixture.url] if explicit_url else []
        process = subprocess.Popen([binary, *service_args, '--data-dir', str(data),
            '--audio-source', 'audiotestsrc is-live=true'], env=fixture.env, stdout=log, stderr=log)
        wait(lambda: subprocess.run(['xdotool', 'search', '--all', '--onlyvisible', '--pid', str(process.pid),
            '--name', '^Paraketto$'], capture_output=True).returncode == 0)
        time.sleep(0.3)  # Let the first API poll update the desktop actions.
        assert process.poll() is None
        return process

    def entries():
        return sorted([json.loads(p.read_text()) for p in (data / 'history').glob('*.json')],
                      key=lambda e: e['id'], reverse=True)

    try:
        app = start()
        assert fixture.status()['status'] == 'sleeping'
        command('--transcribe', str(fixture.audio))
        wait(lambda: fixture.status()['status'] == 'busy')
        worker = fixture.status()['worker_pid']
        time.sleep(1.2)
        action('model')
        assert alive(worker), 'Desktop interrupted active transcription'
        (fixture.directory / 'release-transcription').write_text('continue')
        wait(lambda: entries() and entries()[0]['text'] == 'Lifecycle fixture transcript')
        wait(lambda: fixture.status()['status'] == 'sleeping')
        assert not alive(worker)

        command('--toggle')
        wait(lambda: fixture.status()['status'] == 'ready')
        worker = fixture.status()['worker_pid']
        time.sleep(2.5)
        assert alive(worker), 'Recording did not renew its model reservation'
        command('--cancel')
        wait(lambda: fixture.status()['status'] == 'sleeping')

        fixture.request('/settings', 'PATCH', {'keep_loaded': True})
        action('model')
        wait(lambda: fixture.status()['status'] == 'ready')
        worker = fixture.status()['worker_pid']
        (fixture.directory / 'release-transcription').unlink()
        with ThreadPoolExecutor(max_workers=1) as requests:
            external = requests.submit(fixture.transcribe)
            wait(lambda: fixture.status()['status'] == 'busy')
            command('--quit'); app.wait(timeout=5)
            assert fixture.process.poll() is None and alive(worker), 'Quitting the UI stopped shared inference'
            (fixture.directory / 'release-transcription').write_text('continue')
            assert external.result(timeout=5)[0] == 200
        app = start()
        assert fixture.status()['worker_pid'] == worker, 'Restarting the UI loaded another model'
        action('model')
        wait(lambda: fixture.status()['status'] == 'sleeping')

        fixture.stop()
        time.sleep(0.5)
        assert app.poll() is None, 'Desktop exited when the service stopped'
        fixture.start(missing=True)
        command('--transcribe', str(fixture.audio))
        wait(lambda: entries()[0]['error'] == 'Couldn’t load model')
        assert 'Weights not found' in (fixture.data / 'server.log').read_text()
        old_port = fixture.port
        fixture.stop()
        with socket.socket() as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(('127.0.0.1', old_port)); occupied.listen()
            fixture.port = 0
            fixture.start()
            assert fixture.port != old_port
        previous = entries()[0]['id']
        command('--transcribe', str(fixture.audio))
        wait(lambda: entries()[0]['id'] != previous and entries()[0]['text'] == 'Lifecycle fixture transcript')
        published = fixture.data / 'service.json'
        address = published.read_text()
        stale = json.loads(address)
        stale['pid'] += 1
        published.write_text(json.dumps(stale))
        previous = entries()[0]['id']
        command('--transcribe', str(fixture.audio))
        wait(lambda: entries()[0]['id'] != previous and entries()[0]['error'] == 'Service address is out of date')
        published.write_text(address)
        wait(lambda: fixture.status()['recording_leases'] == 0)
        fixture.request('/model/unload', 'POST', {})
        wait(lambda: fixture.status()['status'] == 'sleeping')
        (fixture.directory / 'fail-load').write_text('fail during recording')
        starts = len(list((fixture.directory / 'spawned').iterdir()))
        previous = entries()[0]['id']
        command('--toggle')
        wait(lambda: entries()[0]['id'] != previous and entries()[0]['error'] == 'Couldn’t load model')
        wait(lambda: fixture.status()['recording_leases'] == 0)
        assert len(list((fixture.directory / 'spawned').iterdir())) == starts + 1, 'A failed preload was silently retried'
        command('--quit'); app.wait(timeout=5)
        assert fixture.request('/health')[0] == 200
        (fixture.directory / 'fail-load').unlink()
        published.write_text('{broken')
        app = start(explicit_url=True)
        previous = entries()[0]['id']
        command('--transcribe', str(fixture.audio))
        wait(lambda: entries()[0]['id'] != previous and entries()[0]['text'] == 'Lifecycle fixture transcript')
        command('--quit'); app.wait(timeout=5)
        print('PASS: desktop API client, automatic port discovery/reconnection, stale-address rejection, explicit URL override, recording reservations, independent lifetimes, and failure recovery')
    finally:
        if app is not None and app.poll() is None:
            app.terminate(); app.wait(timeout=8)
        log.close()
        errors = (fixture.directory / 'app.log').read_text()
        assert 'CRITICAL' not in errors, errors
