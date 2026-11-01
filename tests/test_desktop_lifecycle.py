"""Real GTK client and service, isolated display, clipboard, data and CPU workers."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time
from service_test_support import Fixture, REPO, alive, wait

binary = str(REPO / 'paraketto-desktop')
with Fixture() as fixture:
    fixture.start(vad_model=fixture.directory / 'missing-vad-model')
    data = fixture.data
    app = None
    log = (fixture.directory / 'app.log').open('w')
    release = fixture.directory / 'release-transcription'
    release.write_text('continue')

    def command(*args):
        subprocess.run([binary, *args], check=True, capture_output=True, timeout=8, env=fixture.env)

    def start(explicit_url=False):
        args = ['--service-url', fixture.url] if explicit_url else []
        process = subprocess.Popen([binary, *args, '--data-dir', str(data), '--audio-source',
            'audiotestsrc is-live=true wave=sine volume=0.3'], env=fixture.env, stdout=log, stderr=log)
        wait(lambda: subprocess.run(['xdotool', 'search', '--all', '--onlyvisible', '--pid', str(process.pid),
            '--name', '^Paraketto$'], capture_output=True).returncode == 0)
        time.sleep(0.4)
        assert process.poll() is None
        return process

    def sessions():
        code, body = fixture.request('/sessions')
        assert code == 200, body
        return body['sessions']

    def session(sid):
        code, body = fixture.request('/sessions/' + sid)
        assert code == 200, body
        return body

    def active():
        return fixture.status().get('active_session_id', '')

    def publication_ready(sid):
        return (session(sid).get('publication') or {}).get('complete', False)

    def clipboard(value=None):
        if value is not None:
            subprocess.run(['xclip', '-selection', 'clipboard'], input=value.encode(),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=fixture.env, check=True, timeout=3)
        return subprocess.check_output(['xclip', '-selection', 'clipboard', '-o'], env=fixture.env, timeout=3).decode()

    def begin():
        # Text/session deltas are polled at 1 Hz; wait for the previous session's
        # completion to reach GTK before invoking another recording shortcut.
        time.sleep(1.2)
        command('--toggle')
        sid = wait(active)
        wait(lambda: session(sid)['durable_samples'] >= 16000, timeout=6)
        return sid

    try:
        app = start()
        clipboard('Before import')
        previous = {s['id'] for s in sessions()}
        command('--transcribe', str(fixture.audio))
        imported = wait(lambda: next((s['id'] for s in sessions() if s['id'] not in previous), None))
        wait(lambda: publication_ready(imported), timeout=10)
        assert clipboard() == 'Before import', 'Import unexpectedly wrote the clipboard'
        assert not list((data / 'history').glob('*.json')), 'GTK still writes legacy history'

        sid = begin()
        before = session(sid)['durable_samples']
        app.kill(); app.wait(timeout=5)
        wait(lambda: session(sid)['durable_samples'] > before + 16000, timeout=6)
        assert session(sid)['capture_state'] == 'recording', 'GTK crash stopped service capture'
        clipboard('Reconnect must not copy')
        app = start()
        wait(lambda: subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '^Paraketto recording$'],
                                    capture_output=True).returncode == 0)
        assert clipboard() == 'Reconnect must not copy'
        stop_started = time.monotonic()
        command('--toggle')
        wait(lambda: session(sid)['capture_state'] == 'stopped')
        wait(lambda: publication_ready(sid), timeout=12)
        publication_observed = time.monotonic()
        wait(lambda: clipboard() == 'Lifecycle fixture transcript')
        clipboard_observed = time.monotonic()
        print(f'OBSERVED: complete publication to clipboard {(clipboard_observed-publication_observed)*1000:.1f} ms; '
              f'local Stop to clipboard {(clipboard_observed-stop_started)*1000:.1f} ms (one CPU fixture sample)', flush=True)
        clipboard('Do not replay delivery')
        fixture.request('/sessions/' + sid + '/retry', 'POST', {})
        time.sleep(1.5)
        assert clipboard() == 'Do not replay delivery', 'Retry replayed a clipboard write'
        wait(lambda: subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '^Paraketto recording$'],
                                    capture_output=True).returncode != 0, timeout=4)
        command('--cancel')  # Dismiss the finished pill; do not cancel saved history.
        assert session(sid)['capture_state'] == 'stopped'
        print('PASS: service capture survives GTK death; reconnect, Stop and one-time publication delivery')

        cancelled = begin()
        clipboard('Cancelled recording')
        command('--cancel')
        wait(lambda: session(cancelled)['capture_state'] == 'cancelled')
        assert session(cancelled)['durable_samples'] > 0
        time.sleep(0.5)
        assert clipboard() == 'Cancelled recording'

        quitting = begin()
        assert fixture.request('/sessions/' + quitting + '/pause', 'POST', {})[0] == 200
        wait(lambda: session(quitting)['capture_state'] == 'paused')
        paused_samples = session(quitting)['durable_samples']
        app.kill(); app.wait(timeout=5)
        time.sleep(0.4)
        assert session(quitting)['durable_samples'] == paused_samples, 'Paused capture continued after GTK death'
        app = start()
        assert session(quitting)['capture_state'] == 'paused', 'Reconnect silently resumed capture'
        assert clipboard() == 'Cancelled recording', 'Paused reconnect wrote the clipboard'
        assert fixture.request('/sessions/' + quitting + '/resume', 'POST', {})[0] == 200
        wait(lambda: session(quitting)['durable_samples'] > paused_samples + 4000)
        assert fixture.request('/sessions/' + quitting + '/pause', 'POST', {})[0] == 200
        time.sleep(1.2)  # Let GTK observe the externally paused state before Quit.
        command('--quit'); app.wait(timeout=6)
        assert session(quitting)['capture_state'] == 'stopped', 'Explicit Quit did not save capture'
        assert fixture.request('/health')[0] == 200, 'Quit stopped the shared service'
        app = start()
        time.sleep(0.5)
        assert clipboard() == 'Cancelled recording', 'Restart replayed completion delivery'
        print('PASS: cancel retains audio, paused capture survives reconnect, Resume continues, Quit saves paused capture without replay')

        fixture.request('/model/unload', 'POST', {})
        wait(lambda: fixture.status()['status'] == 'sleeping')
        (fixture.directory / 'fail-load').write_text('model unavailable')
        clipboard('Keep this after ASR failure')
        failed = begin()
        wait(lambda: fixture.status()['status'] == 'error')
        assert session(failed)['capture_state'] == 'recording', 'ASR failure stopped microphone capture'
        command('--toggle')
        wait(lambda: session(failed)['capture_state'] == 'stopped')
        wait(lambda: session(failed)['transcription_state'] == 'failed', timeout=12)
        assert clipboard() == 'Keep this after ASR failure'
        (fixture.directory / 'fail-load').unlink()
        fixture.request('/sessions/' + failed + '/retry', 'POST', {})
        wait(lambda: publication_ready(failed), timeout=12)
        time.sleep(1.2)
        assert clipboard() == 'Keep this after ASR failure', 'Recovering a failed session replayed automatic delivery'
        print('PASS: inference failure preserves recording; failed and retried results do not auto-copy')

        fixture.request('/settings', 'PATCH', {'keep_loaded': True})
        fixture.request('/model/load', 'POST', {})
        wait(lambda: fixture.status()['status'] == 'ready')
        worker = fixture.status()['worker_pid']
        release.unlink()
        with ThreadPoolExecutor(max_workers=1) as requests:
            external = requests.submit(fixture.transcribe)
            wait(lambda: fixture.status()['status'] == 'busy')
            command('--quit'); app.wait(timeout=5)
            assert fixture.process.poll() is None and alive(worker), 'Quit interrupted another client'
            release.write_text('continue')
            assert external.result(timeout=8)[0] == 200
        app = start()

        published = data / 'service.json'
        address = published.read_text()
        stale = json.loads(address); stale['pid'] += 1
        published.write_text(json.dumps(stale))
        count = len(sessions())
        command('--toggle'); time.sleep(1)
        assert not active() and len(sessions()) == count, 'Stale discovery sent a capture request'
        published.write_text(address)
        command('--quit'); app.wait(timeout=5)
        published.write_text('{broken')
        app = start(explicit_url=True)
        previous = {s['id'] for s in sessions()}
        command('--transcribe', str(fixture.audio))
        wait(lambda: any(s['id'] not in previous for s in sessions()))
        command('--quit'); app.wait(timeout=5)
        published.write_text(address)
        print('PASS: shared worker ownership, stale discovery rejection and explicit URL override')
    finally:
        if app is not None and app.poll() is None:
            app.kill(); app.wait(timeout=5)
        log.close()
        errors = (fixture.directory / 'app.log').read_text()
        assert 'CRITICAL' not in errors, errors
