"""Exercise the real desktop process and owned-child lifecycle in a private session."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import wave

repo = Path.cwd()
binary = str(repo / 'paraketto-desktop')
fixture = repo / 'tests/desktop_fixture_server.py'

def wait(check, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.025)
    raise AssertionError('Timed out waiting for desktop lifecycle transition')

def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

with tempfile.TemporaryDirectory(prefix='paraketto-lifecycle-') as tmp:
    directory = Path(tmp)
    weights = directory / 'weights'
    weights.write_text('fixture')
    audio = directory / 'sample.wav'
    with wave.open(str(audio), 'wb') as out:
        out.setparams((1, 2, 16000, 16000, 'NONE', 'not compressed'))
        out.writeframes(b'\0\0' * 16000)
    data = directory / 'data'
    data.mkdir()
    preferences = data / 'preferences.ini'
    preferences.write_text('[Model]\nidle-seconds=1\nkeep-loaded=false\n')
    env = dict(os.environ, PARAKETTO_FIXTURE_DIR=str(directory))
    app = None
    log = (directory / 'app.log').open('w')

    def command(*args):
        subprocess.run([binary, *args], check=True, capture_output=True, timeout=8, env=env)

    def action(name):
        subprocess.run(['gdbus', 'call', '--session', '--dest', 'io.localoptima.Paraketto',
            '--object-path', '/io/localoptima/Paraketto', '--method', 'org.gtk.Actions.Activate',
            name, '[]', '{}'], check=True, capture_output=True, timeout=5, env=env)

    def start(model_weights=weights):
        process = subprocess.Popen([binary, '--server-binary', str(fixture), '--weights', str(model_weights),
            '--data-dir', str(data), '--audio-source', 'audiotestsrc is-live=true'], env=env, stdout=log, stderr=log)
        # A remote no-op confirms that GApplication has registered its bus name.
        time.sleep(0.2)
        command('--cancel')
        assert process.poll() is None
        return process

    def server(previous=0):
        try:
            status = json.loads((directory / 'server.json').read_text())
            if status['pid'] == previous or not alive(status['pid']):
                return None
            return status['pid'] if (directory / 'client-ready').read_text() == str(status['pid']) else None
        except (OSError, ValueError):
            return None

    def entries():
        return sorted([json.loads(p.read_text()) for p in (data / 'history').glob('*.json')],
                      key=lambda e: e['id'], reverse=True)

    try:
        app = start()
        assert not (directory / 'server.json').exists(), 'Opening the app loaded the model'
        command('--transcribe', str(audio))
        pid = wait(server)
        wait(lambda: (directory / 'request-started').exists())
        time.sleep(1.2)
        action('model')  # Must be disabled while transcription is in progress.
        assert alive(pid), 'Model was unloaded during transcription'
        (directory / 'release-transcription').write_text('continue')
        wait(lambda: entries() and entries()[0]['text'] == 'Lifecycle fixture transcript')
        wait(lambda: not alive(pid))

        action('model')
        pid = wait(lambda: server(pid))
        action('model')
        wait(lambda: not alive(pid))

        command('--toggle')
        pid = wait(lambda: server(pid))
        time.sleep(1.3)
        action('model')
        assert alive(pid), 'Model was unloaded while recording'
        command('--cancel')
        wait(lambda: not alive(pid))
        command('--quit'); app.wait(timeout=5)

        preferences.write_text('[Model]\nidle-seconds=1\nkeep-loaded=true\n')
        app = start()
        action('model')
        pid = wait(lambda: server(pid))
        time.sleep(1.5)
        assert alive(pid), 'Keep-loaded preference was ignored'
        action('model')
        wait(lambda: not alive(pid))
        command('--quit'); app.wait(timeout=5)

        app = start(directory / 'missing-weights')
        command('--transcribe', str(audio))
        wait(lambda: entries()[0]['error'] == 'Couldn’t load model')
        assert 'Weights not found' in (data / 'server.log').read_text()
        assert not alive(pid), 'Failed load left an inference process running'
        command('--quit'); app.wait(timeout=5)
        print('PASS: cold start, queued transcription, idle unload, busy guards, manual load/unload, persisted keep-loaded, and load failure')
    finally:
        if app is not None and app.poll() is None:
            app.send_signal(signal.SIGTERM)
            app.wait(timeout=8)
        log.close()
        errors = (directory / 'app.log').read_text()
        assert 'CRITICAL' not in errors, errors
