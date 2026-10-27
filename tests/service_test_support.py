"""Private, CPU-only service fixture shared by API and desktop integration tests."""
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

REPO = Path(__file__).resolve().parent.parent

def wait(check, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.025)
    raise AssertionError('Timed out waiting for service lifecycle transition')

def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

class Fixture:
    def __init__(self, **environment):
        self.temp = tempfile.TemporaryDirectory(prefix='paraketto-service-test-')
        self.directory = Path(self.temp.name)
        self.data = self.directory / 'service'
        self.data.mkdir()
        self.preferences = self.data / 'preferences.ini'
        self.preferences.write_text('[Model]\nidle-seconds=1\nkeep-loaded=false\n')
        self.weights = self.directory / 'weights'
        self.weights.write_text('fixture')
        self.audio = self.directory / 'sample.wav'
        with wave.open(str(self.audio), 'wb') as out:
            out.setparams((1, 2, 16000, 16000, 'NONE', 'not compressed'))
            out.writeframes(b'\0\0' * 16000)
        self.env = dict(os.environ, PARAKETTO_FIXTURE_DIR=str(self.directory), **environment)
        self.process = None
        self.port = 0
        self.log = (self.directory / 'service.log').open('w')

    def start(self, missing=False, load_timeout=5):
        self.process = subprocess.Popen([str(REPO / 'paraketto-service'), '--port', str(self.port),
            '--worker-binary', str(REPO / 'tests/desktop_fixture_server.py'), '--data-dir', str(self.data),
            '--weights', str(self.directory / 'missing-weights' if missing else self.weights),
            '--load-timeout', str(load_timeout)], env=self.env, stdout=subprocess.PIPE, stderr=self.log, text=True)
        assert select.select([self.process.stdout], [], [], 8)[0], 'Service startup timed out'
        line = self.process.stdout.readline()
        assert line, (self.directory / 'service.log').read_text()
        ready = json.loads(line)
        self.port = ready['port']
        self.url = f'http://127.0.0.1:{self.port}'
        wait(lambda: self.status()['status'] == 'sleeping')
        return self

    def request(self, path, method='GET', value=None, headers=None, timeout=8):
        body = json.dumps(value).encode() if value is not None else None
        request_headers = {'Content-Type': 'application/json', **(headers or {})}
        return self.send(path, method, body, request_headers, timeout)

    def send(self, path, method, body, headers, timeout=8):
        try:
            response = urlopen(Request(self.url + path, body, headers, method=method), timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            data = response.read()
            return response.status, json.loads(data) if 'application/json' in response.headers.get('Content-Type', '') else data

    def status(self):
        code, data = self.request('/status')
        assert code == 200
        return data

    def transcribe(self):
        boundary = 'paraketto-fixture-boundary'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
                'Content-Type: audio/wav\r\n\r\n').encode() + self.audio.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        return self.send('/transcribe', 'POST', body, {'Content-Type': f'multipart/form-data; boundary={boundary}'}, 15)

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=6)
            self.process.stdout.close()
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.stop()
        self.log.close()
        if kind:
            print((self.directory / 'service.log').read_text())
            worker_log = self.data / 'server.log'
            if worker_log.exists(): print(worker_log.read_text())
        for path in (self.directory / 'spawned').glob('*'):
            assert not alive(int(path.name)), f'Worker {path.name} survived service shutdown'
        self.temp.cleanup()
