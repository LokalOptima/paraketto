#!/usr/bin/env python3
"""Small HTTP fixture for the desktop lifecycle tests; never loads a GPU model."""
import json
import os
import signal
from pathlib import Path
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

directory = Path(os.environ['PARAKETTO_FIXTURE_DIR'])
(directory / 'spawned').mkdir(exist_ok=True)
(directory / 'spawned' / str(os.getpid())).write_text('worker')
if os.environ.get('PARAKETTO_FIXTURE_IGNORE_TERM'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

class Handler(BaseHTTPRequestHandler):
    def reply(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/info':
            (directory / 'client-ready').write_text(str(os.getpid()))
        self.reply({'status': 'ready', 'pid': os.getpid()} if self.path == '/status' else
                   {'model': 'Fixture', 'backend': 'Fixture', 'device': 'No GPU'})

    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        (directory / 'request-started').write_text('yes')
        deadline = time.monotonic() + 30
        while not (directory / 'release-transcription').exists() and time.monotonic() < deadline:
            time.sleep(0.025)
        self.reply({'text': 'Lifecycle fixture transcript', 'audio_duration_s': 1, 'inference_time_s': 2})

time.sleep(float(os.environ.get('PARAKETTO_FIXTURE_DELAY', '0.5')))
if (directory / 'fail-load').exists():
    raise SystemExit('Fixture model load failed')
server = HTTPServer(('127.0.0.1', 0), Handler)
status = {'event': 'ready', 'pid': os.getpid(), 'port': server.server_port}
(directory / 'server.json').write_text(json.dumps(status))
print(json.dumps(status), flush=True)
server.serve_forever()
