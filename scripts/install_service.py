"""Install the local transcription service for the current user's login session."""
import os
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parent.parent
executable = repo / 'paraketto-service'
if not os.access(executable, os.X_OK):
    raise SystemExit('Build paraketto-service first: make desktop')

def unit_quote(value):
    # systemd unit specifiers and ExecStart escaping, not shell quoting.
    value = str(value).replace('%', '%%').replace('$', '$$')
    value = value.replace('\\', '\\\\').replace('"', '\\"')
    if '\n' in value or '\r' in value:
        raise SystemExit('The executable path must not contain line breaks')
    return '"' + value + '"'

directory = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'systemd/user'
unit = directory / 'paraketto.service'
marker = '# Managed by Paraketto scripts/install_service.py\n'
if unit.exists() and not unit.read_text().startswith(marker):
    raise SystemExit(f'An existing custom service is present at {unit}; leaving it unchanged.')
directory.mkdir(parents=True, exist_ok=True)
unit.write_text(marker + f'''[Unit]
Description=Paraketto local transcription service

[Service]
Type=simple
ExecStart={unit_quote(executable)}
Restart=on-failure
RestartSec=2
TimeoutStopSec=5
KillMode=mixed

[Install]
WantedBy=default.target
''')
subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
subprocess.run(['systemctl', '--user', 'enable', 'paraketto.service'], check=True)
subprocess.run(['systemctl', '--user', 'restart', 'paraketto.service'], check=True)
print(f'Installed {unit}\nAPI: http://127.0.0.1:8080 (model loads on demand)')
