"""Install the local transcription service for the current user's login session."""
import argparse
import os
from pathlib import Path
import subprocess

def unit_quote(value):
    # systemd unit specifiers and ExecStart escaping, not shell quoting.
    value = str(value).replace('%', '%%').replace('$', '$$')
    value = value.replace('\\', '\\\\').replace('"', '\\"')
    if '\n' in value or '\r' in value:
        raise SystemExit('The executable path must not contain line breaks')
    return '"' + value + '"'

MARKER = '# Managed by Paraketto scripts/install_service.py\n'


def service_unit(executable, model=None, weights=None):
    arguments = [str(executable)]
    if model is not None:
        if model not in ('v2', 'v3', 'ultra'):
            raise ValueError('Unknown model preset')
        arguments += ['--model', model]
    if weights is not None:
        arguments += ['--weights', str(Path(weights).expanduser().resolve())]
    return MARKER + f'''[Unit]
Description=Paraketto local transcription service

[Service]
Type=simple
ExecStart={' '.join(unit_quote(argument) for argument in arguments)}
Restart=on-failure
RestartSec=2
TimeoutStopSec=5
KillMode=mixed

[Install]
WantedBy=default.target
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('v2', 'v3', 'ultra'), help='Service preset; defaults to Ultra unless custom weights are supplied')
    parser.add_argument('--weights', type=Path, help='Existing custom weights; architecture auto-detected when --model is omitted')
    args = parser.parse_args()
    executable = Path(__file__).resolve().parent.parent / 'paraketto-service'
    if not os.access(executable, os.X_OK):
        raise SystemExit('Build paraketto-service first: make desktop')
    directory = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'systemd/user'
    unit = directory / 'paraketto.service'
    if unit.exists() and not unit.read_text().startswith(MARKER):
        raise SystemExit(f'An existing custom service is present at {unit}; leaving it unchanged.')
    contents = service_unit(executable, args.model, args.weights)
    directory.mkdir(parents=True, exist_ok=True)
    unit.write_text(contents)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', 'paraketto.service'], check=True)
    subprocess.run(['systemctl', '--user', 'restart', 'paraketto.service'], check=True)
    print(f'Installed {unit}\nThe desktop discovers the local API automatically; the model loads on demand.')


if __name__ == '__main__':
    main()
