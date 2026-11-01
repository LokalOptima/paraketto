"""CPU-only preset/worker propagation and installer checks in private directories."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

from service_test_support import Fixture, REPO, wait


def weights_header(version):
    return b'PRKT' + struct.pack('<I', version)


def metadata(fixture, preset, architecture):
    status = fixture.status()
    assert status['model_preset'] == preset, status
    assert status['model_architecture'] == architecture, status
    assert status['models']['asr']['model_preset'] == preset
    assert status['models']['asr']['model_architecture'] == architecture
    code, info = fixture.request('/info')
    assert code == 200 and info['model_preset'] == preset and info['model_architecture'] == architecture, info


for preset, version in [('v2', 2), ('v3', 3), ('ultra', 3)]:
    with Fixture(PARAKETTO_FIXTURE_DELAY='0.01') as fixture:
        fixture.weights.write_bytes(weights_header(version))
        fixture.start(model=preset)
        metadata(fixture, preset, 'v' + str(version))
        assert not (fixture.directory / 'spawned').exists(), 'Selecting a preset eagerly loaded the worker'
        fixture.request('/model/load', 'POST', {})
        wait(lambda: fixture.status()['status'] == 'ready')
        metadata(fixture, preset, 'v' + str(version))
        arguments = json.loads((fixture.directory / 'worker-arguments.json').read_text())
        assert arguments[arguments.index('--model') + 1] == preset
        assert arguments[arguments.index('--weights') + 1] == str(fixture.weights)
print('PASS: each explicit preset preserves custom weight override and reaches the worker without eager loading')

with Fixture(PARAKETTO_FIXTURE_DELAY='0.01') as fixture:
    fixture.weights = fixture.directory / 'ultra-in-the-filename-is-not-an-identity.bin'
    fixture.weights.write_bytes(weights_header(3))
    fixture.start()
    metadata(fixture, 'custom', 'v3')
    fixture.request('/model/load', 'POST', {})
    wait(lambda: fixture.status()['status'] == 'ready')
    metadata(fixture, 'custom', 'v3')
    arguments = json.loads((fixture.directory / 'worker-arguments.json').read_text())
    assert '--model' not in arguments, 'Custom weights alone must let the native worker infer architecture'
    assert fixture.status()['model'] == 'Fixture · Fixture', 'A path substring was misreported as Ultra'
    fixture.request('/model/unload', 'POST', {})
    wait(lambda: fixture.status()['status'] == 'sleeping')
    fixture.weights.write_bytes(weights_header(2))
    (fixture.directory / 'fail-load').write_text('fail')
    fixture.request('/model/load', 'POST', {})
    wait(lambda: fixture.status()['status'] == 'error')
    metadata(fixture, 'custom', 'v2')
    code, info = fixture.request('/info')
    assert code == 200 and info['model'] == 'Parakeet V2 (custom weights)', info
    assert fixture.status()['model'] == info['model'], 'Failed reload retained the previous model label'
    assert 'version' not in info and info['backend'] == 'CUDA · model unloaded', info
    assert info['model_revision']['weights_sha256'] == hashlib.sha256(weights_header(2)).hexdigest()
print('PASS: custom V3 weights retain custom checkpoint identity and architecture auto-detection')

with Fixture(PARAKETTO_FIXTURE_DELAY='0.01') as fixture:
    fixture.start(missing=True)
    metadata(fixture, 'custom', 'unknown')
    assert fixture.transcribe()[0] == 503
    (fixture.directory / 'missing-weights').write_bytes(weights_header(2))
    (fixture.directory / 'release-transcription').write_text('continue')
    assert fixture.transcribe()[0] == 200
    metadata(fixture, 'custom', 'v2')
    arguments = json.loads((fixture.directory / 'worker-arguments.json').read_text())
    assert '--model' not in arguments, 'Service default Ultra must not override custom V2 weights'
print('PASS: repairing custom weights updates unknown architecture to V2 without imposing a preset')

with Fixture(PARAKETTO_FIXTURE_DELAY='0.01') as fixture:
    # Copy the executable to isolate its beside-executable lookup from real weights.
    executable = fixture.directory / 'paraketto-service'
    shutil.copy2(REPO / 'paraketto-service', executable)
    fixture.env['XDG_CACHE_HOME'] = str(fixture.directory / 'cache')
    fixture.start(default_weights=True, service_binary=executable)
    metadata(fixture, 'ultra', 'unknown')
    assert fixture.transcribe()[0] == 503
    assert fixture.status()['load_failed']
    assert 'scripts/export_ultra_weights.py' in (fixture.data / 'server.log').read_text()
    expected = fixture.directory / 'cache/paraketto/paraketto-ultra-fp16.bin'
    assert fixture.status()['weights'] == str(expected)
    expected.parent.mkdir(parents=True)
    expected.write_bytes(weights_header(3))
    (fixture.directory / 'release-transcription').write_text('continue')
    assert fixture.transcribe()[0] == 200, 'Repairing missing default weights must permit retry'
    metadata(fixture, 'ultra', 'v3')
    arguments = json.loads((fixture.directory / 'worker-arguments.json').read_text())
    assert arguments[arguments.index('--model') + 1] == 'ultra'
print('PASS: service default remains Ultra; missing weights fail lazily and recover on retry')

with Fixture() as fixture:
    result = subprocess.run([str(REPO / 'paraketto-service'), '--model', 'not-a-preset',
                             '--data-dir', str(fixture.data), '--port', '0'], capture_output=True, text=True, timeout=5)
    assert result.returncode != 0 and 'model' in result.stderr.lower(), result
    assert not (fixture.data / 'service.json').exists()
print('PASS: unknown service presets fail before publishing an endpoint')

spec = importlib.util.spec_from_file_location('paraketto_installer', REPO / 'scripts/install_service.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
with tempfile.TemporaryDirectory(prefix='paraketto-installer-test-') as temporary:
    weights = Path(temporary) / 'custom "model" $literal%file.bin'
    with patch.dict(os.environ, {'XDG_CONFIG_HOME': temporary}), patch.object(sys, 'argv',
            ['install_service.py', '--model', 'v3', '--weights', str(weights)]), patch.object(installer.subprocess, 'run') as run:
        installer.main()
        unit = (Path(temporary) / 'systemd/user/paraketto.service').read_text()
        assert '"--model" "v3"' in unit
        assert '"--weights" ' + installer.unit_quote(weights) in unit
        assert [call.args[0] for call in run.call_args_list] == [
            ['systemctl', '--user', 'daemon-reload'], ['systemctl', '--user', 'enable', 'paraketto.service'],
            ['systemctl', '--user', 'restart', 'paraketto.service']]
    custom = installer.service_unit('/tmp/paraketto-service', weights=weights)
    assert '--model' not in custom and '--weights' in custom
    assert '--model' not in installer.service_unit('/tmp/paraketto-service')
print('PASS: installer persists selected preset/weights with literal systemd quoting; no real systemd calls')
