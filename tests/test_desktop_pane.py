"""Seed bounded text fixtures, then exercise real GTK against a private CPU service."""
import subprocess
from service_test_support import Fixture, REPO
with Fixture(PARAKETTO_FIXTURE_TEXT='A live paraketo recording.') as fixture:
    binary = str(REPO / 'scratch/test_desktop_pane')
    subprocess.run([binary, '--seed', str(fixture.data)], check=True, timeout=20)
    fixture.start(vad_model=fixture.directory / 'missing-vad')
    (fixture.directory / 'release-transcription').write_text('continue')
    subprocess.run([binary, fixture.url, str(fixture.data)], env=fixture.env, check=True, timeout=90)
