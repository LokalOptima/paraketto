"""Accuracy test for Parakeet TDT TensorRT transcription.

Verifies that the TensorRT-accelerated model transcribes data/sample.wav
to match the expected text in data/sample.txt.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_WAV = ROOT / "data" / "sample.wav"
SAMPLE_TXT = ROOT / "data" / "sample.txt"


def normalize(s: str) -> str:
    """Normalize for fuzzy comparison: lowercase, collapse whitespace, strip quotes."""
    import re
    s = s.lower().strip()
    s = re.sub(r'["""\']', "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def test_transcribe_sample():
    import parakeet_trt

    parakeet_trt.load_model()

    result = parakeet_trt.transcribe_file(SAMPLE_WAV)
    expected = SAMPLE_TXT.read_text().strip()

    # Exact match (ideal)
    if result == expected:
        return

    # Normalized match (acceptable — minor punctuation/case differences)
    assert normalize(result) == normalize(expected), (
        f"Transcription mismatch:\n"
        f"  expected: {expected!r}\n"
        f"  got:      {result!r}"
    )
