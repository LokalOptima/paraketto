"""Calibration must cover every audio chunk and reject invalid measurements."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from calibrate_fp8 import N_SITES, aggregate_scales, audio_sha256, parse_scales


class CalibrationTests(unittest.TestCase):
    def test_warmup_is_excluded_and_all_audio_chunks_contribute(self):
        chunks = np.ones((3, N_SITES))
        chunks[0] *= 1000  # silence warmup must not dominate
        chunks[1, 5] = 7
        chunks[2, 8] = 11
        stderr = "\n".join("SCALES: " + " ".join(map(str, chunk)) for chunk in chunks)
        scales = aggregate_scales([parse_scales(stderr), np.full(N_SITES, 2)], 1.5)
        self.assertEqual(scales[5], 10.5)
        self.assertEqual(scales[8], 16.5)
        self.assertEqual(scales[0], 3)

    def test_invalid_observations_fail(self):
        for bad in (np.nan, np.inf, -1):
            values = np.ones((2, N_SITES))
            values[0, 4] = bad
            with self.assertRaises(ValueError):
                aggregate_scales(values, 1.5)
        with self.assertRaises(ValueError):
            aggregate_scales(np.zeros((2, N_SITES)), 1.5)
        with self.assertRaises(ValueError):
            parse_scales("SCALES: " + "1 " * N_SITES)  # only warmup

    def test_audio_identity_ignores_container_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = [Path(tmp) / name for name in ("a.wav", "b.wav", "c.wav")]
            samples = np.array([0, 0.25, -0.5, 0.125], dtype=np.float32)
            sf.write(a, samples, 16000, subtype="PCM_16")
            sf.write(b, samples, 16000, subtype="FLOAT")
            sf.write(c, samples, 24000, subtype="FLOAT")
            self.assertEqual(audio_sha256(a), audio_sha256(b))
            self.assertNotEqual(audio_sha256(a), audio_sha256(c))


if __name__ == "__main__":
    unittest.main()
