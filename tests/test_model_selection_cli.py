"""Exercise real CLI argument/asset failures without GPU inference or downloads."""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parent.parent
BINARY = Path(os.environ.get("PARAKETTO_TEST_BINARY", REPO / "paraketto.cuda"))


class ModelSelectionCLI(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paraketto-selector-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # A copy isolates /proc/self/exe lookup from this checkout's real weights.
        self.binary = self.root / "paraketto.cuda"
        shutil.copy2(BINARY, self.binary)
        self.cache = self.root / "cache" / "paraketto"
        self.cache.mkdir(parents=True)
        self.download_marker = self.root / "download-attempted"
        fake_curl = self.root / "curl"
        fake_curl.write_text('#!/bin/sh\n: > "$PARAKETTO_DOWNLOAD_MARKER"\nexit 99\n')
        fake_curl.chmod(0o700)
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1",
                        XDG_CACHE_HOME=str(self.cache.parent),
                        PATH=str(self.root) + os.pathsep + os.environ.get("PATH", ""),
                        PARAKETTO_DOWNLOAD_MARKER=str(self.download_marker))

    def invoke(self, *args, success=False):
        result = subprocess.run([str(self.binary), *map(str, args)], env=self.env,
                                cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0 if success else 1, result.stderr)
        self.assertFalse(self.download_marker.exists(), "Unexpected network download")
        return result.stdout + result.stderr

    def header(self, path, version):
        path.write_bytes(b"PRKT" + struct.pack("<I", version))
        return path

    def test_help_and_invalid_arguments(self):
        self.assertIn("v2|v3|ultra", self.invoke("--help", success=True))
        for args, message in [
            (("--model",), "requires"),
            (("--model", "--server"), "requires"),
            (("--model", "bogus", "audio.wav"), "Unknown model"),
            (("--weights",), "requires"),
            (("--weights", "--model", "ultra"), "requires"),
            (("--modle", "ultra", "audio.wav"), "Unknown option"),
        ]:
            with self.subTest(args=args):
                self.assertIn(message, self.invoke(*args))

    def test_explicit_mismatch_in_both_argument_orders(self):
        weights = self.root / "custom weights.bin"
        for preset, expected, actual in [("v2", 2, 3), ("v3", 3, 2), ("ultra", 3, 2)]:
            self.header(weights, actual)
            for flags in [("--model", preset, "--weights", weights),
                          ("--weights", weights, "--model", preset)]:
                with self.subTest(preset=preset, flags=flags):
                    output = self.invoke(*flags, "audio.wav")
                    self.assertIn(f"Model {preset} requires architecture v{expected}", output)
                    self.assertIn(f"weights have architecture v{actual}", output)

    def test_missing_explicit_weights_never_download(self):
        missing = self.root / "missing.bin"
        for model_flags in [(), ("--model", "v2"), ("--model", "v3"), ("--model", "ultra")]:
            with self.subTest(model_flags=model_flags):
                self.assertIn(str(missing), self.invoke(*model_flags, "--weights", missing, "audio.wav"))

    def test_missing_ultra_has_preparation_instructions(self):
        output = self.invoke("--model", "ultra", "audio.wav")
        self.assertIn(str(self.cache / "paraketto-ultra-fp16.bin"), output)
        self.assertIn("scripts/export_ultra_weights.py", output)
        self.assertIn("moondream/parakeet-ultra", output)

    def test_default_is_v2_and_preset_uses_xdg_cache(self):
        self.header(self.cache / "paraketto-fp16.bin", 3)
        self.assertIn("Model v2 requires architecture v2", self.invoke("audio.wav"))
        self.header(self.cache / "paraketto-v3-fp16.bin", 2)
        self.assertIn("Model v3 requires architecture v3", self.invoke("--model", "v3", "audio.wav"))

    def test_local_file_precedes_cache_and_override_precedes_local(self):
        # Invalid cached magic would give a different error if it took precedence.
        (self.cache / "paraketto-ultra-fp16.bin").write_bytes(b"badmagic")
        self.header(self.root / "paraketto-ultra-fp16.bin", 2)
        self.assertIn("Model ultra requires architecture v3", self.invoke("--model", "ultra", "audio.wav"))
        explicit = self.root / "explicit.bin"
        explicit.write_bytes(b"badmagic")
        output = self.invoke("--model", "ultra", "--weights", explicit, "audio.wav")
        self.assertIn("Invalid PRKT model header", output)
        self.assertIn(str(explicit), output)


if __name__ == "__main__":
    unittest.main()
