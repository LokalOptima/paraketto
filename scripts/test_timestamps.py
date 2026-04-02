#!/usr/bin/env python3
"""Test word-level timestamps by extracting audio snippets and re-transcribing them.

For each test WAV file:
1. Transcribe with --timestamps to get word-level timing
2. Extract phrases from the beginning, middle, and end of the transcript
3. Use ffmpeg to cut the audio at those timestamps
4. Re-transcribe the snippet
5. Verify the phrase appears in the snippet transcription
"""

import subprocess
import tempfile
import re
import sys
import random
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "paraketto.cuda"

TEST_FILES = (
    sorted((REPO / "data" / "librispeech").glob("*.wav")) +
    sorted((REPO / "data" / "earnings22").glob("*.wav"))
)

PHRASE_LEN = 3
MIN_WORDS = 15
PAD_MS = 320
MIN_SNIPPET_MS = 1000  # skip snippets shorter than 1s


@dataclass
class Failure:
    file: str
    position: str
    expected: str
    got: str
    range_ms: tuple[int, int]
    detail: str


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def transcribe_with_timestamps(wav_path: Path) -> list[tuple[int, str]]:
    result = subprocess.run(
        [str(BIN), "--timestamps", str(wav_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"transcribe failed: {result.stderr}")
    words = []
    for line in result.stdout.strip().splitlines():
        ms_str, word = line.split("\t", 1)
        words.append((int(ms_str), word))
    return words


def transcribe_plain(wav_path: Path) -> str:
    result = subprocess.run(
        [str(BIN), str(wav_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"transcribe failed: {result.stderr}")
    return result.stdout.strip()


def extract_audio(wav_path: Path, start_ms: int, end_ms: int, out_path: Path):
    # Prepend 100ms silence to avoid encoder subsampling alignment issues
    # (stride-2 x3 = 8x downsampling makes output sensitive to absolute position)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-i", str(wav_path),
            "-filter_complex",
            f"[0]atrim=duration=0.1[pad];[1]atrim=start={start_ms / 1000:.3f}:end={end_ms / 1000:.3f},asetpts=PTS-STARTPTS[clip];[pad][clip]concat=n=2:v=0:a=1",
            "-ar", "16000", "-ac", "1",
            str(out_path),
        ],
        capture_output=True, check=True,
    )


def get_audio_duration_ms(wav_path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(wav_path)],
        capture_output=True, text=True,
    )
    return int(float(result.stdout.strip()) * 1000)


def fuzzy_match(expected: str, got: str) -> bool:
    """Match words allowing prefix/suffix variations (royalists≈royalist, etc.)."""
    if expected == got:
        return True
    # One is a prefix of the other (handles plurals, tense, etc.)
    shorter, longer = sorted([expected, got], key=len)
    return longer.startswith(shorter) and len(longer) - len(shorter) <= 3


def find_word(word: str, words: list[str]) -> int:
    """Find index of word in words list, with fuzzy matching. Returns -1 if not found."""
    for i, w in enumerate(words):
        if fuzzy_match(word, w):
            return i
    return -1


def check_boundaries(expected: str, got: str) -> tuple[bool, str]:
    """Check that the snippet starts with the first expected word and ends with the last.

    Padding adds extra words at both edges, so we allow:
    - First word within first 3 words of snippet
    - Last word within last 5 words (320ms pad ≈ 2-3 trailing words)
    """
    if not got:
        return False, "empty transcription"

    exp_words = expected.split()
    got_words = got.split()
    first, last = exp_words[0], exp_words[-1]

    START_TOLERANCE = 3
    END_TOLERANCE = 5

    first_idx = find_word(first, got_words[:START_TOLERANCE])
    last_idx = find_word(last, got_words[-END_TOLERANCE:])

    if first_idx >= 0 and last_idx >= 0:
        return True, ""
    parts = []
    if first_idx < 0:
        parts.append(f"start: wanted \"{first}\" in {got_words[:START_TOLERANCE]}")
    if last_idx < 0:
        parts.append(f"end: wanted \"{last}\" in {got_words[-END_TOLERANCE:]}")
    return False, "; ".join(parts)


def test_phrase(wav_path: Path, words: list[tuple[int, str]],
               start_idx: int, end_idx: int, position: str,
               audio_duration_ms: int) -> Failure | None:
    selected = [w for _, w in words[start_idx:end_idx]]
    expected = normalize(" ".join(selected))

    start_ms = words[start_idx][0]
    # End time: start of next word, or last word's start + padding
    if end_idx < len(words):
        end_ms = words[end_idx][0]
    else:
        end_ms = words[-1][0]
    start_ms = max(0, start_ms - PAD_MS)
    end_ms = end_ms + PAD_MS

    # Skip snippets too short for the model to produce meaningful output
    if end_ms - start_ms < MIN_SNIPPET_MS:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        snippet_path = Path(f.name)
    try:
        extract_audio(wav_path, start_ms, end_ms, snippet_path)
        got = normalize(transcribe_plain(snippet_path))
        ok, detail = check_boundaries(expected, got)
        if ok:
            return None
        return Failure(wav_path.name, position, expected, got, (start_ms, end_ms), detail)
    finally:
        snippet_path.unlink(missing_ok=True)


def run_tests():
    if not BIN.exists():
        print(f"error: {BIN} not found, run 'make paraketto.cuda' first")
        sys.exit(1)
    if not TEST_FILES:
        print("error: no test WAV files found")
        sys.exit(1)

    random.seed(42)
    failures: list[Failure] = []
    tested = 0
    skipped = 0

    for wav in TEST_FILES:
        words = transcribe_with_timestamps(wav)
        if len(words) < MIN_WORDS:
            skipped += 3
            continue

        audio_dur = get_audio_duration_ms(wav)
        n = len(words)

        # Three positions: beginning, middle, end
        positions = [
            (0, PHRASE_LEN, "begin"),
        ]
        margin = max(PHRASE_LEN, n // 4)
        mid = random.randint(margin, n - PHRASE_LEN - margin)
        positions.append((mid, mid + PHRASE_LEN, "middle"))
        positions.append((n - PHRASE_LEN, n, "end"))

        file_pass = 0
        file_fail = 0
        for si, ei, pos in positions:
            tested += 1
            fail = test_phrase(wav, words, si, ei, pos, audio_dur)
            if fail:
                failures.append(fail)
                file_fail += 1
            else:
                file_pass += 1

        status = "PASS" if file_fail == 0 else f"{file_fail}/3 FAIL"
        print(f"  {status}  {wav.name} ({n} words)")

    passed = tested - len(failures)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{tested} passed  ({skipped} skipped, {len(TEST_FILES)} files)")

    if not failures:
        print(f"  All tests passed!")
        return

    # Group failures by type
    empty = [f for f in failures if not f.got]
    bad_start = [f for f in failures if f.got and "start:" in f.detail and "end:" not in f.detail]
    bad_end = [f for f in failures if f.got and "end:" in f.detail and "start:" not in f.detail]
    bad_both = [f for f in failures if f.got and "start:" in f.detail and "end:" in f.detail]

    for label, group in [
        ("Empty transcription", empty),
        ("Wrong start boundary", bad_start),
        ("Wrong end boundary", bad_end),
        ("Wrong both boundaries", bad_both),
    ]:
        if not group:
            continue
        print(f"\n  {label} ({len(group)}):")
        for f in group:
            print(f"    {f.file} [{f.position}]  expected: \"{f.expected}\"  got: \"{f.got}\"")
            if f.detail:
                print(f"      {f.detail}")

    print()
    sys.exit(1)


if __name__ == "__main__":
    run_tests()
