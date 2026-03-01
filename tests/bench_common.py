"""Shared utilities for benchmark scripts."""

import json
from pathlib import Path

from jiwer import Compose, ReduceToListOfListOfWords, RemovePunctuation, ToLowerCase

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DATASETS = ["librispeech", "earnings22", "long", "difficult"]

normalize = Compose([ToLowerCase(), RemovePunctuation(), ReduceToListOfListOfWords()])


def load_manifest(name: str) -> list[dict]:
    manifest_path = DATA_DIR / name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        entry["audio_path"] = str(DATA_DIR / name / entry["audio_path"])
    return manifest


def print_results(rows: list[dict]) -> None:
    """Print benchmark results as a Unicode box table.

    Each row dict has: name, wer, rtfx, utts, audio_s, inference_s.
    """
    grand_audio = sum(r["audio_s"] for r in rows)
    grand_inference = sum(r["inference_s"] for r in rows)
    grand_rtfx = grand_audio / grand_inference if grand_inference > 0 else 0

    # Format inference time
    def fmt_time(s):
        return f"{s * 1000:.0f}ms" if s < 1 else f"{s:.2f}s"

    # Column widths
    W_NAME = max(len(r["name"]) for r in rows)
    W_NAME = max(W_NAME, len("Total"), len("Dataset"))
    W_WER = 8
    W_RTFX = 7
    W_UTTS = 6
    W_AUDIO = 7
    W_TIME = 8

    def sep(left, mid, right, fill="─"):
        return (f"{left}{fill * (W_NAME + 2)}{mid}{fill * (W_WER + 2)}"
                f"{mid}{fill * (W_RTFX + 2)}{mid}{fill * (W_UTTS + 2)}"
                f"{mid}{fill * (W_AUDIO + 2)}{mid}{fill * (W_TIME + 2)}{right}")

    def row(name, wer_s, rtfx_s, utts_s, audio_s, time_s):
        return (f"│ {name:<{W_NAME}} │ {wer_s:>{W_WER}} │ {rtfx_s:>{W_RTFX}} │"
                f" {utts_s:>{W_UTTS}} │ {audio_s:>{W_AUDIO}} │ {time_s:>{W_TIME}} │")

    print(sep("┌", "┬", "┐"))
    print(row("Dataset", "WER", "RTFx", "Utts", "Audio", "Time"))
    print(sep("├", "┼", "┤"))
    for r in rows:
        print(row(
            r["name"],
            f"{r['wer']:.2f}%",
            f"{r['rtfx']:.0f}x",
            str(r["utts"]),
            f"{r['audio_s']:.0f}s",
            fmt_time(r["inference_s"]),
        ))
    print(sep("├", "┼", "┤"))
    print(row("Total", "", f"{grand_rtfx:.0f}x", str(sum(r["utts"] for r in rows)),
              f"{grand_audio:.0f}s", fmt_time(grand_inference)))
    print(sep("└", "┴", "┘"))
