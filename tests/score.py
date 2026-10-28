"""Word error rate of paraketto-cli transcripts against references, the Open ASR Leaderboard way:
Whisper's EnglishTextNormalizer on both sides, then jiwer.

Usage (from tools/export): uv run python ../../tests/score.py reference.tsv hypotheses.tsv
Both files hold one `file<TAB>text` per line; lines starting with # are ignored.
"""

import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

import jiwer  # noqa: E402
from whisper_normalizer.english import EnglishTextNormalizer  # noqa: E402


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or "\t" not in line:
            continue
        name, text = line.rstrip("\n").split("\t", 1)
        rows[name.strip()] = text.strip()
    return rows


def main():
    references, hypotheses = load(sys.argv[1]), load(sys.argv[2])
    normalize = EnglishTextNormalizer()
    names = sorted(set(references) & set(hypotheses))
    refs, hyps = [], []
    for name in names:
        ref, hyp = normalize(references[name]), normalize(hypotheses[name])
        refs.append(ref)
        hyps.append(hyp)
        result = jiwer.process_words(ref, hyp)
        errors = result.substitutions + result.deletions + result.insertions
        if errors:
            print(f"{name}: {errors} error(s)")
            r, h = result.references[0], result.hypotheses[0]
            for chunk in result.alignments[0]:
                if chunk.type == "equal":
                    continue
                context = " ".join(r[max(0, chunk.ref_start_idx - 3):chunk.ref_start_idx])
                after = " ".join(r[chunk.ref_end_idx:chunk.ref_end_idx + 2])
                said = " ".join(r[chunk.ref_start_idx:chunk.ref_end_idx]) or "-"
                got = " ".join(h[chunk.hyp_start_idx:chunk.hyp_end_idx]) or "-"
                print(f"   {chunk.type:<10} ...{context} [{said} -> {got}] {after}...")
    total = jiwer.process_words(refs, hyps)
    errors = total.substitutions + total.deletions + total.insertions
    words = sum(len(r.split()) for r in refs)
    print(f"WER {100 * total.wer:.2f}% ({errors} errors in {words} words, {len(names)} files)")


if __name__ == "__main__":
    main()
