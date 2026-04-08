#!/bin/bash
# test_metal_correctness.sh — verify transcription correctness after kernel changes
#
# Usage:
#   ./tests/test_metal_correctness.sh              # run tests (compare against golden)
#   ./tests/test_metal_correctness.sh --update      # regenerate golden references
#
# Uses 16 LibriSpeech WAVs from data/librispeech/ (diverse speakers, 2-20s durations).

set -eo pipefail

BINARY=./paraketto.metal
GOLDEN_DIR=tests/golden
DATA_DIR=data/librispeech

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: $BINARY not found. Run 'make paraketto.metal' first."
    exit 1
fi

# Get list of WAV files
WAVS=$(ls "$DATA_DIR"/*.wav 2>/dev/null)
if [[ -z "$WAVS" ]]; then
    echo "ERROR: no WAV files in $DATA_DIR/. Download from gondola first."
    exit 1
fi

N_WAVS=$(echo "$WAVS" | wc -l | tr -d ' ')

if [[ "${1:-}" == "--update" ]]; then
    mkdir -p "$GOLDEN_DIR"
    echo "Generating golden references for $N_WAVS files..."
    for wav in $WAVS; do
        name=$(basename "$wav" .wav)
        $BINARY "$wav" 2>/dev/null > "$GOLDEN_DIR/$name.txt"
        echo "  SAVED $name: $(cat "$GOLDEN_DIR/$name.txt" | head -c 80)..."
    done
    echo "Golden references updated in $GOLDEN_DIR/"
    exit 0
fi

# Run tests
if [[ ! -d "$GOLDEN_DIR" ]]; then
    echo "ERROR: no golden references. Run with --update first."
    exit 1
fi

pass=0
fail=0
skip=0

for wav in $WAVS; do
    name=$(basename "$wav" .wav)
    golden="$GOLDEN_DIR/$name.txt"

    if [[ ! -f "$golden" ]]; then
        echo "SKIP $name (no golden reference)"
        skip=$((skip + 1))
        continue
    fi

    actual=$($BINARY "$wav" 2>/dev/null)
    expected=$(cat "$golden")

    if [[ "$actual" == "$expected" ]]; then
        echo "PASS $name"
        pass=$((pass + 1))
    else
        echo "FAIL $name"
        echo "  expected: $expected"
        echo "  actual:   $actual"
        fail=$((fail + 1))
    fi
done

echo ""
echo "Results: $pass passed, $fail failed, $skip skipped (of $N_WAVS files)"

if [[ $fail -gt 0 ]]; then
    exit 1
fi
