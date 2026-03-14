"""Benchmark regression testing with statistical confidence.

Each measurement session starts the server, does a warmup pass, runs the full
benchmark suite R times, then stops the server. We repeat for S sessions to
capture between-session variance. The t-test operates on session means, which
is the correct unit of replication.

Usage:
    # Save baseline:
    uv run python tests/bench_regression.py paraketto.fp8 --save baseline_fp8.json

    # Compare (default: --expect-better, one-sided, flags regressions only):
    uv run python tests/bench_regression.py paraketto.fp8 --compare baseline_fp8.json

    # Expect no change (two-sided, flags any difference):
    uv run python tests/bench_regression.py paraketto.fp8 --compare baseline.json --expect-same

    # Tune sessions/runs (default: 5 sessions x 3 runs each):
    uv run python tests/bench_regression.py paraketto.fp8 --save b.json --sessions 7 --runs 5
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
from jiwer import wer as compute_wer
from scipy.stats import t as t_dist, ttest_ind
from whisper_normalizer.english import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASETS = ["librispeech", "earnings22", "long", "difficult"]

whisper_normalize = EnglishTextNormalizer()


def load_manifest(name: str) -> list[dict]:
    manifest_path = DATA_DIR / name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        entry["audio_path"] = str(DATA_DIR / name / entry["audio_path"])
    return manifest


def start_server(binary: Path, port: int = 18080) -> subprocess.Popen:
    server = subprocess.Popen(
        [str(binary), "--server", f":{port}"],
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ret = server.poll()
        if ret is not None:
            print(f"Server exited with code {ret}", file=sys.stderr)
            sys.exit(1)
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=1)
            if r.ok:
                return server
        except requests.ConnectionError:
            pass
        time.sleep(0.1)
    print("Server failed to start (timeout)", file=sys.stderr)
    sys.exit(1)


def stop_server(server: subprocess.Popen):
    server.send_signal(signal.SIGINT)
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def transcribe(path: str, port: int = 18080) -> dict:
    with open(path, "rb") as f:
        r = requests.post(f"http://localhost:{port}/transcribe", files={"file": f})
    r.raise_for_status()
    return r.json()


def run_once(port: int = 18080, label: str = "") -> dict:
    """Run full benchmark suite once. Returns {dataset: {wer, rtfx, ...}}."""
    results = {}
    for name in DATASETS:
        manifest = load_manifest(name)
        ds_audio = sum(e["duration_s"] for e in manifest)
        ds_inference = 0.0
        references = []
        hypotheses = []

        for i, entry in enumerate(manifest):
            result = transcribe(entry["audio_path"], port)
            hypotheses.append(result["text"])
            references.append(entry["reference"])
            ds_inference += result["inference_time_s"]
            print(f"\r    {label}{name}: {i+1}/{len(manifest)}",
                  end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)

        wer_pct = compute_wer(
            [whisper_normalize(r) for r in references],
            [whisper_normalize(h) for h in hypotheses],
        ) * 100
        rtfx = ds_audio / ds_inference if ds_inference > 0 else 0

        results[name] = dict(wer=wer_pct, rtfx=rtfx,
                             inference_s=ds_inference, audio_s=ds_audio)
    return results


def run_session(binary: Path, port: int, runs: int,
                session_idx: int, total_sessions: int) -> dict:
    """One session: start server, warmup, R runs, stop server. Returns session means."""
    tag = f"[session {session_idx+1}/{total_sessions}]"
    print(f"\n{tag} Starting server...", file=sys.stderr)
    server = start_server(binary, port)

    try:
        print(f"{tag} Warmup (discarded)...", file=sys.stderr)
        run_once(port, label=f"{tag} [warmup] ")

        accum = {ds: {"wer": [], "rtfx": []} for ds in DATASETS}

        for r in range(runs):
            t0 = time.monotonic()
            results = run_once(port, label=f"{tag} [run {r+1}/{runs}] ")
            elapsed = time.monotonic() - t0

            for ds in DATASETS:
                accum[ds]["wer"].append(results[ds]["wer"])
                accum[ds]["rtfx"].append(results[ds]["rtfx"])

            rtfx_summary = " | ".join(
                f"{ds[:4]}={results[ds]['rtfx']:.0f}x" for ds in DATASETS
            )
            print(f"  {tag} Run {r+1}/{runs} ({elapsed:.1f}s): {rtfx_summary}",
                  file=sys.stderr)

    finally:
        stop_server(server)

    # Return session means
    session = {}
    for ds in DATASETS:
        session[ds] = {
            "wer": float(np.mean(accum[ds]["wer"])),
            "rtfx": float(np.mean(accum[ds]["rtfx"])),
        }

    return session


def mean_std(values: list[float]) -> tuple[float, float]:
    a = np.array(values)
    return float(a.mean()), float(a.std(ddof=1))


def print_comparison(baseline: dict, current: dict, alpha: float = 0.05,
                     mode: str = "expect-same") -> bool:
    """Print comparison table and return True if expectation is met.

    baseline/current: {dataset: {"wer_sessions": [...], "rtfx_sessions": [...]}}
    Each list contains one value per session (the session mean).
    """
    all_pass = True

    if mode == "expect-same":
        wer_alt = "two-sided"
        rtfx_alt = "two-sided"
        wer_label = "WER (two-sided — any change is a failure)"
        rtfx_label = "RTFx (two-sided — any change is a failure)"
    else:  # expect-better
        wer_alt = "greater"
        rtfx_alt = "less"
        wer_label = "WER (one-sided — only increases flagged)"
        rtfx_label = "RTFx (one-sided — only decreases flagged)"

    print()
    print("=" * 90)
    print(f"  REGRESSION TEST — mode: {mode}")
    print("=" * 90)
    print()

    n_base = len(list(baseline.values())[0]["rtfx_sessions"])
    n_curr = len(list(current.values())[0]["rtfx_sessions"])
    print(f"  Baseline: {n_base} sessions | Current: {n_curr} sessions | alpha={alpha}")
    print()

    # WER
    print(f"  {wer_label}")
    print("  " + "-" * 78)
    for ds in DATASETS:
        b = baseline[ds]["wer_sessions"]
        c = current[ds]["wer_sessions"]
        b_mean, b_std = mean_std(b)
        c_mean, c_std = mean_std(c)
        delta = c_mean - b_mean

        t_stat, p_val = ttest_ind(c, b, equal_var=False, alternative=wer_alt)
        if np.isnan(p_val):
            p_val = 1.0 if delta == 0 else 0.0

        status = "FAIL" if p_val < alpha else "PASS"
        if p_val < alpha:
            all_pass = False

        print(f"  {status:>4}  {ds:<15} "
              f"base={b_mean:6.2f}% (+-{b_std:4.2f})  "
              f"curr={c_mean:6.2f}% (+-{c_std:4.2f})  "
              f"delta={delta:+.2f}pp  p={p_val:.3f}")

    print()

    # RTFx
    print(f"  {rtfx_label}")
    print("  " + "-" * 78)
    for ds in DATASETS:
        b = baseline[ds]["rtfx_sessions"]
        c = current[ds]["rtfx_sessions"]
        b_mean, b_std = mean_std(b)
        c_mean, c_std = mean_std(c)
        pct_change = (c_mean - b_mean) / b_mean * 100

        t_stat, p_val = ttest_ind(c, b, equal_var=False, alternative=rtfx_alt)

        status = "FAIL" if p_val < alpha else "PASS"
        if p_val < alpha:
            all_pass = False

        print(f"  {status:>4}  {ds:<15} "
              f"base={b_mean:7.0f}x (+-{b_std:5.1f})  "
              f"curr={c_mean:7.0f}x (+-{c_std:5.1f})  "
              f"delta={pct_change:+.1f}%  p={p_val:.3f}")

    print()
    if all_pass:
        print("  RESULT: PASS")
    else:
        print("  RESULT: FAIL")
    print("=" * 90)
    print()

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark regression testing")
    parser.add_argument("binary", help="Binary to benchmark (e.g. paraketto.fp8)")
    parser.add_argument("--save", metavar="FILE",
                        help="Run benchmarks and save baseline to FILE")
    parser.add_argument("--compare", metavar="FILE",
                        help="Run benchmarks and compare against baseline FILE")
    parser.add_argument("--sessions", type=int, default=5,
                        help="Number of sessions (server restarts) (default: 5)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per session (default: 3)")
    parser.add_argument("--port", type=int, default=18080,
                        help="Server port (default: 18080)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for t-test (default: 0.05)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--expect-same", action="store_const", dest="mode",
                            const="expect-same",
                            help="Two-sided test: flag any change (for refactoring)")
    mode_group.add_argument("--expect-better", action="store_const", dest="mode",
                            const="expect-better",
                            help="One-sided test: flag regressions only (for optimizations)")
    parser.set_defaults(mode="expect-better")
    args = parser.parse_args()

    if not args.save and not args.compare:
        parser.error("Specify --save FILE or --compare FILE")

    binary = ROOT / args.binary
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        print(f"Run 'make {args.binary}' first.", file=sys.stderr)
        sys.exit(1)

    if not (DATA_DIR / "librispeech" / "manifest.json").exists():
        print("Benchmark data not found. Run 'make download-data' first.",
              file=sys.stderr)
        sys.exit(1)

    # Collect S sessions, each with R runs
    all_sessions = {ds: {"wer_sessions": [], "rtfx_sessions": []} for ds in DATASETS}

    for s in range(args.sessions):
        session = run_session(binary, args.port, args.runs, s, args.sessions)

        for ds in DATASETS:
            all_sessions[ds]["wer_sessions"].append(session[ds]["wer"])
            all_sessions[ds]["rtfx_sessions"].append(session[ds]["rtfx"])

    # Print summary
    print(file=sys.stderr)
    print(f"  {'Dataset':<15} {'WER mean':>10} {'WER std':>10} "
          f"{'RTFx mean':>10} {'RTFx std':>10}", file=sys.stderr)
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", file=sys.stderr)
    for ds in DATASETS:
        wm, ws = mean_std(all_sessions[ds]["wer_sessions"])
        rm, rs = mean_std(all_sessions[ds]["rtfx_sessions"])
        print(f"  {ds:<15} {wm:>9.2f}% {ws:>9.2f} "
              f"{rm:>9.0f}x {rs:>9.1f}", file=sys.stderr)
    print(file=sys.stderr)

    if args.save:
        Path(args.save).write_text(json.dumps(all_sessions, indent=2))
        print(f"Baseline saved to {args.save} ({args.sessions} sessions x "
              f"{args.runs} runs)", file=sys.stderr)

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        passed = print_comparison(baseline, all_sessions, args.alpha, args.mode)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
