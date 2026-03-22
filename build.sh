#!/usr/bin/env bash
# Local builds only: uses CUDA_HOME for this make invocation (default /usr/local/cuda).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
exec make -C "$ROOT" "$@"
