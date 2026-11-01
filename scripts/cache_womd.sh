#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

# Activate an environment with preprocessing dependencies before running.
# Example: bash scripts/cache_womd.sh --input_dir /path/to/scenario --split training
exec python -m src.data_preprocess --num_workers 12 "$@"
