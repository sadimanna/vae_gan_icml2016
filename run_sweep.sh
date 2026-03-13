#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

cd "${PROJECT_ROOT}"

# Optional virtual environment activation
if [[ -n "${VENV_ACTIVATE:-}" && -f "${VENV_ACTIVATE}" ]]; then
  # Allow callers to provide an explicit venv path
  source "${VENV_ACTIVATE}"
elif [[ -f "${PROJECT_ROOT}/bin/activate" ]]; then
  source "${PROJECT_ROOT}/bin/activate"
elif [[ -f "${PROJECT_ROOT}/../bin/activate" ]]; then
  source "${PROJECT_ROOT}/../bin/activate"
else
  echo "Note: No virtual environment found; continuing without activation."
fi

DATASET="cifar10"
DATAROOT="data"
OUTDIR="outputs"

GAMMAS=(15 10 5 1)
KLD_WTS=(0.0001 0.00025 0.0005 0.001 0.0025 0.005 0.01)

for gamma in "${GAMMAS[@]}"; do
  for kld in "${KLD_WTS[@]}"; do
    echo "Running gamma=${gamma}, kld_wt=${kld}"
    python run.py \
      --dataset "${DATASET}" \
      --dataroot "${DATAROOT}" \
      --outf "${OUTDIR}" \
      --gamma "${gamma}" \
      --kld_wt "${kld}" \
      --cuda \
      --imageSize 64 \
      --batchSize 64 \
      --niter 50 \
      --eval_samples 5000
  done
done
