#!/bin/bash
#SBATCH --job-name=vae_gan_sweep
#SBATCH --output=slurm_logs/master_%j.out
#SBATCH --error=slurm_logs/master_%j.err
#SBATCH --partition=short
#SBATCH --nodelist=cn5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Load Python module
# module load cuda/12.4
mkdir -p slurm_logs
# -----------------------------------------------------------------------------
# Activate virtual environment (SLURM-safe)
# -----------------------------------------------------------------------------

echo "Activating virtual environment..."

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set"
    exit 1
fi

echo "SLURM_SUBMIT_DIR = ${SLURM_SUBMIT_DIR}"

# breaching/ is the submit directory
BASE_DIR="${SLURM_SUBMIT_DIR}"

# project root is one level above
PROJECT_ROOT="$(cd "${BASE_DIR}/.." && pwd)"

echo "BASE_DIR     = ${BASE_DIR}"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

if [[ ! -f bin/activate ]]; then
    echo "ERROR: Virtual environment not found at ${PROJECT_ROOT}/bin/activate"
    exit 1
fi

source bin/activate

# Sanity checks
which python
python --version

# Return to breaching
cd "${BASE_DIR}"

echo "Virtual environment activated successfully."

DATASET="mnist"
DATAROOT="data"
OUTDIR="outputs"

GAMMAS=(5 10 15)
KLD_WTS=(0.0001 0.00025 0.0005)

for gamma in "${GAMMAS[@]}"; do
  for kld in "${KLD_WTS[@]}"; do
    echo "Running gamma=${gamma}, kld_wt=${kld}"
    python run.py \
      --dataset "${DATASET}" \
      --dataroot "${DATAROOT}" \
      --outf "${OUTDIR}" \
      --gamma "${gamma}" \
      --kld_wt "${kld}" \
      --cuda
  done
done
