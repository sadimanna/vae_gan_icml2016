#!/bin/bash

#SBATCH --job-name=vae_gan_sweep
#SBATCH --output=slurm_logs/master_%j.out
#SBATCH --error=slurm_logs/master_%j.err
#SBATCH --partition=long
#SBATCH --nodelist=cn4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

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

# Metrics dependencies
# python -m pip install -q torchmetrics lpips

# Return to breaching
cd "${BASE_DIR}"

echo "Virtual environment activated successfully."

DATASET="mnist"
DATAROOT="data"
OUTDIR="outputs"

GAMMAS=(0.000001 0.00001 0.0001 0.001 0.01 0.1) # 15 10 5 1)
KLD_WTS=(0.00001 0.00005 0.0001 0.00025 0.0005 0.001 0.0025 0.005 0.01)

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
      --lr 0.0001 \
      --imageSize 64 \
      --batchSize 64 \
      --niter 10 \
      --eval_samples 1000 \
      --hook_layers 'conv4'
  done
done
