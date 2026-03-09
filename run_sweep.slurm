#!/bin/bash
#SBATCH --job-name=vae_gan_sweep
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Optional: load modules
# module load cuda/11.8

# Optional: activate your environment
# source activate.sh

DATASET="mnist"
DATAROOT="/path/to/data"
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
