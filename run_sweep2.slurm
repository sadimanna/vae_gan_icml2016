#!/bin/bash

#SBATCH --job-name=vae_gan_sweep
#SBATCH --output=slurm_logs/master_%j.out
#SBATCH --error=slurm_logs/master_%j.err
#SBATCH --partition=short
#SBATCH --nodelist=cn5
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
OUTDIR="outputs_addedeqmargin"
export OUTDIR
export DATASET

GAMMAS=(0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0) # 15 10 5 1)
KLD_WTS=(0.00001 0.0001 0.001 0.01) #0.00005 0.00025 0.0005 0.001 0.005 0.01 0.1 1.0)

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
      --niter 5 \
      --eval_samples 100 \
      --hook_layers 'conv4' \
      --nz 16 \
      --equillibrium 0.69 --margin 0.15
  done
done

SUMMARY_FILE="slurm_logs/summary_${SLURM_JOB_ID}.tsv"
export SUMMARY_FILE

python - <<'PY'
import glob
import json
import os

root = os.environ.get("OUTDIR", "outputs")
dataset = os.environ.get("DATASET", "").strip()
summary_path = os.environ.get("SUMMARY_FILE", "slurm_logs/summary.tsv")
records = []

def _parse_run_ts(run_dir):
    # Expected suffix like ..._DDMMYYYY_HHMMSS
    parts = run_dir.split("_")
    if len(parts) < 3:
        return None
    ts = "_".join(parts[-2:])
    try:
        import datetime as _dt
        return _dt.datetime.strptime(ts, "%d%m%Y_%H%M%S")
    except Exception:
        return None

paths = glob.glob(os.path.join(root, "run_*", "metrics.json"))
if dataset:
    paths = [p for p in paths if f"run_{dataset}_" in os.path.basename(os.path.dirname(p))]

paths = sorted(
    paths,
    key=lambda p: (_parse_run_ts(os.path.basename(os.path.dirname(p))) or 0),
)

for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rec = {"run_dir": os.path.basename(os.path.dirname(path))}
    rec.update(data.get("hyperparams", {}))
    rec.update(data.get("metrics", {}))
    rec["status"] = data.get("status", "")
    records.append(rec)

if not records:
    print("No metrics.json files found in outputs/run_*/")
    raise SystemExit(0)

columns = [
    "run_dir",
    "status",
    "dataset",
    "imageSize",
    "batchSize",
    "nz",
    "gamma",
    "kld_wt",
    "niter",
    "lr_enc",
    "lr_dec",
    "lr_dis",
    "lr_decay_enc",
    "lr_decay_dec",
    "lr_decay_dis",
    "eval_samples",
    "fid",
    "ssim",
    "lpips",
]

def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\t".join(columns) + "\n")
    for rec in records:
        row = [_fmt(rec.get(col, "")) for col in columns]
        f.write("\t".join(row) + "\n")

print(f"Summary written to {summary_path}")
print("\t".join(columns))
for rec in records:
    print("\t".join(_fmt(rec.get(col, "")) for col in columns))
PY
