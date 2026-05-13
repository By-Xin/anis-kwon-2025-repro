#!/usr/bin/env bash
#SBATCH --job-name=ak25-e2e
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs
# Example: split by method and cardinality. Adjust for your scheduler.
METHODS=(e2e_m e2e_socp e2e_sdp)
KS=(10 15 20)
MID=$((SLURM_ARRAY_TASK_ID / 3))
KID=$((SLURM_ARRAY_TASK_ID % 3))
METHOD=${METHODS[$MID]}
K=${KS[$KID]}

python scripts/run_reproduction.py \
  --config configs/reproduce_2015_2020.yaml \
  --methods "$METHOD" \
  --cardinalities "$K" \
  --verbose
