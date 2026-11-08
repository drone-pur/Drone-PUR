#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'Usage: %s BASE_MODEL PREPARED_STAGE_ROOT RUN_ROOT\n' "$0" >&2
  exit 2
fi

MODEL="$1"
DATA="$2"
RUN="$3"
NPROC="${NPROC_PER_NODE:-8}"
ACC="${GRAD_ACC:-4}"
PIXELS="${MAX_PIXELS:-1003520}"
LENGTH="${MAX_LENGTH:-8192}"

# Run in the same environment used to install this package.
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m drone_pur.training.train_stage --stage s1 --model "$MODEL" --data-dir "$DATA/s1" --output-dir "$RUN/s1" --grad-acc "$ACC" --max-pixels "$PIXELS" --max-length "$LENGTH"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m drone_pur.training.train_stage --stage s2 --model "$MODEL" --data-dir "$DATA/s2" --output-dir "$RUN/s2" --init-checkpoint "$RUN/s1/final" --grad-acc "$ACC" --max-pixels "$PIXELS" --max-length "$LENGTH"
# S3 completes its configured single epoch and saves s3/final.
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m drone_pur.training.train_stage --stage s3 --model "$MODEL" --data-dir "$DATA/s3" --output-dir "$RUN/s3" --init-checkpoint "$RUN/s2/final" --grad-acc "$ACC" --max-pixels "$PIXELS" --max-length "$LENGTH"
