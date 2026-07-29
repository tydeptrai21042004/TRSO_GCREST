#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${DEPTH_DATA_PATH:?Set DEPTH_DATA_PATH to an open paired depth dataset}"
python -m tools.run_fair_suite \
  --dataset depth_folder --task depth_estimation --data_path "$DEPTH_DATA_PATH" \
  --backbones auto --methods linear,norm,bias,last_block,trso,full \
  --seeds 0,1,2 --epochs 50 --batch_size 16 --execute "$@"
