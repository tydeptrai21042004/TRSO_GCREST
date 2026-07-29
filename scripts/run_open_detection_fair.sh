#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
python -m tools.run_fair_suite \
  --dataset voc_detection --task object_detection \
  --download True --backbones auto \
  --methods linear,norm,bias,last_block,full \
  --seeds 0,1,2 --epochs 24 --batch_size 4 --execute "$@"
