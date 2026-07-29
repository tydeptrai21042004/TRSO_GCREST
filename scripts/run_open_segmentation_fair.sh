#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
python -m tools.run_fair_suite \
  --dataset voc2012_segmentation --task semantic_segmentation \
  --download True --backbones auto \
  --methods linear,norm,bias,last_block,trso,full \
  --seeds 0,1,2 --epochs 50 --batch_size 16 --execute "$@"
