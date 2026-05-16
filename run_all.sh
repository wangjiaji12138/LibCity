#!/usr/bin/env bash
set -euo pipefail

DATASETS=(PEMSD8 PEMSD3 PEMSD4 PEMSD7)
MODELS=(AGCRN ASTGCN D2STGNN DCRNN GWNET MTGNN STGCN STGNCDE STAEformer STID)
for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    
    cmd=(python run_model.py --task traffic_state_pred --dataset "$ds" --model "$model" --max_epoch 1)
    echo ">>> running: ${cmd[*]}"
    "${cmd[@]}" 2>&1
  done
done

echo "全部任务已结束"
