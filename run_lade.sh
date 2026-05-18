#!/usr/bin/env bash
set -euo pipefail

DATASETS=(LaDe_SH LaDe_CQ LaDe_HZ LaDe_YT LaDe_JL)
MODELS=(AGCRN ASTGCN D2STGNN DCRNN GWNET GMAN MTGNN STGCN STGNCDE STAEformer)
for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    cmd=(python run_model.py --task traffic_state_pred --dataset "$ds" --model "$model")
    echo ">>> running: ${cmd[*]}"
    "${cmd[@]}" 2>&1
  done
done

echo "全部任务已结束"
