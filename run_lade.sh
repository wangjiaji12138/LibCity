#!/bin/bash

# 定义模型列表
MODELS=(AGCRN ASTGCN DCRNN DSTGN D2STGNN GWNET MTGNN STAEformer STID TimeMixer)
# 定义数据集列表
DATASETS=(LaDe_SH LaDe_CQ LaDe_HZ LaDe_YT LaDe_JL)

# 固定任务
TASK="traffic_state_pred"

# 遍历所有模型和数据集
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "Running model $MODEL on dataset $DATASET"
        python run_model.py --task $TASK --dataset $DATASET --model $MODEL 
        # 检查执行状态
        if [ $? -ne 0 ]; then
            echo "Error: Failed to run $MODEL on $DATASET"
            # 可以选择退出或继续
            # exit 1
        fi
    done
done