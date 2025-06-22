#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_multi_region_structured.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

export PYTHONPATH="/remote-home1/moss/LearningVerl/verl"
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    --config-name=demo \
    data.train_files=/remote-home1/moss/LearningVerl/verl_bak/examples/sft/customized/multi_region_data.parquet \
    data.val_files=/remote-home1/moss/LearningVerl/verl_bak/examples/sft/customized/multi_region_data.parquet \
    model.partial_pretrain=/remote-home1/share/models/Qwen2.5-0.5B-Instruct \
    optim.lr=1e-4 \
    data.micro_batch_size=4 \
    trainer.logger=['console'] \
    trainer.total_training_steps=1000 \
    trainer.default_hdfs_dir=null $@ \
    trainer.project_name=multi-region-sft \
    $@

    /remote-home1/moss/LearningVerl/verl/verl/trainer/fsdp_sft_trainer.py \