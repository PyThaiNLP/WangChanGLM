#!/bin/bash

CUDA_VISIBLE_DEVICES=${1:-2,3,4,5}
N_GPUS=${2:-4}
BS=${3:-1}
GD_ACC=${4:-32}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export LOCAL_RANK=${LOCAL_RANK}

nvidia-smi
python3 -c "import torch;print('# gpus: %d'%torch.cuda.device_count())"

python3 -m torch.distributed.launch \
    --nproc_per_node=${N_GPUS} \
    train_sft.py \
    --per_device_train_batch_size=${BS} \
    --per_device_eval_batch_size=${BS} \
    --gradient_accumulation_steps=${GD_ACC} \
    --model_name=facebook/xglm-7.5B \
    --bf16 \
    --deepspeed=../config/sft_deepspeed_config.json \
    --is_logging 1