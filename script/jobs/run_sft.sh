#!/bin/bash -l
#SBATCH --error=log/task.out.%j  # STDOUT output is written in slurm.out.JOBID
#SBATCH --output=log/task.out.%j # STDOUT error is written in slurm.err.JOBID
#SBATCH --job-name=ChomGPT       # Job name
#SBATCH --mem=16GB                  # Memory request for this job
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1                   # The number of nodes
#SBATCH --partition=scads-a100
#SBATCH --account=scads
#SBATCH --time=72:0:0                # Runing time 72 hours
#SBATCH --gpus=1

# module load OpenMPI/4.1.1-GCC-10.3.0

# cd /ist/users/patompornp/wangchanx/ChomGPT/script
# source ../../../GMQA/envQA/bin/activate


cd /ist/users/patompornp/wangchanx/ChomGPT/script
conda activate chat

nvidia-smi
python3 -c "import torch;print('# gpus: %d'%torch.cuda.device_count())"

### Vanilla training script
python train_sft.py \
    --model_name=/ist/users/patompornp/models/facebook/xglm-7.5B \
    --dataset_name=/ist/users/patompornp/datasets/pythainlp/alpaca_en_sft \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=16 \
    --model_name=facebook/xglm-7.5B \
    --bf16 \
    --deepspeed=../config/sft_deepspeed_config.json \
    --is_logging 0

### Distributed training script
python -m torch.distributed.launch --nproc_per_node=1 train_sft.py \
    --model_name=/ist/users/patompornp/models/facebook/xglm-7.5B \
    --dataset_name=/ist/users/patompornp/datasets/pythainlp/alpaca_en_sft \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=16 \
    --model_name=facebook/xglm-7.5B \
    --bf16 \
    --deepspeed=../config/sft_deepspeed_config.json \
    --is_logging 0