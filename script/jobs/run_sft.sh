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


cd /ist/users/patompornp/wangchanx/ChomGPT/script
source ../../../GMQA/envQA/bin/activate

nvidia-smi
python3 -c "import torch;print('# gpus: %d'%torch.cuda.device_count())"

# python3 -m torch.distributed.launch --nproc_per_node=8 train_sft.py \
python3 train_sft.py \
    --model_name=/ist/users/patompornp/models/facebook/xglm-7.5B \
    --dataset_name=/ist/users/patompornp/datasets/pythainlp/alpaca_en_sft \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=16 \
    --bf16 \
    --deepspeed=../config/sft_deepspeed_config.json
