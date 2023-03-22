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


conda activate envWX
module load Anaconda3/2020.11

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ist/apps/modules/software/Anaconda3/5.3.0/lib

cd /ist/users/patompornp/wangchanx/ChomGPT

nvidia-smi
python3 -c "import torch;print('# gpus: %d'%torch.cuda.device_count())"

python3 script/ppo_with_trl/pantip.py
