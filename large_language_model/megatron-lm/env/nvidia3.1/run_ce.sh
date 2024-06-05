#!/bin/bash -l
#SBATCH --job-name="ce-gpt3"
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00            # total run time limit (HH:MM:SS)
# #SBATCH --output=log/%x-%j-%t.out  # https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E

set -x

# Setting the environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

# Extra debugging flags, slow down training
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Trace output
# TFOLDER_NAME=r1_7B_tp2_arOFF_topON
# mkdir -p traces/${TFOLDER_NAME}
# cp jobs/round1/${TFOLDER_NAME}.sbatch traces/${TFOLDER_NAME}/

# Distributed training variables
NNODES=${SLURM_NNODES}
GPUS_PER_NODE=4
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
export WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
export MASTER_PORT=12345
export MASTER_ADDR=$(hostname)


srun -ul --environment=$PWD/mega.toml --container-workdir=$PWD bash -c "
  set -x
  export TORCH_LOGS="+dynamo"
  export TORCHDYNAMO_VERBOSE=1
#   export NODE_RANK=\${SLURM_NODEID}
  SEED=42  bash run_and_time.sh  # WALLTIME=00:30:00 
"
