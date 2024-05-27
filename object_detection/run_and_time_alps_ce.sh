#!/bin/bash
#SBATCH --job-name mlperf-object-detection
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2

mkdir -p logs

# Set torch.distributed variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

export DATASET_CATALOG_DIR="/mchstor2/scratch/cscs/lukasd/mlperf/data/object_detection/pytorch/datasets"
export TORCH_MODEL_ZOO="/mchstor2/scratch/cscs/lukasd/mlperf/data/object_detection/pytorch/models"

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
  export TORCH_NCCL_BLOCKING_WAIT=1
  ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
else
  ENROOT_ENTRYPOINT=""
fi

set -x
srun -ul --environment="$(realpath env/ngc-object_detection-24.03.toml)" ${ENROOT_ENTRYPOINT} bash -c "\
  hostname
  if [ \"\$SLURM_LOCALID\" -eq 0 ]; then
    cp pytorch/maskrcnn_benchmark/config/paths_catalog.py /usr/local/lib/python3.10/dist-packages/maskrcnn_benchmark/config/
  fi
  RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID ./run_and_time.sh
"
