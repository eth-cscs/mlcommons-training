#!/bin/bash

#SBATCH --job-name mlperf-image_segmentation
#SBATCH --time 00:30:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH --output logs/slurm-%j.out

set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

SEED=${1:--1}

MAX_EPOCHS=4000
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=1000
EVALUATE_EVERY=20
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=200
DATASET_DIR="/capstor/scratch/cscs/dealmeih/ds/mlperf/data/image_segmentation/kits19/preprocessed"
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

mkdir -p results logs

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
    ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
else
    ENROOT_ENTRYPOINT=""
fi

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
  srun -u -n 1 --container-workdir=$(pwd) --environment="$(realpath env/ngc-image_segmentation-24.03.toml)" python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"

  srun -ul --container-workdir=$(pwd) --environment="$(realpath env/ngc-image_segmentation-24.03.toml)" ${ENROOT_ENTRYPOINT} \
bash -c "\
hostname
RANK=\$SLURM_PROCID \
WORLD_SIZE=\$SLURM_NTASKS \
LOCAL_RANK=\$SLURM_LOCALID \
python main.py --data_dir ${DATASET_DIR} \
  --epochs ${MAX_EPOCHS} \
  --evaluate_every ${EVALUATE_EVERY} \
  --start_eval_at ${START_EVAL_AT} \
  --quality_threshold ${QUALITY_THRESHOLD} \
  --batch_size ${BATCH_SIZE} \
  --optimizer sgd \
  --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --learning_rate ${LEARNING_RATE} \
  --seed ${SEED} \
  --lr_warmup_epochs ${LR_WARMUP_EPOCHS}
"
	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi