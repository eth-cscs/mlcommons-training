#!/bin/bash
#SBATCH --job-name mlperf-dlrm
#SBATCH --time=02:00:00
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/slurm-%x.%j.out


# Create results directory
mkdir -p logs

## Prep run and launch
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE  # required by torchrec model-parallelism (adjust with CUDA_VISIBLE_DEVICES)


export TOTAL_TRAINING_SAMPLES=4195197692
export GLOBAL_BATCH_SIZE=65536

DATASET_DIR="/capstor/scratch/cscs/dealmeih/ds/mlperf/data/recommendation/criteo"
USE_MATERIALIZED_SYNTHETIC_MULTIHOT_DATASET=1

if [ "${USE_MATERIALIZED_SYNTHETIC_MULTIHOT_DATASET:-0}" -eq 1 ]; then
    DATASET_ARGS="\
    --synthetic_multi_hot_criteo_path ${DATASET_DIR}/synthetic_multihot \
"
else  # generate synthetic multi-hot dataset on-the-fly
    DATASET_ARGS="\
    --in_memory_binary_criteo_path ${DATASET_DIR}/numpy_contiguous_shuffled \
    --multi_hot_distribution_type uniform \
    --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
"
fi

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
    ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
else
    ENROOT_ENTRYPOINT=""
fi

set -x
# Launch benchmark
srun -ul --container-workdir=$(pwd) --environment="$(realpath env/ngc-recommendation_v2-24.03.toml)" ${ENROOT_ENTRYPOINT} bash -c "
hostname
RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID CUDA_VISIBLE_DEVICES=0,1,2,3 \
python dlrm_main.py \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    ${DATASET_ARGS} \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20))) \
    --epochs 1 \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --adagrad \
    --learning_rate 0.005
"
