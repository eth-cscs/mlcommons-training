#!/bin/bash

#SBATCH --job-name criteo-materialize
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288

DATA_DIR="/mchstor2/scratch/cscs/lukasd/mlperf/data/recommendation/criteo"
PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH=${DATA_DIR}/numpy_contiguous_shuffled
MATERIALIZED_DATASET_PATH=${DATA_DIR}/synthetic_multihot

mkdir -p logs ${MATERIALIZED_DATASET_PATH}

set -x
srun -ul --wait=0 --environment=$(realpath env/ngc-recommendation_v2-24.03.toml) bash -c "\
cd scripts && \
python materialize_synthetic_multihot_dataset.py \
    --in_memory_binary_criteo_path $PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH \
    --output_path $MATERIALIZED_DATASET_PATH \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
    --multi_hot_distribution_type uniform
"  # --copy_labels_and_dense is optional, copies the labels and dense features (using symbolic links instead)

for d in $(seq 0 23); do
    ln -s ${PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH}/day_${d}_dense.npy ${MATERIALIZED_DATASET_PATH}/day_${d}_dense.npy
    ln -s ${PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH}/day_${d}_labels.npy ${MATERIALIZED_DATASET_PATH}/day_${d}_labels.npy
done