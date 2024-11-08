#!/bin/bash -l

#SBATCH -J mlperf-imagenet
#SBATCH --time 24:00:00
#SBATCH --output logs/slurm-%x.%j.out

DATA_DIR=/capstor/scratch/cscs/dealmeih/ds/mlperf/data/image_classification

# `local_scratch_dir` will be where the TFRecords are stored.`
srun -u --container-workdir=$(pwd) --environment="$(realpath env/ngc-image_classification-24.03.toml)" bash -c "
cd preprocess && \
python imagenet_to_gcs.py \
  --raw_data_dir=${DATA_DIR}/ \
  --local_scratch_dir=${DATA_DIR}/tfrecords \
  --nogcs_upload \
"