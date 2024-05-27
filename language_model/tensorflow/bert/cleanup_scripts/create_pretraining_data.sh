#!/bin/bash

#SBATCH --job-name mlperf-bert-preprocess
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --output logs/slurm-%x.%j.out

set -e

mkdir -p logs

DATASET_DIR="/mchstor2/scratch/cscs/lukasd/mlperf/data/language_model_bert"

set -x
srun -u --environment="$(realpath env/ngc-language_model_bert-24.04.toml)" bash -c "
  export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID

  # Generate one TFRecord for each part-00XXX-of-00500 file. The following command is for generating one corresponding TFRecord
  for part_file in \"${DATASET_DIR}/processed_dataset/results4/part-\"*; do
    python cleanup_scripts/create_pretraining_data.py \
      --input_file=\$part_file \
      --output_file=${DATASET_DIR}/pretraining_dataset/results4/\$(basename \$part_file) \
      --vocab_file=${DATASET_DIR}/input_files/vocab.txt \
      --do_lower_case=True \
      --max_seq_length=512 \
      --max_predictions_per_seq=76 \
      --masked_lm_prob=0.15 \
      --random_seed=12345 \
      --dupe_factor=10
  done

  # Use the following steps for the eval set:
  python cleanup_scripts/create_pretraining_data.py \
  --input_file=${DATASET_DIR}/processed_dataset/results4/eval.txt \
  --output_file=${DATASET_DIR}/pretraining_dataset/results4/eval_intermediate \
  --vocab_file=${DATASET_DIR}/input_files/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

python cleanup_scripts/pick_eval_samples.py \
  --input_tfrecord=${DATASET_DIR}/pretraining_dataset/results4/eval_intermediate \
  --output_tfrecord=${DATASET_DIR}/pretraining_dataset/results4/eval_10k \
  --num_examples_to_pick=10000
"
