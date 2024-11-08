#!/bin/bash

#SBATCH --job-name mlperf-bert
#SBATCH --time 2:00:00
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --output logs/slurm-%x.%j.out

set -e

LOG_DIR=logs

mkdir -p ${LOG_DIR}/BERT-$SLURM_JOB_ID

DATASET_DIR="/capstor/scratch/cscs/dealmeih/ds/mlperf/data/language_model_bert"

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
    ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
else
    ENROOT_ENTRYPOINT=""
fi

set -x
srun -ul --mpi=pmi2 --container-workdir=$(pwd) --environment="$(realpath env/ngc-language_model_bert-24.04.toml)" ${ENROOT_ENTRYPOINT} bash -c "
  hostname
  unset http_proxy https_proxy && \
  CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID \
  TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
  python run_pretraining.py \
    --bert_config_file=${DATASET_DIR}/input_files/bert_config.json \
    --output_dir=${LOG_DIR}/BERT-$SLURM_JOB_ID \
    --input_file=\"${DATASET_DIR}/pretraining_dataset/results4/part*\" \
    --nodo_eval \
    --do_train \
    --eval_batch_size=8 \
    --learning_rate=0.0001 \
    --init_checkpoint=${DATASET_DIR}/input_files/tf2_ckpt/model.ckpt-28252 \
    --iterations_per_loop=1000 \
    --max_predictions_per_seq=76 \
    --max_seq_length=512 \
    --num_train_steps=107538 \
    --num_warmup_steps=1562 \
    --optimizer=lamb \
    --save_checkpoints_steps=6250 \
    --start_warmup_step=0 \
    --num_gpus=\$SLURM_NTASKS \
    --train_batch_size=24
"
