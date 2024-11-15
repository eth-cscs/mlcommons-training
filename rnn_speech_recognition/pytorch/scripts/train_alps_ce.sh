#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH -J mlperf-rnn-speech-recognition
#SBATCH --time 2:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH --output logs/slurm-%x.%j.out

export OMP_NUM_THREADS=1

: ${DATA_DIR:=${1:-"/capstor/scratch/cscs/dealmeih/ds/mlperf/data/rnn_speech_recognition/datasets/LibriSpeech/"}}
: ${MODEL_CONFIG:=${2:-"configs/baseline_v3-1023sp.yaml"}}
: ${OUTPUT_DIR:=${3:-"./results"}}
: ${CHECKPOINT:=${4:-}}
: ${CUDNN_BENCHMARK:=true}
: ${NUM_GPUS:=${SLURM_NTASKS}}
: ${AMP:=false}
: ${GLOBAL_BATCH_SIZE:=1024}
: ${VAL_BATCH_SIZE:=2}
: ${GRAD_ACCUMULATION_STEPS:=8}
: ${LEARNING_RATE:=0.004}
: ${LR_EXP_GAMMA:=0.935}  # ~0.005 in 80 epochs
: ${NUM_BUCKETS=1} # empty means to use torch.utils.data.distributed.DistributedSampler
: ${EMA:=0.999}
: ${SEED=1}
: ${EPOCHS:=100}
: ${WARMUP_EPOCHS:=6}  # 8000 steps with 1x8x24 should be ~5.6 epochs
: ${HOLD_EPOCHS:=40}
: ${SAVE_AT_THE_END:=false}
: ${EPOCHS_THIS_JOB:=0}
: ${RESUME:=true}
: ${DALI_DEVICE:="cpu"}
: ${VAL_FREQUENCY:=1}
: ${PREDICTION_FREQUENCY:=1000}
: ${BETA1:=0.9}
: ${BETA2:=0.999}
: ${LOG_FREQUENCY:=1}
: ${TRAIN_MANIFESTS:="$DATA_DIR/librispeech-train-clean-100-wav.json \
                      $DATA_DIR/librispeech-train-clean-360-wav.json \
                      $DATA_DIR/librispeech-train-other-500-wav.json"}
: ${VAL_MANIFESTS:="$DATA_DIR/librispeech-dev-clean-wav.json"}
: ${LOG_NORM:=false}
: ${USE_OLD_VAL:=true}
: ${USE_NEW_VAL:=false}
: ${MAX_SYMBOL_PER_SAMPLE=300}
: ${WEIGHTS_INIT_SCALE=0.5}
: ${CLIP_NORM:=1}

BATCH_SIZE=$(( $GLOBAL_BATCH_SIZE / $NUM_GPUS ))

mkdir -p "$OUTPUT_DIR"

ARGS="--dataset_dir=$DATA_DIR"
ARGS+=" --val_manifests $VAL_MANIFESTS"
ARGS+=" --train_manifests $TRAIN_MANIFESTS"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --lr=$LEARNING_RATE"
ARGS+=" --batch_size=$BATCH_SIZE"
ARGS+=" --val_batch_size=$VAL_BATCH_SIZE"
ARGS+=" --min_lr=1e-5"
ARGS+=" --lr_exp_gamma=$LR_EXP_GAMMA"
ARGS+=" --epochs=$EPOCHS"
ARGS+=" --warmup_epochs=$WARMUP_EPOCHS"
ARGS+=" --hold_epochs=$HOLD_EPOCHS"
ARGS+=" --epochs_this_job=$EPOCHS_THIS_JOB"
ARGS+=" --ema=$EMA"
ARGS+=" --seed=$SEED"
ARGS+=" --weight_decay=1e-3"
ARGS+=" --log_frequency=$LOG_FREQUENCY"
ARGS+=" --val_frequency=$VAL_FREQUENCY"
ARGS+=" --grad_accumulation_steps=$GRAD_ACCUMULATION_STEPS "
ARGS+=" --dali_device=$DALI_DEVICE"
ARGS+=" --beta1=$BETA1"
ARGS+=" --beta2=$BETA2"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$RESUME" = true ] &&              ARGS+=" --resume"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"
[ "$LOG_NORM" = true ] &&            ARGS+=" --log_norm"
[ "$SAVE_AT_THE_END" = true ] &&     ARGS+=" --save_at_the_end"
[ -n "$CHECKPOINT" ] &&              ARGS+=" --ckpt=$CHECKPOINT"
[ -n "$NUM_BUCKETS" ] &&             ARGS+=" --num_buckets=$NUM_BUCKETS"
[ -n "$TARGET" ] &&                  ARGS+=" --target=$TARGET"
[ -n "$CLIP_NORM" ] &&               ARGS+=" --clip_norm=$CLIP_NORM"
[ -n "$PREDICTION_FREQUENCY" ] &&    ARGS+=" --prediction_frequency=$PREDICTION_FREQUENCY"
[ -n "$SAVE_MILESTONES" ] &&         ARGS+=" --keep_milestones $SAVE_MILESTONES"
[ -n "$SAVE_BEST" ] &&               ARGS+=" --save_best_from=$SAVE_BEST"
[ -n "$SAVE_FREQUENCY" ] &&          ARGS+=" --save_frequency=$SAVE_FREQUENCY"
[ -n "$START_CLIP" ] &&              ARGS+=" --start_clip=$START_CLIP"
[ -n "$HIDDEN_HIDDEN_BIAS_SCALED" ] && ARGS+=" --hidden_hidden_bias_scale=$HIDDEN_HIDDEN_BIAS_SCALED"
[ -n "$WEIGHTS_INIT_SCALE" ] &&      ARGS+=" --weights_init_scale=$WEIGHTS_INIT_SCALE"
[ -n "$MAX_SYMBOL_PER_SAMPLE" ] &&  ARGS+=" --max_symbol_per_sample=$MAX_SYMBOL_PER_SAMPLE"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
    ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
else
    ENROOT_ENTRYPOINT=""
fi

srun -ul --container-workdir=$(pwd) --environment="$(realpath env/ngc-rnn_speech_recognition-24.03.toml)" ${ENROOT_ENTRYPOINT} bash -c " \
    hostname
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    python train.py ${ARGS}
"