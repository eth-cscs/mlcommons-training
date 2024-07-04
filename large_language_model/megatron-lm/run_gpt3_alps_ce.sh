#!/bin/bash

#SBATCH --job-name mlperf-megatron
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
##SBATCH -p luna -A mlperf -t 00:20:00 --nodes=8 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --job-name=mlperf-megatron:megatron

# Execute via:
# GBS=4 USE_BF16=true EXTERNAL_GBS=4 sbatch run_gpt3_alps_ce.sh logs /capstor/scratch/cscs/dealmeih/ds/mlperf/data/megatron-lm/preprocessed_c4_spm none

set -ex
export DIR=$(dirname $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}' | head -n 1))

# Vars without defaults
LOG_DIR=${1:-$DIR/logs}
BPE_DIR=${2:-/capstor/scratch/cscs/dealmeih/ds/mlperf/data/megatron-lm/preprocessed_c4_spm}
# CONT="${3:?CONT not set}"

# Set torch.distributed variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500 # default from torch launcher
export WORLD_SIZE=$SLURM_NTASKS

# export TORCH_EXTENSIONS_DIR=$SCRATCH/.torch_ext
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=64

# Vars with defaults
: "${MEGATRON_DIR:=$DIR}"
: "${GBS:=1536}"
: "${LR:=2.0e-5}"
: "${MIN_LR:=2.0e-6}"
: "${EVAL_INTERVAL:=$(( (24576 + ${GBS} - 1) / ${GBS} ))}"
: "${USE_BF16:=true}"  # set to false for FP32
: "${EXTERNAL_MODEL_CHECKPOINT_DIR:=}"
: "${EXTERNAL_TRAINING_ITERATIONS:=4000}"
: "${EXTERNAL_GBS:=1536}"

# Setup directories
CHECKPOINT_DIR="${LOG_DIR}/GPT3-175B/${SLURM_JOBID}"
TENSORBOARD_DIR="${LOG_DIR}/GPT3-175B"

mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. $DIR/gpt3_blend.sh

################################################################################
### Set exit duration based on variable time allocated for this specific job ###
# Query Slurm for the remaining job time left in the format [days-]hh:mm:ss
# format and pass the time (in units of minutes) to Megatron using variable
# EXIT_DURATION. The actual value passed is actually 13 minutes less for time
# to save model and extra margin. For our purposes we assume the days field
# will never be present to make parsing in bash easier. Note that setting
# EXIT_DURATION to 0 will terminate the job after 1 iteration.
timeleft=`squeue -j ${SLURM_JOBID} --noheader --format=%L`
timeleft=(`echo $timeleft | tr ':' ' '`)
EXIT_DURATION=$((timeleft[0]*60 + timeleft[1] - 15))
echo "setting exit duration to $EXIT_DURATION minutes"
################################################################################

options=" \
--exit-duration-in-mins ${EXIT_DURATION} \
--tensor-model-parallel-size 4 \
--pipeline-model-parallel-size 16 \
--sequence-parallel \
--recompute-activations \
--num-layers 96 \
--hidden-size 12288 \
--num-attention-heads 96 \
--seq-length 2048 \
--max-position-embeddings 2048 \
--micro-batch-size 1 \
--global-batch-size ${GBS} \
--train-samples 20000000 \
--lr-warmup-samples 407040 \
--lr-decay-samples 166809600 \
--lr ${LR} \
--min-lr ${MIN_LR} \
--lr-decay-style cosine \
--log-interval 1 \
--eval-iters -1 \
--eval-interval ${EVAL_INTERVAL} \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--train-data-path ${DATA_BLEND} \
--valid-data-path ${VALID_DATA_BLEND} \
--vocab-file ${BPE_DIR}/vocab.json \
--merge-file ${BPE_DIR}/merges.txt \
--save-interval 500 \
--save ${CHECKPOINT_DIR} \
--do-layernorm-bias-weight-decay \
--no-scaled-init \
--loss-scale 1.0 \
--split 100,0,0 \
--clip-grad 1.0 \
--weight-decay 0.1 \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--init-method-std 0.006 \
--log-params-norm \
--log-num-zeros-in-grad \
--log-validation-ppl-to-tensorboard \
--DDP-impl local \
--tensorboard-dir ${TENSORBOARD_DIR} \
--no-query-key-layer-scaling \
--no-seq-len-plus-one-tokens \
--seed ${RANDOM} "

[ ${USE_BF16} = true ] && options+=" --bf16"
if [ -n "${EXTERNAL_MODEL_CHECKPOINT_DIR}" ]; then
  options+=" \
		--no-load-rng \
		--use-ext-ckpt \
		--ext-iterations $(( $EXTERNAL_TRAINING_ITERATIONS * $EXTERNAL_GBS / $GBS)) \
		--ext-lr-steps $(( $EXTERNAL_TRAINING_ITERATIONS * $EXTERNAL_GBS)) \
		--load ${EXTERNAL_MODEL_CHECKPOINT_DIR}"
else
  options+=" --load ${CHECKPOINT_DIR}"
fi

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
  echo "Enabling debugging..."
  ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
else
  ENROOT_ENTRYPOINT=""
fi

# Run
# run_cmd="python -u ${MEGATRON_DIR}/pretrain_gpt.py ${options}"
# run_cmd="WANDB_BASE_URL=... WANDB_API_KEY=... RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python -u ./pretrain_gpt.py ${options}"
run_cmd="RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID python ./pretrain_gpt.py ${options}"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

# export TORCH_NCCL_BLOCKING_WAIT=1
srun -ul --environment=$DIR/env/ngc-megatron.toml --container-workdir=$DIR \
  ${ENROOT_ENTRYPOINT} sh -c "
  hostname
  ${run_cmd}
"

# srun -l \
#      --container-image $CONT \
#      --container-mounts "$DIR:$DIR,${COM_DIR}:${COM_DIR},${LOG_DIR}:${LOG_DIR},${BPE_DIR}:${BPE_DIR}" \
#      --output=$LOG_DIR/GPT3-175B-runlog-$DATETIME.log sh -c "${run_cmd}"

set +x

