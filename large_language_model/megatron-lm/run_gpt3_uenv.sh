#!/bin/bash

#SBATCH --job-name mlperf-megatron
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --error=logs/slurm-%x.%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive

# Execute via:
# GBS=4 USE_BF16=true sbatch --uenv=/bret/scratch/cscs/boeschf/images/ml-base-megatron-pytorch-2.3.0.squashfs run_gpt3_uenv.sh logs /mchstor2/scratch/cscs/lukasd/mlperf/data/megatron-lm/preprocessed_c4_spm

set -x

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

. /user-environment/env/default/activate.sh
export LD_LIBRARY_PATH=/user-environment/env/default/lib64:$LD_LIBRARY_PATH

# Vars without defaults
LOG_DIR=${1:?LOG_DIR not set}
BPE_DIR=${2:?BPE_DIR not set}

export NCCL_DEBUG=INFO
#export NCCL_DEBUG=TRACE

# Vars with defaults
: "${MEGATRON_DIR:=$PWD}"
: "${GBS:=1536}"
: "${LR:=2.0e-5}"
: "${MIN_LR:=2.0e-6}"
: "${EVAL_INTERVAL:=$(( (24576 + ${GBS} - 1) / ${GBS} ))}"
: "${USE_BF16:=true}"  # set to false for FP32
: "${EXTERNAL_MODEL_CHECKPOINT_DIR:=}"
: "${EXTERNAL_TRAINING_ITERATIONS:=4000}"
: "${EXTERNAL_GBS:=1536}"

# Setup directories
CHECKPOINT_DIR="${LOG_DIR}/GPT3-175B-checkpoints"
TENSORBOARD_DIR="${LOG_DIR}/GPT3-175B-tensorboard"

mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. $PWD/gpt3_blend.sh

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
--tensor-model-parallel-size 8 \
--pipeline-model-parallel-size 8 \
--distributed-backend nccl \
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

# Run
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

srun -ul \
    --cpu-bind=verbose,rank_ldom \
    --output=$LOG_DIR/GPT3-175B-runlog-$SLURM_JOB_ID-$DATETIME.log \
    ../../launch_wrapper \
    python -u ${MEGATRON_DIR}/pretrain_gpt.py ${options}

set +x

