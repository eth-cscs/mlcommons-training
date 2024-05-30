#!/bin/bash -l

#SBATCH -J mlperf-imagenet
#SBATCH --time 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output logs/slurm-%x.%j.out

DATA_DIR=/mchstor2/scratch/cscs/lukasd/mlperf/data/image_classification/tfrecords/ILSVRC/Data/CLS-LOC
MODEL_DIR=/mchstor2/scratch/cscs/lukasd/mlperf/data/image_classification/model

mkdir -p logs

export TF_CPP_MIN_LOG_LEVEL=0
export NCCL_DEBUG=INFO

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
    ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
else
    ENROOT_ENTRYPOINT=""
fi

srun -ul --mpi=pmi2 --environment="$(realpath env/ngc-image_classification-24.03.toml)" ${ENROOT_ENTRYPOINT} bash -c "
hostname
cd tensorflow2 && \
unset http_proxy https_proxy && \
CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID \
python3 ./resnet_ctl_imagenet_main.py \
--base_learning_rate=8.5 \
--batch_size=1024 \
--clean \
--data_dir=${DATA_DIR} \
--datasets_num_private_threads=32 \
--dtype=fp32 \
--device_warmup_steps=1 \
--distribution_strategy=mirrored \
--all_reduce_alg=nccl \
--noenable_device_warmup \
--enable_eager \
--noenable_xla \
--epochs_between_evals=4 \
--noeval_dataset_cache \
--eval_offset_epochs=2 \
--eval_prefetch_batchs=192 \
--label_smoothing=0.1 \
--lars_epsilon=0 \
--log_steps=125 \
--lr_schedule=polynomial \
--model_dir=${MODEL_DIR} \
--momentum=0.9 \
--num_accumulation_steps=2 \
--num_classes=1000 \
--num_gpus=$SLURM_NTASKS \
--optimizer=LARS \
--noreport_accuracy_metrics \
--single_l2_loss_op \
--noskip_eval \
--steps_per_loop=1252 \
--target_accuracy=0.759 \
--notf_data_experimental_slack \
--tf_gpu_thread_mode=gpu_private \
--notrace_warmup \
--train_epochs=41 \
--notraining_dataset_cache \
--training_prefetch_batchs=128 \
--nouse_synthetic_data \
--warmup_epochs=5 \
--weight_decay=0.0002 \
--use_tf_keras_layers
"
