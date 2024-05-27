#!/usr/bin/env bash

: "${BASE_DATA_DIR:=/mchstor2/scratch/cscs/aurianer/mlperf/data/stable_diffusion}"

# : "${NUM_NODES:=8}"
: "${NUM_NODES:=1}"
# : "${GPUS_PER_NODE:=8}"
: "${GPUS_PER_NODE:=4}"
# : "${CONFIG:=./configs/train_512_latents.yaml}"
: "${CONFIG:=./configs/alps_train_01x04x08.yaml}"  # or alps_train_32x04x08.yaml
# : "${WORKDIR:=/workdir}"
: "${RESULTS_MNT:=./results}"
# : "${MOUNTS:=}"
# : "${CONTAINER_IMAGE:=mlperf_sd:22.12-py3}"
: "${CHECKPOINT:=${BASE_DATA_DIR}/checkpoints/sd/512-base-ema.ckpt}"

while [ "$1" != "" ]; do
    case $1 in
        --num-nodes )       shift
                            NUM_NODES=$1
                            ;;
        --gpus-per-node )   shift
                            GPUS_PER_NODE=$1
                            ;;
        --config )          shift
                            CONFIG=$1
                            ;;
        --checkpoint )      shift
                            CHECKPOINT=$1
                            ;;
        # --workdir )         shift
        #                     WORKDIR=$1
        #                     ;;
        --results-dir )         shift
                            RESULTS_MNT=$1
                            ;;
        # --mounts )          shift
        #                     MOUNTS=$1
        #                     ;;
        # --container )       shift
        #                     CONTAINER_IMAGE=$1
        #                     ;;
    esac
    shift
done

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
    echo "Enabling debugging..."
    ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
    if [ "${DEBUG_RANK:-0}" -ge "$SLURM_NTASKS" ]; then
        echo "DEBUG_RANK = ${DEBUG_RANK:-0} is not a valid rank (#ranks = $SLURM_NTASKS), exiting..."
        exit 1
    fi
else
    ENROOT_ENTRYPOINT=""
fi

    # --container-image="${CONTAINER_IMAGE}" \
    # --container-mounts="${MOUNTS}" \
    # --container-workdir="${WORKDIR}" \
srun \
    --environment="$(realpath env/ngc-stable-diffusion-24.01.toml)" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --nodes="${NUM_NODES}" \
    ${ENROOT_ENTRYPOINT} \
    bash -c  "if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ] && [ \"\${SLURM_PROCID:-0}\" -eq ${DEBUG_RANK:-0} ]; then
                  echo \"Running training script with debugpy on \$(hostname)\"
                  mkdir -p ${SCRATCH}/.tmp
                  echo \"\$(hostname)\" > ${SCRATCH}/.tmp/debug-\${SLURM_JOB_NAME}
              fi
              CUDA_VISIBLE_DEVICES=\"\$(seq -s, 0 $((${GPUS_PER_NODE}-1)) )\" ./run_and_time.sh \
                --num-nodes ${NUM_NODES} \
                --gpus-per-node ${GPUS_PER_NODE} \
                --checkpoint ${CHECKPOINT} \
                --results-dir ${RESULTS_MNT}  \
                --config ${CONFIG}"
