#!/bin/bash -l
#SBATCH --job-name="podman-gpt3"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00            # total run time limit (HH:MM:SS)

set -ex

TAG=nvidia-megatron:24.05
IMAGE_NAME=$(echo ${TAG} | sed 's/:/./g')

cat ../Dockerfile
podman build -t mlperf-megatron:24.05 -f ../Dockerfile

cat Dockerfile
podman build -t $TAG .
enroot import -x mount -o $SCRATCH/images/$IMAGE_NAME.sqsh podman://$TAG
