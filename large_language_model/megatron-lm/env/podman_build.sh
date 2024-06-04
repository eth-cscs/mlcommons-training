#!/bin/bash
 
podman build --ulimit nofile=$(ulimit -n):$(ulimit -n) -t mlperf-megatron:24.05 .

mkdir -p $SCRATCH/images
enroot import -x mount -o $SCRATCH/images/mlperf-megatron+24.05.sqsh podman://mlperf-megatron:24.05

