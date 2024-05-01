#!/bin/bash
 
podman build --ulimit nofile=$(ulimit -n):$(ulimit -n) -t lukasgd/ngc-megatron:24.03 .
enroot import -x mount -o /bret/scratch/cscs/lukasd/images/ngc-megatron+24.03.sqsh podman://lukasgd/ngc-megatron:24.03
