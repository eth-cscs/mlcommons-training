#!/bin/bash

set -euo pipefail

set -x
cd $(dirname $0)

pushd ..
podman build --ulimit nofile=$(ulimit -n):$(ulimit -n) -f env/Dockerfile -t lukasgd/ngc-stable-diffusion:24.01 .
enroot import -x mount -o /bret/scratch/cscs/lukasd/images/ngc-stable-diffusion+24.01.sqsh podman://lukasgd/ngc-stable-diffusion:24.01
popd
set +x
