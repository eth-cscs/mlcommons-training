#!/bin/bash

set -euo pipefail

set -x
cd $(dirname $0)

podman build --ulimit nofile=$(ulimit -n):$(ulimit -n) -t lukasgd/ngc-object_detection:24.03 -f env/Dockerfile ..
enroot import -x mount -o /capstor/scratch/cscs/lukasd/images/ngc-object_detection+24.03.sqsh podman://lukasgd/ngc-object_detection:24.03
set +x
