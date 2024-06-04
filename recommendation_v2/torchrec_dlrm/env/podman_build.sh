#!/bin/bash

set -euo pipefail

set -x
cd $(dirname $0)

podman build --ulimit nofile=$(ulimit -n):$(ulimit -n) -t lukasgd/ngc-recommendation_v2:24.03 -f env/Dockerfile ..
enroot import -x mount -o /bret/scratch/cscs/lukasd/images/ngc-recommendation_v2+24.03.sqsh podman://lukasgd/ngc-recommendation_v2:24.03
set +x
