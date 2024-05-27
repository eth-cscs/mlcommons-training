#!/bin/bash

# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set +x
set -e

# Only rank print
[ "${SLURM_LOCALID-}" -ne 0 ] && set +x


# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set variables
[ "${DEBUG}" = "1" ] && set -x
BATCHSIZE=${BATCHSIZE:-2}
EVALBATCHSIZE=${EVALBATCHSIZE:-${BATCHSIZE}}
NUMEPOCHS=${NUMEPOCHS:-30}
LOG_INTERVAL=${LOG_INTERVAL:-20}
DATASET_DIR=${DATASET_DIR:-"/mchstor2/scratch/cscs/lukasd/mlperf/data/single_stage_detector/open-images-v6-mlperf"}
TORCH_HOME=${TORCH_HOME:-"/mchstor2/scratch/cscs/lukasd/mlperf/data/single_stage_detector/torch-model-cache"}

# Handle MLCube parameters
while [ $# -gt 0 ]; do
  case "$1" in
    --data_dir=*)
      DATASET_DIR="${1#*=}"
      ;;
    --log_dir=*)
      LOG_DIR="${1#*=}"
      ;;
    *)
  esac
  shift
done

# run training
python train.py \
  --batch-size "${BATCHSIZE}" \
  --eval-batch-size "${EVALBATCHSIZE}" \
  --epochs "${NUMEPOCHS}" \
  --print-freq "${LOG_INTERVAL}" \
  --data-path "${DATASET_DIR}" \
  --output-dir "${LOG_DIR}" \
  ${EXTRA_PARAMS} ; ret_code=$?

# Copy log file to MLCube log folder
if [ "$LOG_DIR" != "" ]; then
  timestamp=$(date +%Y%m%d_%H%M%S)
  cp mlperf_compliance.log "$LOG_DIR/mlperf_compliance_$timestamp.log"
fi

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="SINGLE_STAGE_DETECTOR"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
