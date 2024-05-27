#!/bin/bash

set -euo pipefail

set -x
cd $(dirname $0)

podman build --ulimit nofile=$(ulimit -n):$(ulimit -n) -t lukasgd/ngc-rnn_speech_recognition:24.03 .
enroot import -x mount -o /bret/scratch/cscs/lukasd/images/ngc-rnn_speech_recognition+24.03.sqsh podman://lukasgd/ngc-rnn_speech_recognition:24.03
set +x
