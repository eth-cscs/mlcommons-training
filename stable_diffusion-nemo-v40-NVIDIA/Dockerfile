# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

# Note: this should be build with --ulimit nofile=$(ulimit -n):$(ulimit -n)

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.04-py3
FROM ${FROM_IMAGE_NAME}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -yqq ffmpeg libsm6 libxext6
# Cloning nemo is failing with "CRLfile: none" without this
RUN apt-get install --reinstall ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid warnings, pip used for benchmark 24.1.1
RUN pip install --quiet --upgrade pip

# NeMo dependencies
RUN pip install -vvv tensorstore==0.1.45
RUN pip install -vvv causal-conv1d==1.2.0.post2 opencc-python-reimplemented==0.1.7

# We use a different tag to install the dependencies otherwise nemo fails at runtime
ENV NEMO_REVISION=v2.0.0.rc0.beta
ENV NEMO_REVISION_DEPS=v2.0.0rc0
RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    git checkout ${NEMO_REVISION_DEPS} && \
    echo NEMO_DEPS_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip uninstall -y nemo-toolkit && \
    pip install pkgconfig py-cpuinfo "cython<3.0.0" && \
    pip install --no-build-isolation -e ".[nlp]" && \
    echo NEMO_REVISION=${NEMO_REVISION} && \
    git checkout ${NEMO_REVISION} && \
    echo NEMO_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip install --no-build-isolation --no-deps -e ".[nlp]" \

# Build NeMo (has to be called after all changes to repo)
RUN cd NeMo && \
      cd nemo/collections/nlp/data/language_modeling/megatron && \
      make

ARG MEGATRON_REVISION=core_v0.7.0.beta
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    echo MEGATRON_REVISION=${MEGATRON_REVISION} && \
    git checkout ${MEGATRON_REVISION} && \
    echo MEGATRON_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip install . && \
    cd megatron/core/datasets && \
    make
ENV PYTHONPATH "${PYTHONPATH}:/workspace/Megatron-LM"

## 2. Transformer Engine +fix for arm64
ARG TE_REVISION=3b4d9e8766b829d50ac78bb26f770fb8d9825ae7 # v1.6rc1
RUN if [ "${TE_REVISION}" != SKIP ]; then \
      NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --force-reinstall --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@${TE_REVISION} \
    ; fi

ENV MAX_JOBS=4
ARG APEX_REVISION=24.04.01
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    git checkout ${APEX_REVISION} && \
    CFLAGS="-g0" NVCC_APPEND_FLAGS="--threads 8" pip install -v --no-build-isolation --no-cache-dir --disable-pip-version-check --config-settings "--build-option=--cpp_ext --cuda_ext --bnp --xentropy --deprecated_fused_adam --deprecated_fused_lamb --fast_multihead_attn --distributed_lamb --fast_layer_norm --transducer --distributed_adam --fmha --fast_bottleneck --nccl_p2p --peer_memory --permutation_search --focal_loss --fused_conv_bias_relu --index_mul_2d --cudnn_gbn --group_norm --gpu_direct_storage" . && \
    rm -rf build

# Set working directory
WORKDIR /workspace/sd

# Copy code
COPY . .

# install LDM
RUN pip install -r requirements.txt
