# build on top of ../Dockerfile
FROM mlperf-megatron:24.05

RUN echo '\
Acquire::Retries "100";\
Acquire::https::Timeout "240";\
Acquire::http::Timeout "240";\
APT::Get::Assume-Yes "true";\
APT::Install-Recommends "false";\
APT::Install-Suggests "false";\
Debug::Acquire::https "true";\
' > /etc/apt/apt.conf.d/99custom

RUN apt-get update
RUN apt-get install -y python3-venv
RUN apt-get install -y libhwloc-dev  # for the NCCL OFI plugin
RUN apt-get install -y libsndfile1 ffmpeg libopencc-dev  # NeMo dependencies

RUN pip install --no-cache-dir -vvv \
  transformers==4.39.3

# Custom libraries version
WORKDIR /workspace/

# ## 0. Fix for https://github.com/NVIDIA/nccl/issues/957
# ARG NCCL_COMMIT_ID=4365458757e4107ecbf629b2fd6e0e19a5d237c2

# RUN export DEBIAN_FRONTEND=noninteractive &&        \
#     apt-get update &&                               \
#     apt-get remove -y libnccl2 &&                   \
#     git clone https://github.com/NVIDIA/nccl.git && \
#     cd nccl &&                                      \
#     echo "NCCL_COMMIT_ID=${NCCL_COMMIT_ID}" &&      \
#     git checkout "${NCCL_COMMIT_ID}" &&             \
#     make -j24 install
# ENV NCCL_VERSION "${NCCL_VERSION}-fix-${NCCL_COMMIT_ID}"

# ## 1. Apex
# ARG APEX_REVISION=SKIP
# ENV CUSTOM_APEX_REVISION ${APEX_REVISION}
# ARG APEX_MAX_JOBS=4

# RUN if [ "${APEX_REVISION}" != SKIP ]; then \
#       git clone https://github.com/NVIDIA/apex && \
#       cd apex && \
#       echo APEX_REVISION=${APEX_REVISION} && \
#       git checkout ${APEX_REVISION} && \
#       MAX_JOBS=${APEX_MAX_JOBS} pip3 install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./ \
#     ; fi


RUN pip freeze

## 2. Transformer Engine
# the following two variables are needed when building TE
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_WITH_USERBUFFERS=1
ENV MPI_HOME=/usr/local/mpi
# ARG TE_REVISION=09ba84278a0e5b2eff80239599fbfc5e6beb389f
ARG TE_REVISION=3b4d9e8766b829d50ac78bb26f770fb8d9825ae7
RUN if [ "${TE_REVISION}" != SKIP ]; then \
      pip install --force-reinstall --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@${TE_REVISION} -vvv \
    ; fi

#  CXX=mpic++ MPI_CXX_COMPILER=CC MPI_C_COMPILER=cc MPI_HOME=$CUDA_HOME NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 pip install -vvv git+https://github.com/NVIDIA/TransformerEngine.git@3b4d9e8766b829d50ac78bb26f770fb8d9825ae7


# ARG TE_REVISION=build_aarch64
# RUN if [ "${TE_REVISION}" != SKIP ]; then \
#       pip install --force-reinstall -vvv --no-deps git+https://github.com/aleskubicek/TransformerEngine.git@${TE_REVISION}; \
#     ; fi

# ARG TE_REVISION=fix_barrier_for_aarch64
# RUN if [ "${TE_REVISION}" != SKIP ]; then \
#       pip install --force-reinstall -vvv --no-deps git+https://github.com/aurianer/TransformerEngine.git@${TE_REVISION}; \
#       cat te.log \
#     ; fi

# ARG TE_REVISION=ub_mpibarrier
# RUN if [ "${TE_REVISION}" != SKIP ]; then \
#       MAX_JOBS=40 pip install --force-reinstall -vvv --no-deps git+https://github.com/henrique/TransformerEngine.git@${TE_REVISION}; \
#       cat te.log \
#     ; fi

ENV CUSTOM_TE_REVISION ${TE_REVISION}

RUN pip install --no-cache-dir h5py==3.11.0 pkgconfig -vvv


## 3. NeMo
ARG NEMO_REVISION=bd0b818ac2eb420b1cea19eb8bcc6215c87b06d0
# ARG NEMO_REVISION=v1.21.0
ENV CUSTOM_NEMO_REVISION ${NEMO_REVISION}
ARG NEMO_BASE_VERSION=r1.20.0
# ARG NEMO_BASE_VERSION=${NEMO_REVISION}
ENV CUSTOM_NEMO_BASE_VERSION ${NEMO_BASE_VERSION}

### Base version
RUN if [ "${NEMO_REVISION}" == SKIP ]; then \
      if [ -d /opt/bignlp/NeMo ]; then \
        ln -s /opt/bignlp/NeMo \
      ; else \
        echo "Error: NEMO_REVISION=SKIP but there is no BigNLP NeMo installation in base image." && \
        exit 1 \
      ; fi \
    ; else \
      git clone https://github.com/NVIDIA/NeMo.git && \
      cd NeMo && \
      echo NEMO_REVISION=${NEMO_REVISION} && \
      git checkout ${NEMO_REVISION} && \
      pip uninstall -y nemo-toolkit && \
      pip install "cython<3.0.0" && \
      pip install --no-build-isolation -e ".[nlp]" \
    ; fi

# RUN pip install --no-cache-dir tensorstore==0.1.45 -vvv
# RUN pip install --no-cache-dir tensorstore -vvv

# RUN pip install "cython<3.0.0"
# RUN pip install --no-build-isolation nemo_toolkit['nlp']==1.23.0 -vvv

# # # Install flash-attention 2.0.8
# # RUN MAX_JOBS=40 pip install --upgrade --no-deps flash-attn==2.0.8

### Make (has to be called after all changes to repo)
RUN cd NeMo && \
      cd nemo/collections/nlp/data/language_modeling/megatron && \
      make

## 4. Megatron-core
ARG MEGATRON_REVISION=df7271285d3e29c13865515a51dbbd6e25f68f5f
# ARG MEGATRON_REVISION=core_v0.6.0
ENV CUSTOM_MEGATRON_REVISION ${MEGATRON_REVISION}

RUN if [ "${MEGATRON_REVISION}" != SKIP ]; then \
      pip uninstall -y megatron-core && \
      MAX_JOBS=72 pip install git+https://github.com/nvidia/Megatron-LM.git@${MEGATRON_REVISION} \
    ; fi


## 4.5 Add old Nsight
#RUN installNSYS.sh 2023.2.1

# # Benchmark code
# WORKDIR /workspace/llm
# COPY . .

RUN pip install sentencepiece==0.2.0

RUN pip list -v

# ENV PYTHONPATH "/workspace/llm:/workspace/NeMo:${PYTHONPATH}"
