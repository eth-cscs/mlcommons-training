FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd


RUN apt-get update && \
    apt-get install -y libsndfile1 sox git cmake jq && \
    apt-get install -y --no-install-recommends numactl && \
    rm -rf /var/lib/apt/lists/*

# RUN COMMIT_SHA=f546575109111c455354861a0567c8aa794208a2 && \
#     git clone https://github.com/HawkAaron/warp-transducer deps/warp-transducer && \
#     cd deps/warp-transducer && \
#     git checkout $COMMIT_SHA && \
#     sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")/#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")/g' CMakeLists.txt && \
#     sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")/g' CMakeLists.txt && \
#     mkdir build && \
#     cd build && \
#     cmake .. && \
#     make VERBOSE=1 && \
#     export CUDA_HOME="/usr/local/cuda" && \
#     export WARP_RNNT_PATH=`pwd` && \
#     export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME && \
#     export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH" && \
#     export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH && \
#     export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
#     export CFLAGS="-I$CUDA_HOME/include $CFLAGS" && \
#     cd ../pytorch_binding && \
#     python3 setup.py install && \
#     rm -rf ../tests test ../tensorflow_binding && \
#     cd ../../..

WORKDIR /workspace/rnnt

# RUN pip install --no-cache --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110==0.28.0

# RUN pip install --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex/archive/8a1ed9e8d35dfad26fb973996319965e4224dcdd.zip

RUN pip install --no-cache --disable-pip-version-check -U \
    git+https://github.com/mlcommons/logging.git@3.1.0-rc1 \
    https://github.com/NVIDIA/dllogger/archive/26a0f8f1958de2c0c460925ff6102a4d2486d6cc.zip \
    inflect==4.1.0 \
    librosa==0.10.2 \
    pandas==1.1.5 \
    sentencepiece==0.1.94 \
    soundfile==0.12.1 \
    sox==1.4.1 \
    tensorboard==2.6.0 \
    unidecode==1.1.1 \
    git+https://github.com/lukasgd/warp-rnnt.git@master#subdirectory=pytorch_binding

# ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# COPY . .
