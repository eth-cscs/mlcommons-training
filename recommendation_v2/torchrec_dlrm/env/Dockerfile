FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd


RUN pip install -vvv --no-cache-dir \
  git+https://github.com/mlperf/logging.git
  # fbgemm-gpu \
  # torchmetrics \
  # torchrec

# due to set_python_module not defined in TORCH_LIBRARY_FRAGMENT
RUN cd /tmp \
  && export CUDA_BIN_PATH="/usr/local/cuda/" \
  && export CUDACXX="/usr/local/cuda/bin/nvcc" \
  && export CUDNN_INCLUDE_DIR="/usr/local/cuda/include/" \
  && export CUDNN_LIBRARY="/usr/local/cuda/lib64/" \
  && export TORCH_CUDA_ARCH_LIST="Hopper" \
  && export _GLIBCXX_USE_CXX11_ABI=1 \
  && git clone --branch v0.6.0 --recursive https://github.com/pytorch/FBGEMM.git \
  && cd FBGEMM/fbgemm_gpu \
  && git submodule sync \
  && git submodule update --init --recursive \
  && pip install -r requirements.txt \
  && sed -i 's/os\.cpu_count() \/\/ 2/os.cpu_count() \/\/ 8/g' setup.py \
  && MAX_JOBS=16 python setup.py install

RUN cd /tmp \
  && export CUDA_NVCC_EXECUTABLE="/usr/local/cuda/bin/nvcc" \
  && export CUDA_HOME="/usr/local/cuda" \
  && export CUDNN_INCLUDE_PATH="/usr/local/cuda/include/" \
  && export CUDNN_LIBRARY_PATH="/usr/local/cuda/lib64/" \
  && export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda/lib64" \
  && export USE_CUDA=1 USE_CUDNN=1 \
  && export TORCH_CUDA_ARCH_LIST="Hopper" \
  && export _GLIBCXX_USE_CXX11_ABI=1 \
  && git clone --branch v0.6.0 --recursive https://github.com/pytorch/torchrec \
  && cd torchrec \
  && python setup.py bdist_wheel \
  && cd dist \
  && python3 -m pip install torchrec-0.6.0-py3-none-any.whl

RUN pip install -vvv --no-cache-dir \
  black \
  iopath \
  pyre-extensions \
  usort