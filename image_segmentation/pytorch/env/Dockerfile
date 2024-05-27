FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check \
  git+https://github.com/NVIDIA/dllogger \
  https://github.com/mlcommons/logging/archive/refs/tags/1.1.0-rc4.zip \
  nibabel


#RUN pip uninstall -y apex; pip uninstall -y apex; git clone --branch seryilmaz/fused_dropout_softmax  https://github.com/seryilmaz/apex.git; cd apex;  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--xentropy" --global-option="--deprecated_fused_adam" --global-option="--deprecated_fused_lamb" --global-option="--fast_multihead_attn" .
