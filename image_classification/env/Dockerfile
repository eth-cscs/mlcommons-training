FROM nvcr.io/nvidia/tensorflow:24.04-tf2-py3

RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd

RUN python -m pip install --no-cache-dir -U pip

# Install MLPerf-logging
RUN pip install --no-cache-dir "git+https://github.com/mlcommons/logging.git"