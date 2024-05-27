FROM nvcr.io/nvidia/jax:24.04-paxml-py3

RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd

RUN pip install -vvv \
  git+https://github.com/mlcommons/logging.git@f6ee121
