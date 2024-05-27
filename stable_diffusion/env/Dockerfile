ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.01-py3
FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive

# apt dependencies
RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
    ffmpeg libsm6 libxext6 clang lld \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd

# Remove the opencv version shipped with the base image
# https://github.com/opencv/opencv-python/issues/884
RUN pip uninstall -y opencv
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/

# install LDM
COPY . /diffusion
RUN cd /diffusion && \
    TRITON_BUILD_WITH_CLANG_LLD=true pip -vvv --no-cache-dir install --no-build-isolation \
    git+https://github.com/openai/triton.git@989adb9a29496c22a36ef82ca69cad5dad536b9c#subdirectory=python && \
    MAX_JOBS=16 pip install -vvv --no-cache-dir -r env/requirements.txt
