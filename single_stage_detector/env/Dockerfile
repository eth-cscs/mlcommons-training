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

FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libcurl4 \
 && rm -rf /var/lib/apt/lists/*

# # Install python dependencies
# COPY requirements.txt /
# # Cython needs to be present before attempting to install pycocotools
# RUN pip install --no-cache-dir Cython
RUN pip install --no-cache-dir \
       scikit-image>=0.15.0 \
       ujson>=4.0.2 \
       matplotlib>=3.5.1 \
       pycocotools>=2.0.4 \
       git+https://github.com/mlcommons/logging.git@1.1.0-rc4 \
       opencv-python-headless==4.7.0.72 \
       fiftyone==0.15.1 \
       mlperf-compliance
# opencv-python-headless version chosen for compatibility with NGC's preinstalled opencv
#-r /requirements.txt

# # Copy code
# COPY . /workspace/single_stage_detector

# # Set working directory
# WORKDIR /workspace/single_stage_detector/ssd
