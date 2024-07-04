#!/bin/bash -l

#SBATCH -J mlperf-imagenet
#SBATCH --time 24:00:00
#SBATCH --output logs/slurm-%x.%j.out

set -euxo pipefail

cd /capstor/scratch/cscs/dealmeih/ds/mlperf/data/image_classification
srun -u unzip imagenet-object-localization-challenge.zip