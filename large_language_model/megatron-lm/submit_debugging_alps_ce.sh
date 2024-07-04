#!/bin/bash

i=0 # i is the batch size per data-parallel unit
ENABLE_DEBUGGING=1 DEBUG_RANK=7 GBS=$((4*2**i)) USE_BF16=true EXTERNAL_GBS=$((4*2**i)) sbatch --nodes=$((16*2**i)) run_gpt3_alps_ce.sh ./logs /capstor/scratch/cscs/dealmeih/ds/mlperf/data/megatron-lm/preprocessed_c4_spm none
