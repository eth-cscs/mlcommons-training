#!/bin/bash

for i in $(seq 0 4); do # i is the batch size per data-parallel unit
	GBS=$((4*2**i)) USE_BF16=true EXTERNAL_GBS=$((4*2**i)) sbatch --nodes=$((8*2**i)) run_gpt3.sh ./logs /mchstor2/scratch/cscs/lukasd/mlperf/data/megatron-lm/preprocessed_c4_spm none;
done
