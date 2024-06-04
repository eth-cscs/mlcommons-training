#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

display_help() {
   echo "Three command line arguments are required."
   echo "Example usage:"
   echo "bash process_Criteo_1TB_Click_Logs_dataset.sh \\"
   echo "./criteo_1tb/raw_input_dataset_dir \\"
   echo "./criteo_1tb/temp_intermediate_files_dir \\"
   echo "./criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir"
   exit 1
}

[ -z "$1" ] && display_help
[ -z "$2" ] && display_help
[ -z "$3" ] && display_help

# Input directory containing the raw Criteo 1TB Click Logs dataset files in tsv format.
# The 24 dataset filenames in the directory should be day_{0..23} with no .tsv extension.
raw_tsv_criteo_files_dir=$(readlink -m "$1")

# Directory to store temporary intermediate output files created by preprocessing steps 1 and 2.
temp_files_dir=$(readlink -m "$2")

# Directory to store temporary intermediate output files created by preprocessing step 1.
step_1_output_dir="$temp_files_dir/temp_output_of_step_1"

# Directory to store temporary intermediate output files created by preprocessing step 2.
step_2_output_dir="$temp_files_dir/temp_output_of_step_2"

# Directory to store the final preprocessed Criteo 1TB Click Logs dataset.
step_3_output_dir=$(readlink -m "$3")

# Step 2. Convert all sparse indices in day_{0..23}_sparse.npy to contiguous indices and save the output.
# The output filenames are day_{0..23}_sparse_contig_freq.npy
mkdir -p "$step_2_output_dir"
date
python -m data.scripts.contiguous_preproc_criteo --input_dir "$step_1_output_dir" --output_dir "$step_2_output_dir" --frequency_threshold 0 --step 'b' || exit
