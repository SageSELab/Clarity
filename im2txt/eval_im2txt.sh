#!/bin/bash

#Script to evaluate an im2txt model. 

# Directory containing preprocessed Clarity data.
VALIDATION_DATA_DIR="/scratch/Clarity/im2txt/data/tf-data/low"

# Directory to save the model.
MODEL_DIR="/scratch/Clarity/im2txt/models/inception_v3_no_fine_tuning/low-level"

export CUDA_VISIBLE_DEVICES=""

bazel-bin/im2txt/evaluate \
  --input_file_pattern="${VALIDATION_DATA_DIR}/val-?????-of-00004" \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --eval_dir="${MODEL_DIR}/eval"
