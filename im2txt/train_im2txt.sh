#!/bin/bash

#Script to train an im2txt model. The script takes one parameter - the number of iterations

if [[ $# -eq 0 || -n ${1//[0-9]/} ]]; then
  echo "Must input an integer representing the number of steps"
  exit 1
fi

# Directory containing preprocessed Clarity data.
TRAINING_DATA_DIR="/scratch/Clarity/im2txt/data/tf-data/low"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="/scratch/gwpurn/Clarity/im2txt/data/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="/scratch/Clarity/im2txt/models/inception_v3_no_fine_tuning/low-level"

bazel build -c opt //im2txt/...

bazel-bin/im2txt/train \
  --input_file_pattern="${TRAINING_DATA_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps="$1"
