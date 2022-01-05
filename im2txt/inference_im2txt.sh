#!/bin/bash

#Script to evaluate an im2txt model. Takes and optional parameter of IMAGE_FILE. If not 
#provided the default will be used.

# Directory containing preprocessed Clarity data.
VOCAB_FILE="/scratch/gwpurn/Clarity/im2txt/data/Clarity/word_counts.txt"

# Directory to save the model.
CHECKPOINT_PATH="/scratch/gwpurn/Clarity/im2txt/models/inception_v3_no_fine_tuning/combined/train"


IMAGE_FILE=${1:-/scratch/gwpurn/Clarity/im2txt/data/Clarity/raw-data/val/image_id_*}

export CUDA_VISIBLE_DEVICES=""

cd research/im2txt
bazel build -c opt //im2txt:run_inference

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Run inference to generate captions.
bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}
