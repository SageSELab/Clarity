#!/bin/bash


OUTPUT_DIR="${1%/}"
BASE_DIR="${OUTPUT_DIR}/raw-data"

CURRENT_DIR=$(pwd)
BUILD_SCRIPT="${CURRENT_DIR}/build_mscoco_data"
"${BUILD_SCRIPT}" \
  --train_image_dir="${BASE_DIR}/train" \
  --val_image_dir="${BASE_DIR}/val" \
  --test_image_dir="${BASE_DIR}/test" \
  --train_captions_file="${BASE_DIR}/annotations/captions_train.json" \
  --val_captions_file="${BASE_DIR}/annotations/captions_val.json" \
  --test_captions_file="${BASE_DIR}/annotations/captions_test.json" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" \