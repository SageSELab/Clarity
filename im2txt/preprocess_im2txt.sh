#!/bin/bash
#This script must be run from the Clarity/im2txt directory

if [[ $# -eq 0 ]]; then
  echo "Must enter the Input directory, Ouput directory, and caption level ('low', 'high', or 'combined')."
  exit 1
fi

OUTPUT_DIR="${1%/}"
BASE_DIR="${2%/}"
CAPTION_TYPE="${3%/}"
CURRENT_DIR=$(pwd)

BUILD_SCRIPT="${CURRENT_DIR}/im2txt/data/build_mscoco_data.py"
python "${BUILD_SCRIPT}" \
  --train_image_dir="${BASE_DIR}/train" \
  --val_image_dir="${BASE_DIR}/val" \
  --test_image_dir="${BASE_DIR}/test" \
  --train_captions_file="${BASE_DIR}/captions/${CAPTION_TYPE}/captions_train.json" \
  --val_captions_file="${BASE_DIR}/captions/${CAPTION_TYPE}/captions_val.json" \
  --test_captions_file="${BASE_DIR}/captions/${CAPTION_TYPE}/captions_test.json" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" \
