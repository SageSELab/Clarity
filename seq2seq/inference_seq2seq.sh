#!/bin/bash

DATA_PATH="${HOME}/gwpurnell/Clarity/seq2seq/data/type/combined"

VOCAB_SOURCE="${DATA_PATH}/train/type.vocab.txt"
VOCAB_TARGET="${DATA_PATH}/train/caption.vocab.txt"
TRAIN_SOURCES="${DATA_PATH}/train/type.csv"
TRAIN_TARGETS="${DATA_PATH}/train/caption.csv"
DEV_SOURCES="${DATA_PATH}/val/type.csv"
DEV_TARGETS="${DATA_PATH}/val/caption.csv"
DEV_TARGETS_REF="${DATA_PATH}/val/caption.csv"

TRAIN_STEPS="1000000"

MODEL_DIR="${HOME}/gwpurnell/Clarity/seq2seq/models/type/combined/"
PRED_DIR="${MODEL_DIR}/pred"

python -m bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: False" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt
