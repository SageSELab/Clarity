#The arguments to this script are:
# 1.  the model type(type, text. type-text,
#     or type-text-loc) 
#
# 2.  the level(high_level, low_level. or combined)
DATA_PATH="${HOME}/gwpurnell/Clarity/seq2seq/data/$1/$2"

VOCAB_SOURCE="${DATA_PATH}/train/$1.vocab.txt"
VOCAB_TARGET="${DATA_PATH}/train/caption.vocab.txt"
TRAIN_SOURCES="${DATA_PATH}/train/$1.csv"
TRAIN_TARGETS="${DATA_PATH}/train/caption.csv"
DEV_SOURCES="${DATA_PATH}/val/$1.csv"
DEV_TARGETS="${DATA_PATH}/val/caption.csv"

DEV_TARGETS_REF="${DATA_PATH}/val/caption.csv"
TRAIN_STEPS=500000

SAVE_STEPS=2000
EVAL_STEPS=500000
MAX_CHECKPOINTS=999
MODEL_DIR="${HOME}/gwpurnell/Clarity/seq2seq/models/$1/$2/"

python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR\
  --save_checkpoints_steps $SAVE_STEPS\
  --keep_checkpoint_max $MAX_CHECKPOINTS\
  --eval_every_n_steps $EVAL_STEPS
