#edit this line to the appropriate path
DATA_PATH="${HOME}/gwpurnell/Clarity/seq2seq/data/type/high_level"

VOCAB_SOURCE_IN="${DATA_PATH}/train/type.tok"
VOCAB_SOURCE_OUT="${DATA_PATH}/train/type.vocab.txt"
VOCAB_TARGET_IN="${DATA_PATH}/train/caption.tok"
VOCAB_TARGET_OUT="${DATA_PATH}/train/caption.vocab.txt"

TRAIN_SOURCES_IN="${DATA_PATH}/train/type.csv"
TRAIN_SOURCES_OUT="${DATA_PATH}/train/type.tok"
TRAIN_TARGETS_IN="${DATA_PATH}/train/caption.csv"
TRAIN_TARGETS_OUT="${DATA_PATH}/train/caption.tok"


python ${HOME}/gwpurnell/Clarity/seq2seq/bin/tools/simple_tokenizer.py ${TRAIN_SOURCES_IN} ${TRAIN_SOURCES_OUT}

python ${HOME}/gwpurnell/Clarity/seq2seq/bin/tools/simple_tokenizer.py $TRAIN_TARGETS_IN $TRAIN_TARGETS_OUT

python ${HOME}/gwpurnell/Clarity/seq2seq/bin/tools/generate_vocab.py\
  --min_frequency 4\
  --max_vocab_size 10000\
  $VOCAB_SOURCE_IN > $VOCAB_SOURCE_OUT

python ${HOME}/gwpurnell/Clarity/seq2seq/bin/tools/generate_vocab.py\
  --min_frequency 4\
  --max_vocab_size 10000\
  $VOCAB_TARGET_IN > $VOCAB_TARGET_OUT

