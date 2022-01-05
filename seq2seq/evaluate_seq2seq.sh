DATA_PATH="${HOME}/gwpurnell/Clarity/seq2seq/data/type/combined"
DEV_TARGETS_REF="${DATA_PATH}/val/caption.csv"
MODEL_DIR="${HOME}/gwpurnell/Clarity/seq2seq/models/type/combined/"
PRED_DIR="${MODEL_DIR}/pred"

./bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt
