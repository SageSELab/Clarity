#script to train InceptionV3 from scratch

DATASET_DIR=../dataset/  # directory containing the dataset as a list of tfrecords (as well as test, train, val folders)
CHECKPOINT_DIR=../ckpts/      # directory to hold written checkpoints
TF_SLIM_DIR=../slim/     # directory containing slim code and files

# save_interval_secs is how often the model saves checkpoints
# save_summaries_secs is how often the model saves summaries used by tensorboard

python ${TF_SLIM_DIR}/train_image_classifier.py \
    --train_dir=${CHECKPOINT_DIR} \
    --dataset_name=clarity \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3 \
    --save_interval_secs=420 \
    --save_summaries_secs=120 \
    --batch_size=64
