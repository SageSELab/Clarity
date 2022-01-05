#script to train an InceptionV3 model by fine tuning


#script to evaluate on the test set given a checkpoint
  
DATASET_DIR=../dataset/  # directory containing the dataset as a list of tfrecords (as well as test, train, val folders)
CHECKPOINT_DIR=../ckpts/      # directory to hold written checkpoints
TF_SLIM_DIR=../slim/     # directory containing slim code and files

if [ "$1" != "" ]; then

CHECKPOINT_PATH=$1  # Example

python ${TF_SLIM_DIR}/train_image_classifier.py \
    --train_dir=${CHECKPOINT_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=clarity \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --save_interval_secs=420 \
    --save_summaries_secs=120 \
    --batch_size=64

else
    echo "usage: train-finetune.sh <pretrained inceptionV3 checkpoint file>"
fi 
