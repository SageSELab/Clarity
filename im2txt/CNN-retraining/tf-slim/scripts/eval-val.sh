#script to evaluate on the validation set given a checkpoint

DATASET_DIR=../dataset  # directory containing the dataset as a list of tfrecords (as well as test, train, val folders)
CHECKPOINT_DIR=../ckpts/      # directory to hold written checkpoints
TF_SLIM_DIR=../slim/     # directory containing slim code and files

if [ "$1" != "" ]; then

CHECKPOINT_FILE=${CHECKPOINT_DIR}/$1  # Example

python ${TF_SLIM_DIR}/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=clarity \
    --dataset_split_name=validation \
    --model_name=inception_v3
else
    echo "usage: eval-val.sh <checkpoint_name>"
fi 
