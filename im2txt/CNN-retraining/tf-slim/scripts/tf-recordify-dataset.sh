# script to prepare a dataset in the tfrecord format
# prequisite: DATA_DIR must contain 'train', 'test', and 'val' directories corresponding to the split of the dataset you're trying to train on

DATA_DIR=../dataset/

TF_SLIM_DIR=../slim/

python ${TF_SLIM_DIR}/download_and_convert_data.py \
    --dataset_name=clarity \
    --dataset_dir="${DATA_DIR}"
