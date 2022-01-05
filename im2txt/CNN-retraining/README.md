# Setup

1. To retrain an inceptionV3 CNN, first copy the directory `tf-slim` to the machine you want to train on, then make sure you are in a virtual environment with python3.6 where the `tensorflow-gpu` package is installed. On hudson, you do not need to make your own virtual environment, since `source activate george` will get you into an environment with tensorflow. If you need to create a virtual environment with these specifications, run the following:

    `conda create -n myenv`
    
    `conda install tensorflow-gpu`
    
    Then to activate the environment so you can run the CNN retraining and evaluation scripts:
    
    `source activate myenv`
    
    Then to deactivate the environment:
    
    `source deactivate`
    
    **Do not start a screen when in a virtual environment that you want to enter while in the screen, as this usually causes errors when running things in the screen. For instance, always do this when starting a screen **
    
    `(george) semeru@hudson:~/ayachnes/tf-slim/scripts$ source deactivate`
    
    `semeru@hudson:~/ayachnes/tf-slim/scripts$ screen -L -S training`
    
    `semeru@hudson:~/ayachnes/tf-slim/scripts$ source activate george`
    
    `(george) semeru@hudson:~/ayachnes/tf-slim/scripts$ echo "Virtualenv in a screen"`

2. Once you have a virtual environment with `tensorflow-gpu`, copy the `tf-slim` directory to the machine and cd into it. **On hudson, `tf-sim` is located in `/home/semeru/ayachnes/tf-slim/` so you do not need to copy it.

    There are several folders in `tf-slim`. 

    * `ckpts` holds tensorflow checkpoints and tensorboard event files. On hudson, there is a folder called `adam-150k` within `ckpts` that holds several checkpoint files for the first CNN retraining attempt on the ReDraw dataset using adam as the optimizer, and training from scratch. After 150,000 iterations, this attempt was terminated because it seems to have stopped learning. 

    * `scripts` holds scripts that will:
        * `train-scratch.sh` - train from scratch (inceptionV3 initialized with randomized weights)
        
        * `train-finetune.sh` - fine tune (inceptionV3 initialized with imagenet weights)
        
        * `eval-val.sh` - evaluate an inceptionV3 checkpoint on the validation set to print out accuracy and recall
        
        * `eval-test.sh` - evaluate an inceptionV3 checkpoint on the test set to print out accuracy and recall
        * `tf-recordify-dataset.sh` - turn a dataset in the folder `tf-slim/dataset` with `train`, `test`, and `val` folders and turn it into tfrecords needed for training. **This script must be run after moving the train, test, and val folders to the folder `tf-slim/dataset`**.


    * `dataset` holds the dataset to train on.  Before training, you have to put the `train` `test` and `val` folders corresponding to  corresponding to the dataset you want to train on inside `dataset`. After placing the three folders in `dataset`, cd into the `scripts` folder and run `./tf-recordify-dataset.sh`. This script creates tfrecord files in the folder `dataset` and this script must be run before training can start. On hudson, this folder already currently holds the ReDraw dataset so you don't need to modify anything unless you want to retrain on the google play dataset on hudson. See step 3 for obtaining and preparing the dataset.

    * `slim` holds all of the python tf-slim code, which was modified to work with the clarity datasets. The only file in this folder of interest is `train_image_classifier.py`, which is the engine that trains InceptionV3. Read the first ~ 100 lines of this file for a comprehensive list of flags you can pass into it for training (such as learning rate, optimization technique) and either edit the script `train-scratch.sh` or `train-finetune.sh` with any additional flags or modification to flags you want to make.

3. After `tf-slim` is copied, you must prepare the dataset on which you want to train.

    * **Google play: ** Go to `/data/processing-code/python/gp-categories-processing-scripts` in the Clarity repository and upload the tar corresponding to the machine you want to train on. For instance, to train on bg9, upload `Google-Play-Split-bg9.tar.gz` to bg9 and extract it to the `dataset` folder such that the children of `dataset` are `train`, `test`, and `val`. Then run `tf-slim/scripts/tf-recordify-dataset.sh` to turn the data into tf-records.
    * **ReDraw: ** Move the `train`, `test`, and `val` folders on the machine you want to train on to the `dataset` folder. The location of the redraw dataset depends on the machine.
        * **bg9: ** the redraw dataset is located at `/scratch/ayachnes/ReDraw-Dataset` (you have to fix the permissions on it after moving the folders since permissions are currently set to 777).
        * **hudson: ** the redraw dataset is located at `/home/semeru/ayachnes/tf-slim/dataset` and it is already prepared, so you do not need to do anything to train inceptionV3 on the redraw dataset. However, if you want to train inceptionV3 on the google play dataset on hudson, you need to move all of the files in `dataset` to somewhere else temporarily.
        * **tower2: ** the redraw dataset is located at `/home/ayachnes/Clarity-Data/Redraw-Split/`. Its permissions are 777.

        Once you have the `train`, `test`, and `val` folders placed in `dataset`, you must run `tf-slim/scripts/tf-recordify-dataset.sh` to turn the data into tf-records.
        
4. After the dataset is prepared, you can train the inceptionV3 either from scratch or by finetuning. See below.

# Training InceptionV3

1. Run `scripts/train-scratch.sh` to train from scratch and run `scripts/train-finetune.sh` to fine-tune. You can terminate training at any time and it will resume at the last saved checkpoint when you rerun this script. The current save interval is every 7 minutes, but this can be changed (it is the flag `--save_interval_secs`). If you want to modify any flags for training, edit this script. The training script only saves the last five checkpoints, irrespective of the `--save_interval_secs` flag.

# Evaluating InceptionV3 to Get Accuracy

1. Run `scripts/eval-val.sh <checkpoint_name>` to evaluate on the validation set and run `scripts/eval-test.sh <checkpoint_name>` to evaluate on the test set. `checkpoint_name` is the name of a saved checkpoint located in `tf-slim/ckpts/`. For instance, an example run of `eval-val.sh` would be: `eval-val.sh model.ckpt-834` where `834` corresponds to the iteration at which the checkpoint was saved. If the machine you're training on only has one GPU, you must terminate training before running the evaluation script; this is because the training script uses all of the GPU. The output of evaluation will look like the following: 

    `eval/Accuracy[0.5294773523]`
    
    `eval/Recall_5[0.850836238]`

# Using tensorboard while training

To get tensorboard to work while training:

1. ssh into the machine you're training on with the following:

    `ssh -L 16006:127.0.0.1:6006 semeru@rocco.cs.wm.edu -p 6175`

2. Then run `source activate <env>` where `<env>` is the name of the virtual environment with tensorflow-gpu installed.

    

3. In the virtual environment, run tensorboard: 

    `tensorboard --logdir="/home/semeru/ayachnes/tf-slim/ckpts"`

4. Then on your local machine, you can view tensorboard by going to `http://127.0.0.1:16006`.


# Support

If you have any questions, message Ali on slack, or email him at [ayachnes@email.wm.edu](ayachnes@email.wm.edu).

