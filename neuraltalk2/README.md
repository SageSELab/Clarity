
# NeuralTalk2 for Android App Screenshots

This repository contains the code and most of the files for the NeuralTalk2-Android project, which allows for the training on low-level, high-level, and combined description-screenshot pairs. Large files not included in this repository include the model checkpoints (as well as their log files and evaluation visualization) for the first two attempts of training the model and the Clarity JPEGs and ReDraw PNGs needed to train the model.


## Repository Overview

Before setting up, here's an overview of the important contents of the repository:

  * `/coco-caption` - a folder containing evaluation scripts from the MS COCO caption repository, used in training of models and evaluation of models to generate bleu scores. In this directory, json files containing the scores of models are written at runtime, though this is just to communicate bleu scores globally between several scripts.

  * `/developer-tools` - a folder containing files with commands needed for training the model, evaluating the model, and performing a random hyperparameter search. I used these files while working on the project, and they are reproduced here for convenience. It also includes python script for graphing the loss of a model from its log file, in `/developer-tools/loss-graphing`. See the README.txt file in this directory for more detailed information of its contents.

* `/hyperparam-search` - a folder containing results from our hyperparameter search. The file `hyperparams.csv` contains a list of all the hyperparameters tuned during the search with a description for each. To run a hyperparameter search, compile the file `/nt2-android-java/src/edu/semeru/android/clarity/pipeline/HyperparamSearcher.java` into an executable jar file and run it with the command found in `/developer-tools/hyperparameter-cmd.txt` **Note: since TrainingWrapper.jar was rewritten in python, soon the hyperparameter search will be too; as such, hyperparameter searching is dysfunctional until then.**

* '/inputs' - a folder to hold `.json` files aggregating file paths with their high/low/combined level captions, depending on the preprocessing type. `prepro.py` reads these files in preprocessing to get the captions and their split.

* `/misc` - a folder containing scripts used by the training script (`train.lua`).

* `/model` - a folder containing a pretrained VGGnet CNN used to initialize the CNN used for the model.

* `/nt2-android-java-deprecated` - a folder containing the **deprecated** eclipse project for NeuralTalk2-Android. Do not touch or use it. It will be removed in the future, since the TrainingWrapper jar has been replaced by a python script and the Hyperparameter search jar soon will be too.

* `/vis` - a folder containing visualization files for the evaluation of a model. When you evaluate a trained model from a .t7 checkpoint, running the evaluation script (evaluate.lua) with the argument `-dump_images 1` will copy all test images into this visualization folder. `/vis/index.html` is a useful html file that organizes all of the test set images into a grid with the model's text predictions overlaid onto each image.

* `convert_checkpoint_gpu_to_cpu.lua` - a script converting gpu checkpoints to cpu checkpoints, so that the model can be evaluated from a cpu (it was never used or tested, since cpu computation is slow).

* `eval.lua` - a script to evaluate a model from a checkpoint. Evaluation consists of passing the test images (which are separated from the other images in preprocessing) through the model to generate captions for each image. This script provides model loss and bleu scores as calculated from the test data, and it may optionally produce a visualization of the images with their corresponding predictions in `/vis`. See `/developer-tools/README.txt` for more information.

* `CNN-retraining` - a directory containing a retraining script and its models for retraining a VGG16 model on the ReDraw cropped dataset and the Google Play categories dataset (transfer learning, fine tuning, and learning from scratch).

## Setup

You need a scratch directory on bg9 to set up this project, since the images alone are more than 11GB.

 1. cd to your scratch directory and clone the Clarity repo:

    `git clone this-repo`

 3. The only folder needed from the Clarity repo is `neuraltalk2`.

 2. Download the full [screenshots](https://doi.org/10.5281/zenodo.5822884) tar to your cloned repo and then extract it:
 
    `tar -xvzf Data.tar.gz`

 3. Before running TrainingWrapper.py, run the following command so that torch can find a required CUDA library:

    `export CUDNN_PATH="/usr/lib/x86_64-linux-gnu/libcudnn.so.7"`

    Additionally, you should append this command to `~/.bashrc`, otherwise you will have to run this command each time you log in to train.


4. To run on bg9, you also need to add this command to your ~/.bashrc:

    `export PATH=/opt/anaconda3/bin:$PATH`

    (run `source ~/.bashrc` or log out then log in for this command to take effect)


5. To run on hudson, you need to activate a virtual environment with h5py installed (currently the virtual environment py27 on hudson works with TrainingWrapper.py). So to run on hudson, do:

`source activate py27`

`python2.7 TrainingWrapper.py high hudson`

# Training

* Run TrainingWrapper.py (in the root of the neuraltalk2 directory) with the command python2.7 TrainingWrapper.py <preprocessing type (high, low, or both)> <machine (bg9 or hudson)>. For testing purposes, it does not matter which command you run (i.e. high, low, or both). 

* After preprocessing (which will take awhile, and will only happen once), the training wrapper will begin training. It  is recommended to train on a screen via the `screen` command for long training sessions.

* If it runs successfully, you should see a new process named `luajit` when you run `nvidia-smi`.

* For a comprehensive list of training arguments, see `train.lua`. Extra arguments can be added to the TrainingWrapper.py script.

* Note that you may train all three types of data (high, low, both) at the same time, since the training wrapper assigns each type to its own gpu. If you wish to train multiple models of the same data type at the same time, simply change the `-gpuid` argument of `train.lua`.

# Preprocessing

There are three forms of preprocessing, in which image-description pairs are formatted into `json` and `h5` files needed for training.

**high**: Images are paired with only their high level descriptions.

**low**: Images are paired with only their low level descriptions

**both**: Images are paired with the combined low level and high level descriptions.

Preprocessing happens automatically by the training wrapper. The training wrapper preprocesses data if and only if the 
`.h5` file in `/` and the `.json` file in `/` are both missing. After preprocessing is performed once on a given data type, it need not be performed again unless more data is added to the dataset or a different testing/validation/training split is created. To force preprocessing to occur, delete the `.json` file from `/` and delete the `.h5` file from `/` and then run the training wrapper.

# Evaluation

Once you train a model and obtain a `.t7` checkpoint file, you can run the evaluation script on it to obtain bleu scores and other metrics on the test data (data that the model has never seen before). The evaluation script also optionally produces a visualization of images combined with their descriptions at `./vis`.

`#(note: to do all test images, do -num_images -1, or if you only want to do N images, do -num_images N. To copy all images over to the ./vis folder as a nice viewable html, do -dump_images 1)`

`#bg9`
`source activate python2; /opt/torch/install/bin/th eval.lua -model ./checkpoint/model.t7 -num_images -1 -dump_images 0 -image_root /scratch/ayachnes/Clarity/Clarity-Data/ClarityJpegs`

`#hudson`
`source activate python2; th eval.lua -model ./checkpoint/model.t7 -num_images -1 -dump_images 0 -image_root <location of Clarity jpegs on hudson>`