# Show, Attend and Tell

The source code for the show, attend and tell model is adapted from [*here*](https://github.com/coldmanck/show-attend-and-tell). The directory structure will be the same as the adapted source code, except we need to update the location of the images and the caption files. The directory of those files and several other hyperparameters can be updated from the config.py file.

## Train
To train the model, first we need to update different hyperparameters and run this command:

python main.py --phase=train \
    --load_cnn \
    --cnn_model_file='./vgg16_no_fc.npy'\
    --train_cnn

If we want to resume the training from a particular checkpoint, we can run the following command:

python main.py --phase=train \
    --load \
    --model_file='./models/xxxxxx.npy'\
    --train_cnn


## Evaluation
To generate results for different checkpoints (e.g., checkpoint x), we can run the follwing command:

python main.py --phase=eval \
	--model_file='./models/combined-captions-100K/$x$.npy' \
    --beam_size=3 > checkpoint-results/combined-captions-check-$x$-results.txt
