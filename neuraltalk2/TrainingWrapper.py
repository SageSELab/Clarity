# python script to start training neuraltalk2
# this deprecates the java based training wrapper

import os
import sys
import prepro as prepro_script# i.e. prepro.py
import time # for getting a timestamp for training
from subprocess import call # to call the lua train file with torch


if len(os.sys.argv) < 3:
    print("\nTraining wrapper for neuraltalk2 (replaces the old java based training wrapper).\nThis wrapper runs prepro.py to create an .h5 file if needed, and it then begins training.\n")
    print("usage: python2.7 " + __file__ + " <preprocessing type (high, low, or both)> <machine (bg9 or hudson> <optional: gpuid>")
    print("ex: python2.7 " + __file__ + " both bg9")
    exit()

prepro = os.sys.argv[1].lower()

if (prepro != "high" and prepro != "low" and prepro != "both"):
    print("Error: invalid preprocessing type! Valid types are 'high', 'low', and 'both'.")
    exit()
    
print("Got preprocessing type '" + prepro + "'.")

machine = os.sys.argv[2].lower().strip()

if (machine != "hudson" and machine != "bg9"):
    print("Error: invalid machine: '" + machine + "'")
    exit()
    
print("Got machine: " + machine)

################################  Define constants  ################################  

WORD_COUNT_THRESHOLD = 4 # number of times a word must appear for it to be added to the vocabulary

MAX_CAPTION_LENGTH = 30 # maximum length of an individual caption in words; captions longer than this get clipped; this is set to an arbitrarily large number since clipping is already done in preprocessing beforehand


CLARITY_JPG_PATH = ""
torch_path = ""

if machine == "bg9":
    CLARITY_JPG_PATH = "/scratch/ayachnes/Clarity-Data/ClarityJpegs/"
elif machine == "hudson":
    CLARITY_JPG_PATH = "/home/semeru/ClarityJpegs/"
    
if machine == "bg9":
    torch_path = "/opt/torch/install/bin"
elif machine == "hudson":
    torch_path = ""

cnn_proto = "" # path to .prototxt file (unused for now)

cnn_model = "" # path to .caffemodel file (unused for now)

gpuid = -1 #gpuid is an integer that is set based on the preprocesstype and the machine

if len(os.sys.argv) == 4:
    gpuid = int(os.sys.argv[3])
    print("Got GPU " + str(gpuid))
else:

    if machine == "bg9":
        if prepro == "low":
            gpuid = 0
        elif prepro == "high":
            gpuid = 1
        elif prepro == "both":
            gpuid = 2
    elif machine == "hudson":
        gpuid = 0



# hyperparameters for the model
# note that these were all set here to match the hyperparameters from Michael's thesis

# chart from Michael's thesis:
'''
        RNN Size                 256
        Input Encoding Size      256
        LM Drop Probability      0.6
        Optimization Technique   adam
        LM Learning Rate         .0006
        LM Decay Start           7000
        LM Decay Every           5000
        CNN Learn Rate           0.000015
'''

language_eval = 0 # whether to calculate BLEU scores during training

rnn_size = 512 #  number of hidden nodes in each layer of the RNN

input_encoding_size = 256 #  the encoding size of each token in the vocabulary and the image

max_iters = 500000 # the maximum number of iterations for the model to run; -1 for unlimited training

drop_prob_lm = 0.5 #0.7 # strength of dropout in the Language Model RNN (reduces overfitting); on the interval 0 to 1

finetune_cnn_after = -1 # After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)

optim = "sgd" # the gradient descent optimization algorithm to use for the language model; there are six choices: "rmsprop", "sgd", "sgdm", "sgdmom", "adagrad", and "adam"

learning_rate = 0.0004 #  the learning rate for the language model

learning_rate_decay_start = 10000 #-1 # at what iteration to start decaying learning rate? (-1 = dont)

learning_rate_decay_every = 5000 #50000 # every how many iterations thereafter to drop LR by half?

cnn_optim = "sgd" # optim for CNN

start_from = "" #"checkpoint/" + "model-1536692960-low.t7" # checkpoint to start from for training

finetune_cnn_after = -1 # after which iteration should CNN finetuning occur

cnn_learning_rate = 0.000015 #0.00001 #  learning rate for the CNN

cnn_weight_decay = 0 # L2 weight decay just for the CNN

save_checkpoint_every = 1500 # how often to save checkpoints (in number of iterations)

beam_size = 3 # beam size of 2 - 3 typically performs well (according to karpathy)



################################ End of constants  ################################  










    
output_json = ("split-" + prepro + ".json")
output_h5 = "data-" + prepro + ".h5"

needs_preprocessing = ( (not os.path.isfile(output_json)) or (not os.path.isfile(output_h5)) )# whether we need to do preprocessing on the given preprocessing type

    
if needs_preprocessing:
        
    input_json = os.path.join("inputs",("data-"+prepro+".json")) # json containing all files w/ their captions and their split, used as input for prepro.py
        
    if not os.path.isfile(input_json):
        print("Error: could not preprocess because " + input_json + " does not exist.")
        exit()
            
    
    ref_path_json = "ref-path-" + prepro + ".json"
        
    # dictionary mapping arguments to their values; needed for prepro.main
    params = {"input_json" : input_json, "output_json" : output_json, "output_h5" : output_h5,
              "max_length" : MAX_CAPTION_LENGTH, "images_root" : CLARITY_JPG_PATH, 
              "word_count_threshold" : WORD_COUNT_THRESHOLD, "ref_path_json" : ref_path_json} 

    
    # run the main method of prepro to do UNK magic and write the split json and the h5 file
    
    print("Running prepro.py with params = " + str(params))
    prepro_script.main(params)
    
    print("\n\nDone with preprocessing! Now running the training script.")


# At this point, preprocessing is done, so begin training


timestamp = str(int(time.time()))

job_id = timestamp + "-" + prepro #id for the training session (ex: 152871293-low)




# list of arguments for train.lua ; there are more arguments that can be passed in, but these are the ones that were passed in with the old training wrapper (see train.lua)
args = [("-preprotype " + prepro),
        ("-input_h5 " + output_h5), 
        ("-input_json " + output_json), 
        ("-id " + job_id), 
        ("-gpuid " + str(gpuid)), 
        ("-language_eval " + str(language_eval)),
        ("-rnn_size " + str(rnn_size)), 
        ("-input_encoding_size " + str(input_encoding_size)),
        ("-max_iters " + str(max_iters)),
        ("-drop_prob_lm " + str(drop_prob_lm)),
        ("-finetune_cnn_after " + str(finetune_cnn_after)),
        ("-optim " + optim),
        ("-learning_rate " + str(learning_rate)),
        ("-learning_rate_decay_start " + str(learning_rate_decay_start)),
        ("-learning_rate_decay_every " + str(learning_rate_decay_every)),
        ("-cnn_learning_rate " + str(cnn_learning_rate)),
        ("-cnn_weight_decay " + str(cnn_weight_decay)),
        ("-save_checkpoint_every " + str(save_checkpoint_every)),
        ("-beam_size " + str(beam_size)),
        ("-cnn_optim " + str(cnn_optim)),
        ( ("-start_from " + str(start_from)) if (start_from != "") else (""))]

cmd = "source activate python2; " + ( (os.path.join(torch_path, "th")) + " train.lua " + " ".join(args) )


# Now that the command has been built, execute the command

os.system("clear")

print(cmd)

os.system(cmd)

os.system("source deactivate") # deactivate the created virtual environment at the very end












