'''
    Script to generate BLEU scores on a directory of neuraltalk2 checkpoints.
'''
    
#@author Ali Yachnes ayachnes@email.wm.edu

import json
import sys
import os
import hashlib # for hashing ckpts
import time

# custom sort function to sort by checkpoint number
def sort_by_ckpt_num(elem):
    return elem[1]

# nfb
# python generate-predictions-ntk2.py /home/scratch/ayachnes/NTK2-CKPTS/no-finetune/both/

# nfh
# python generate-predictions-ntk2.py /home/scratch/ayachnes/NTK2-CKPTS/no-finetune/high/

# nfl
# python generate-predictions-ntk2.py /home/scratch/ayachnes/NTK2-CKPTS/no-finetune/low/


##########################################################################################


# fb
# python generate-predictions-ntk2.py /home/scratch/ayachnes/NTK2-CKPTS/finetune/both/

# fh
# python generate-predictions-ntk2.py /home/scratch/ayachnes/NTK2-CKPTS/finetune/high/

# fl
# python generate-predictions-ntk2.py /home/scratch/ayachnes/NTK2-CKPTS/finetune/low/



if len(sys.argv) < 2:
    print("\nRuns BLEU score evaluation on a given directory of neuraltalk2 checkpoints,")
    print("outputting a json of 3 predictions per test image per checkpoint")
    print("NOTE: ALL checkpoints in the given directory are evaluated")
    print("\nusage: python3 " + __file__ + " <directory with ntk2 checkpoint files>")
    print("ex: python3 " + __file__ + " /home/scratch/ayachnes/NTK2-CKPTS/finetune/high/\n")
    exit()


# time this script started
start_time = time.time()


# size of chunks to use when hashing ntk2 ckpts
BUFFER_SIZE = 1000000000 # 1000 MB chunks for hashing files
        
        
CKPT_DIR = sys.argv[1]

# remove trailing / if it exists on CKPT_DIR
if CKPT_DIR[-1] == "/":
    CKPT_DIR = CKPT_DIR[0:-1]

if not os.path.isdir(CKPT_DIR):
    print("Error: invalid input directory '" + CKPT_DIR + "'")
    exit()

    
print("Generating predictions on ckpts in '%s'"%CKPT_DIR)

# training ID used to create the predictions directory
# i.e. turns finetune/high into finetune-high
indiv_dirs = CKPT_DIR.split("/")

assert(len(indiv_dirs) >= 2)

training_ID = indiv_dirs[len(indiv_dirs)-2] + "-" + indiv_dirs[len(indiv_dirs)-1]

# directory to hold predictions

PREDICTIONS_DIR = "./predictions-" + training_ID

# if predictions directory does not exist
if (not os.path.isdir(PREDICTIONS_DIR)) and (not os.path.isfile(PREDICTIONS_DIR)):
    print("\nNote: '%s' does not exist, so creating directory."%PREDICTIONS_DIR)
    os.mkdir(PREDICTIONS_DIR)

# if predictions directory is not empty
elif len(os.listdir(PREDICTIONS_DIR)) != 0:
    print("Error: predictions directory '%s' is not empty; must be empty before prediction."%PREDICTIONS_DIR)
    exit()
    

# path to a validation predictions file written by each checkpoint (validation set)
# note: the first %s is the checkpoint's number, the second %s is the checkpoints md5
PREDICTION_VAL_PATH = os.path.join(PREDICTIONS_DIR, "preds_" + training_ID.replace("-","_") + "_val_%s_%s.json")

# path to a validation predictions file written by each checkpoint (test set)
# note: the first %s is the checkpoint's number, the second %s is the checkpoints md5
PREDICTION_TEST_PATH = os.path.join(PREDICTIONS_DIR, "preds_" + training_ID.replace("-","_") + "_test_%s_%s.json")







# list of all the checkpoint files paierd with their checkpoint number
# i.e. tuples of ("model-1544627444-both-91.t7", 91)
ALL_CKPT_FILES = []


for f in os.listdir(CKPT_DIR):
    
    # if the file is a checkpoint
    if f.find(".t7") != -1:
        
        # extension (i.e. "92.t7")
        ext = f.split("-")[-1]
        ext = ext[0:ext.find(".t7")]
        
        ckpt_num = int(ext)

        ALL_CKPT_FILES.append((f,ckpt_num))

# sort these by decreasing number
ALL_CKPT_FILES.sort(reverse=True, key=sort_by_ckpt_num)

            
#print(ALL_CKPT_FILES)
#print(len(ALL_CKPT_FILES))






for elem in ALL_CKPT_FILES:
    b_time = time.time() # begin time for this checkpoint
    # ckpt is the filename of the checkpoint
    ckpt = elem[0]
    
    # absolute path of the file
    abs_path = os.path.abspath(os.path.join(CKPT_DIR,ckpt))
        # get the ckpt's hash

    md5_hash = hashlib.md5()
    with open(os.path.join(CKPT_DIR,ckpt), "rb") as to_hash:
        while True:
            data = to_hash.read(BUFFER_SIZE)
                                    
            if not data: # if we're at the end of the file
                break
                                    
            md5_hash.update(data)
        
    md5_hex = md5_hash.hexdigest()


    print("Generating test predictions for %s ..." % (ckpt) )

    '''
    [
        {
            "image_id": "0009628",
            "captions": [
                "in the top left hand corner is a back button",
                "in the top left hand corner is a back button for the user to click on",
                "at the bottom of the screen is a button to start the game"
            ]
        }
    ]
    '''
    
    # fill in the filename with (ckpt_number, md5_hex)
    test_pred_file = PREDICTION_TEST_PATH%(str(elem[1]), md5_hex)
    
    args = [("-model " + os.path.join(CKPT_DIR,ckpt)),
            ("-batch_size 1"),
            ("-num_images -1"), # evaluate all the images in the test set
            ("-language_eval 0"), # don't calculate BLEU scores
            ("-dump_images 0"), # don't dump images into a vis folder
            ("-dump_json 1"), # dump the json with predictions into a given folder
            ("-dump_json_path " + test_pred_file), # path to the prediction json written out (including filename)
            ("-dump_path 1"), # write filenames with predictions in json
            ("-sample_max 1"), # because we want to do a beam search
            ("-beam_size 3"), # beam size in the beam search
            ("-temperature 1"),  # temperature when sampling
            ("-split test"), # evaluate on the test set
            ("-gpuid 2") # use GPU 2 for evaluation
            ]

    cmd = "/opt/torch/install/bin/th eval.lua " + " ".join(args) 


    # Now that the command has been built, execute the command

    print(cmd)
    os.system(cmd)

    test_preds = []
    
    print("Generated test predictions for %s in %s seconds" % (ckpt, str(time.time()-b_time)))



    print("\n\n")




elapsed_time = time.time() - start_time
print("\n\nFinished evaluating " + str(len(ALL_CKPT_FILES)) + " checkpoints in " + str(int(elapsed_time*100)/100.0) + " seconds") 
