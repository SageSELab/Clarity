'''
    Script to generate BLEU scores on a directory of im2txt checkpoints,
    producing two csv files total:
    
    one csv file contains BLEU scores for each checkpoint with prediction on the validation set
    the other csv file contains BLEU scores for each checkpoint with prediction on the test set


'''

'''

 To obtain BLEU scores for a set of predicted captions, two json files
 are needed

 1) ground_truth.json: A JSON with a list of image ids with their ground truth captions 
    from a given data partition (either validation or test)
 2) predictions.json: A JSON with a list of image ids with their predicted captions from a 
    a given data partition (either validation or test)
    
 For instance, one could put all of the images with their captions from
 the validation set into ground_truth.json, then generate predictions
 on all of these images from the same data partition 
 (one prediction caption per image) and put these predictions into 
 predictions.json. Finally, one would run the coco evaluation script
 with these two jsons as input to receive BLEU scores.


 ground_truth.json resembles the following (note: entries with the same
 image_id correspond to the fact that each image can have multiple captions,
 as shown below). Note also that each entry has a unique "id".


    {
        "annotations": [
            {
                "caption": "A bicycle replica with a clock as the front wheel.",
                "id": 37,
                "image_id": 203564
            },
            {
                "caption": "A black Honda motorcycle parked in front of a garage.",
                "id": 38,
                "image_id": 179765
            },
            {
                "caption": "A black Honda motorcycle parked in front of a garage.",
                "id": 39,
                "image_id": 179765
            }
        ]
    }


 predictions.json resembles the following (note that the "image_id" tags
 in each entry corresponds to the "image_id" tags in ground_truth.json 
 so that prediction captions can be aligned with ground truth captions).
 
    [
        {
            "caption": "black and white photo of a man standing in front of a building",
            "image_id": 404464
        },
        {
            "caption": "group of people are on the side of a snowy field",
            "image_id": 380932
        },
        {
            "caption": "train traveling down a train station",
            "image_id": 565778
        },
    ]

'''

# write out json
'''
with open("test.json", "w") as json_file:
    
    json_out = {"test":3}
    json.dump(json_out, json_file)
'''

# read in json

'''
with open("test.json", "r") as json_file:
    json_in = json.load(json_file)
    print(json_in)
'''

    
#@author Ali Yachnes ayachnes@email.wm.edu

import json
import sys
import os
import hashlib # for hashing im2txt ckpts
import time
from im2txt import run_inference

# custom sort function to sort by checkpoint number
def sort_by_ckpt_num(elem):
    return elem[1]

# coco_caption libraries
sys.path.append('./coco_caption')
sys.path.append('./coco_caption/pycocotools')
sys.path.append('./coco_caption/pycocoevalcap')
sys.path.append('./coco_caption/pycocoevalcap/bleu')
sys.path.append('./coco_caption/pycocoevalcap/cider')
sys.path.append('./coco_caption/pycocoevalcap/meteor')
sys.path.append('./coco_caption/pycocoevalcap/rouge')
sys.path.append('./coco_caption/pycocoevalcap/spice')
sys.path.append('./coco_caption/pycocoevalcap/tokenizer')
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab

#pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


# common command:
# export CUDA_VISIBLE_DEVICES="1" && python generate-BLEU.py models/train/ combined
# export CUDA_VISIBLE_DEVICES="2" && python generate-BLEU.py models/train/ high

# lo2txtIN
# export CUDA_VISIBLE_DEVICES="1" && python generate-BLEU.py models/train/ low

# lo2txtRC, lo2txtGP
# export CUDA_VISIBLE_DEVICES="2" && python generate-BLEU.py models/train/ low


if len(sys.argv) < 3:
    print("\nRuns predictions on a given directory of im2txt checkpoints, outputting jsons of predictions:")
    print("into a new directory")
    print("\nNOTE: this must be run within an im2txt directory with all of the im2txt scripts on a machine")
    print("that can run im2txt, otherwise prediction will not be possible (vocab file needed, tf shards needed)")
    print("\nusage: python3 " + __file__ + " <directory with im2txt checkpoint files> <whether these models were trained on high, low, combined>")
    print("\nex: python3 " + __file__ + " ./models/train high\n")
    exit()


# time this script started
start_time = time.time()

# relative path to the vocab file
VOCAB_PATH = "./data/Clarity/word_counts.txt"

IMAGE_FILE_PATTERN = "./data/Clarity/raw-data/%s/image_id_*"

if not os.path.isfile(VOCAB_PATH):
    print("Error: vocab file '" + VOCAB_PATH + "' is not a file")
    exit()
    


# size of chunks to use when hashing im2txt ckpts
BUFFER_SIZE = 100000000 # 100 MB chunks #65536 * 10000 # 64 KB
        
        
CKPT_DIR = sys.argv[1]
model_type = sys.argv[2]
ext = ".data-00000-of-00001" # extension for im2txt ckpts


if not os.path.isdir(CKPT_DIR):
    print("Error: invalid input directory '" + CKPT_DIR + "'")
    exit()

if model_type not in ["high","low","combined"]:
    print("Error: invalid model type '" + model_type + "'")
    exit()
    
print("Generating predictions on " + model_type.upper())

# training ID (configuration) for im2txt model; i.e. bo2txtIN which is
# (both) (im2txt) (ImageNet)

training_ID = (os.path.abspath(".")).split("/")[-1].strip("/")

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
PREDICTION_VAL_PATH = os.path.join(PREDICTIONS_DIR, "preds_" + training_ID + "_val_%s_%s.json")

# path to a validation predictions file written by each checkpoint (test set)
# note: the first %s is the checkpoint's number, the second %s is the checkpoints md5
PREDICTION_TEST_PATH = os.path.join(PREDICTIONS_DIR, "preds_" + training_ID + "_test_%s_%s.json")


# dictionary with each checkpoint file, its hash, and its BLEU scores
# on the test and validation sets (csv files are written from this data structure)
ckpt_files = {}

i = 0

# for each file in CKPT_DIR

# every file in the checkpoint directory
ALL_CKPT_FILES = []


# for every file in the ckpt directory, only pull out the actual checkpoint data files
for f in os.listdir(CKPT_DIR):
    
    ext_index = f.find(".data-00000-of-00001")
    
    if ext_index != -1:
        
        ckpt_num = int(f[(len("model.ckpt-")):ext_index])
        ALL_CKPT_FILES.append((f,ckpt_num))
    
# sort these by decreasing number
ALL_CKPT_FILES.sort(reverse=True, key=sort_by_ckpt_num)



# only a quarter of checkpoint files, sorted
# (note: includes the last element of ALL_CKPT_FILES by default
# (and will not include the first element (ckpt 0) )
QUARTER_CKPT_FILES = []

for i in range(0, len(ALL_CKPT_FILES), 4):
    if i != len(ALL_CKPT_FILES)-1: # if it isn't checkpoint 0 (since ALL_CKPT_FILES is reverse sorted)
        QUARTER_CKPT_FILES.append(ALL_CKPT_FILES[i])

QUARTER_CKPT_FILES.sort(reverse=True, key=sort_by_ckpt_num)

#print(QUARTER_CKPT_FILES)
#print(len(QUARTER_CKPT_FILES))


#print(QUARTER_CKPT_FILES)
#print(len(QUARTER_CKPT_FILES))

# go through 1/4 of the ckpt files (all of QUARTER_CKPT_FILES)
for elem in QUARTER_CKPT_FILES:
    b_time = time.time() # begin time for this checkpoint
    # f is the filename of the checkpoint
    f = elem[0]
    
    # if the file is an im2txt checkpoint
    
        
    # absolute path of the file
    abs_path = os.path.abspath(os.path.join(CKPT_DIR,f))
        # get the ckpt's hash

    md5_hash = hashlib.md5()
    with open(os.path.join(CKPT_DIR,f), "rb") as to_hash:
        while True:
            data = to_hash.read(BUFFER_SIZE)
                                    
            if not data: # if we're at the end of the file
                break
                                    
            md5_hash.update(data)
        
    md5_hex = md5_hash.hexdigest()
    #print(f + ": " + md5_hex)
        

    # trimmed path is model.ckpt-25468 (trims off .data-00000-of-00001)
    # needed for inferencing
        
        
    trimmed_path = abs_path[0:abs_path.find(ext)]
        
    # print(trimmed_path)
        
    # inference off of ckpt (validation set) and print the predictions

    print("Generating test predictions for %s ..." % (f) )
    
    #val_preds = run_inference.inference_on_ckpt(trimmed_path, VOCAB_PATH, (IMAGE_FILE_PATTERN%"val"))

    # inference off of ckpt (test set) and print the predictions
    test_preds = run_inference.inference_on_ckpt(trimmed_path, VOCAB_PATH, (IMAGE_FILE_PATTERN%"test"))
    
    print("Generated test predictions for %s in %s seconds" % (f, str(time.time()-b_time)))
    
    # fill in the filename with (ckpt_number, md5_hex)
    #val_pred_file = PREDICTION_VAL_PATH%(str(elem[1]), md5_hex)
    
    # fill in the filename with (ckpt_number, md5_hex)
    test_pred_file = PREDICTION_TEST_PATH%(str(elem[1]), md5_hex)
    
    '''
    with open(val_pred_file, "w") as outfile:
        json.dump(val_preds, outfile)
        print("Wrote " + outfile.name)
    '''
    
    with open(test_pred_file, "w") as outfile:
        json.dump(test_preds, outfile)
        print("Wrote " + outfile.name)
        
   
    print("\n\n")


elapsed_time = time.time() - start_time
print("\n\nFinished evaluating " + str(len(QUARTER_CKPT_FILES)) + " checkpoints in " + str(int(elapsed_time*100)/100.0) + " seconds") 
