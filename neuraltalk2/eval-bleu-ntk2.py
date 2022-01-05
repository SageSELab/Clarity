# @author Ali Yachnes

import sys
import time
import os
import io
sys.path.append('./coco_caption')
sys.path.append('./coco_caption/pycocotools')
sys.path.append('./coco_caption/pycocoevalcap')
sys.path.append('./coco_caption/pycocoevalcap/bleu')
sys.path.append('./coco_caption/pycocoevalcap/cider')
sys.path.append('./coco_caption/pycocoevalcap/meteor')
sys.path.append('./coco_caption/pycocoevalcap/rouge')
sys.path.append('./coco_caption/pycocoevalcap/spice')
sys.path.append('./coco_caption/pycocoevalcap/tokenizer')

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from multiprocessing.dummy import Pool as ThreadPool 
#import matplotlib.pyplot as plt
#import skimage.io as io
#import pylab

#pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


# function to allow sorting of tuples where the second element is a number
def sort_by_ckpt_num(elem):
    return elem[1]

# simple function to compute the average of a list of numbers
# average is computed as a float

def avg(l):
    return float(sum(l))/len(l)

'''
# dictionary of ground truths
gts = {
    "annotations": [
        {
            "caption": "The scissors with black handles are sitting open.",
            "id": 829139,
            "image_id": 565778
        }
    ],
    # dummy values for everything except 'id'
    "images": [
        {
            "date_captured": "???????",
            "file_name": "???????",
            "height": 1000,
            "id": 565778,
            "license": 1,
            "url": "?????",
            "width": 1000
        }
    ],
    
    "info": {
        "contributor": "??????",
        "date_created": "?????",
        "description": "Dataset with one caption/image",
        "url": "????",
        "version": "???",
        "year": 2019
    },
    "licenses": [
        {
            "id": 1,
            "name": "????",
            "url": "?????"
        }
    ],
    "type": "captions"
}
'''

'''
preds = [
            {
                "caption": "train handles traveling open sitting sitting open down a train station",
                "image_id": 565778
            }
        ]
'''

# function that gets bleu scores for a single caption given a 
# gts_by_id dictionary; this returns bleu_1, bleu_2, bleu_3, bleu_4 
# in that order only
# note that image_id is a string
def get_bleu_scores(caption, image_id, gts_by_id):
    # dictionary to be fed into coco caption's code
    gts = {
        "annotations": [
        ],
        # dummy values for everything except 'id'
        "images": [
            {
                "date_captured": "???????",
                "file_name": "???????",
                "height": 1000,
                "id": int(image_id),
                "license": 1,
                "url": "?????",
                "width": 1000
            }
        ],
        
        "info": {
            "contributor": "??????",
            "date_created": "?????",
            "description": "Dataset with one caption/image",
            "url": "????",
            "version": "???",
            "year": 2019
        },
        "licenses": [
            {
                "id": 1,
                "name": "????",
                "url": "?????"
            }
        ],
        "type": "captions"
    }
    
    
    # list of 1 or more ground truth captions; i.e. ["caption A", "caption B"]
    ground_truth_captions = gts_by_id[image_id]
    
    
    
    # append all ground truth captions for this image id to gts["annotations"]
    HARD_CODED_ID = 1000
    for truth_caption in ground_truth_captions:
        
        # new entry to append to ground truth annotations
        new_entry = {}
        new_entry["caption"] = truth_caption
        new_entry["id"] = HARD_CODED_ID
        new_entry["image_id"] = int(image_id)
        
        HARD_CODED_ID += 1
        
        gts["annotations"].append(new_entry)
    
    #print(gts["annotations"])
    
    # dictionary of predictions (only one for now)
    preds = [
                {
                    "caption": caption,
                    "image_id": int(image_id)
                }
            ]

    #print(preds)
    
    
    ### suppress output from coco caption ###
    
    old_stdout = sys.stdout
    #sys.stdout = io.BytesIO()
    
    
    
    # create coco object and cocoRes object
    coco = COCO(annotation_dict=gts)
    cocoRes = coco.loadRes(res_dict=preds)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    #cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    
    
    cocoEval.evaluate()

    #sys.stdout = old_stdout

    #print(cocoEval.eval.items())
    # print output evaluation scores
    
    bleu_1 = 0.0
    bleu_2 = 0.0
    bleu_3 = 0.0
    bleu_4 = 0.0
    
    for metric, score in cocoEval.eval.items():
        
        if metric.lower() == "bleu_1":
            bleu_1 = score
        elif metric.lower() == "bleu_2":
            bleu_2 = score
        elif metric.lower() == "bleu_3":
            bleu_3 = score
        elif metric.lower() == "bleu_4":
            bleu_4 = score
        
        #print('%s: %.3f'%(metric, score))
    
    
    return bleu_1, bleu_2, bleu_3, bleu_4
    


# common command

# python eval-bleu-ntk2.py ./ntk2-predictions/val/predictions-no-finetune-both-val/ combined val
# python eval-bleu-ntk2.py ./ntk2-predictions/val/predictions-finetune-both-val/ combined val


# python eval-bleu-ntk2.py ./ntk2-predictions/test/predictions-no-finetune-both-test/ combined test
# python eval-bleu-ntk2.py ./ntk2-predictions/test/predictions-finetune-both-test/ combined test

# set up file names and paths

if len(sys.argv) < 4:
    print("\nRuns BLEU score evaluation on a given directory of ntk2 prediction jsons")
    print("can calculate BLEU scores on the test set or the validation set")
    print("\nNOTE: this must be run within an ntk2 directory with the coco_caption code on a machine")
    print("that can run coco_caption, otherwise bleu scores will not be possible")
    print("\nNOTE 2: The files in the input prediction directory must have the .json extension to be recognized.")
    print("\nNOTE 3: This script outputs one json file for each prediction json it reads in. The output file is identical to the prediction file except it has a list of 3 sets of Bleu scores in the same JSON object as the list of 3 captions (per image)") 
    print("\nusage: python2 " + __file__ + " <directory with ntk2 prediction jsons> <whether the predictions come from a model trained on high, low, or combined> <whether these jsons are predictions on the val or test set>")
    print("\nex: python2 " + __file__ + " ./predictions low test\n")
    exit()
global PREDS_DIR
PREDS_DIR = sys.argv[1]
model_type = sys.argv[2]
global data_split
data_split = sys.argv[3].lower()

if not os.path.isdir(PREDS_DIR):
    print("Error: invalid predictions directory '" + PREDS_DIR + "'")
    exit()

if model_type not in ["high","low","combined"]:
    print("Error: invalid model type '" + model_type + "'")
    exit()

if data_split not in ["val", "test"]:
    print("Error: invalid data split '" + data_split + "'; valid options are 'val' and 'test'.")
    exit()

# path to the ground truths of the validation split (on the given 'model_type')
GROUND_TRUTH_PATH = "./coco_caption/ground_truth/ntk2/data-%s.json"%model_type


# make sure the validation and test ground truth jsons exist for the given model type
if not os.path.isfile(GROUND_TRUTH_PATH):
    print("Error: missing ground truth file '"+GROUND_TRUTH_PATH+"'")
    exit()
    
    
    
# the ground truth file that needs to be opened for the given data split;
# i.e. the one that's relevant to us here
RELEVANT_GT_FILE = open(GROUND_TRUTH_PATH)

# assert that RELEVANT_GT_FILE was assigned as it should have been
assert(RELEVANT_GT_FILE != None)
print("Opened ground truth file '%s'"%RELEVANT_GT_FILE.name)

# dictionary of all the ground truths
# for ntk2 this is of the form:

'''
[
    {
        "captions": [
            "this is a mobile phone settings app or a background and themes app",
            "in the center of the screen and in the foreground is a dialog box asking the user if they want to unlock the background",
            "in the bottom right hand corner of the message box is a cancel button",
            "in the bottom left hand corner of the message box is an ok button",
            "at the bottom of the screen is a cancel button which is dimmed and in the background"
        ],
        "file_path": "com.WaterfallLiveWallpaper-screens/screenshot_1.jpg",
        "split": "train"
    },
    {
        "captions": [
            "this screen allows users to capture images of their face",
            "in the top center is a widget to capture the users face",
            "below the widget to the left is a button to view the favorite images",
            "to the right is a button to open the catalogue",
            "below the catalogue button to the right is a sliding menu to choose different faces"
        ],
        "file_path": "me.msqrd.android-screens/screenshot_2.jpg",
        "split": "train"
    }
]
'''


all_gts = json.load(RELEVANT_GT_FILE)

assert(all_gts != None)

global SCORES_DIR
SCORES_DIR = "scores".join(PREDS_DIR.rsplit("predictions", 1))

# if scores directory does not exist
if (not os.path.isdir(SCORES_DIR)) and (not os.path.isfile(SCORES_DIR)):
    print("\nNote: '%s' does not exist, so creating directory."%SCORES_DIR)
    os.mkdir(SCORES_DIR)

# if scores directory is not empty
elif len(os.listdir(SCORES_DIR)) != 0:
    print("Error: scores directory '%s' is not empty; must be empty before BLEU score evaluation."%SCORES_DIR)
    exit()

print("Using scores directory '%s'"%SCORES_DIR)



# even though this is for neuraltalk, we use the im2txt image id number
# scheme since coco caption works with this natively

# thus, we need a mapping from screenshot names (i.e. file_path": "com.WaterfallLiveWallpaper-screens/screenshot_1.jpg)
# to image ids

# open the image_ids -> screenshot names json and create a reverse mapping as well

IMAGESIDS_TO_SCREENS_PATH = "./imageids-to-screens.json"

if not os.path.isfile(IMAGESIDS_TO_SCREENS_PATH):
    print("Error: '%s' does not exist; it's needed to map image ids to screenshot names"%IMAGESIDS_TO_SCREENS_PATH)
    exit()

imageids_to_screens_file = open(IMAGESIDS_TO_SCREENS_PATH, "r")

# image_ids (str) -> screenshot name
ids_to_screens = json.load(imageids_to_screens_file)

# screenshot name -> image_ids
screens_to_ids = {}

for key in ids_to_screens:
    # make a reverse mapping
    screens_to_ids[ids_to_screens[key]] = key


imageids_to_screens_file.close()

print("Successfully populated ids_to_screens and screens_to_ids!")


# ground truth captions sorted by image id (a string such as "0009425")
# i.e. gts_by_id["0009425"] == ["caption A"] # for high, OR
# i.e. gts_by_id["0009425"] == ["caption A", "caption B", "caption C", "caption D"] # for low, OR
# i.e. gts_by_id["0009425"] == ["caption A", "caption B", "caption C", "caption D", "caption E"] # for combined

global gts_by_id
gts_by_id = {}

for entry in all_gts:
    
    # entry has the keys 'captions' (list), 'file_path' (string), and 'split' (string)
    
    # e.g. 
    '''
    {
        "captions": [
            "this is a mobile phone settings app or a background and themes app",
            "in the center of the screen and in the foreground is a dialog box asking the user if they want to unlock the background",
            "in the bottom right hand corner of the message box is a cancel button",
            "in the bottom left hand corner of the message box is an ok button",
            "at the bottom of the screen is a cancel button which is dimmed and in the background"
        ],
        "file_path": "com.WaterfallLiveWallpaper-screens/screenshot_1.jpg",
        "split": "train"
    }
    '''
    
    # first stop this iteration early if the split is not the split we're interested in
    split = entry["split"].lower()
    
    if split != data_split:
        # Wrong split! Continue
        continue
    
    
    captions = entry["captions"]
    
    file_path = entry["file_path"]

    if file_path not in screens_to_ids:
        print("Error: could not find screenshot '%s' from the ground truth file in screens_to_ids. Exiting...")
        exit()
    
    image_id = screens_to_ids[file_path]
    
    # add the image_id key if it doesn't exist yet
    if image_id not in gts_by_id:
        gts_by_id[image_id] = captions
    else:
        print("Error: duplicate image_id key in gts_by_id; this should not happen with neuraltalk2. Exiting...")
        exit()



#print(json.dumps(gts_by_id, sort_keys=False, indent=2))

#print(len(gts_by_id))

# at this point, gts_by_id has the form (sample from combined - test ground truths)

'''


{
  "0009466": [
    "the top right corner x icon allows user to close the current screen page", 
    "to the left of x icon introduction of private groups are displayed with marked map locations", 
    "in the bottom of map all group member pics are shown as icon", 
    "bottom of the screen displays a blue button which enable user to enter into the private groups application"
  ], 
  "0009467": [
    "in the top left corner is a red arrow that takes the user back to the previous page", 
    "below that is the app version with the software source in parentheses to the right of it", 
    "below that is a clickable button that allows the user to send a message to the developers", 
    "underneath that are various settings links that can be clicked and adjusted to better suit the user and how they want to interact with the app"
  ]
}



'''


# list holding all of the filenames of the prediction jsons in the given directory
prediction_jsons = []

for pred_json_name in os.listdir(PREDS_DIR):
    if pred_json_name[-5:] == ".json":
        prediction_jsons.append(pred_json_name)


if len(prediction_jsons) == 0:
    print("Error: directory '%s' contains no json files."%PREDS_DIR)
    exit()


# time that the WHOLE PROCESS starts
start_time = time.time()

# each entry is a row in the csv
# and each row of the csv will have "preds_file_name, avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4"
global scores_csv_list
scores_csv_list = [("preds_file_name, avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4",-99999)]


# function to replace the innards of the `prediction_jsons` for loop
# such that the work can be multithreaded with ThreadPools
def main_work_function(pred_json_name):

    if pred_json_name.find(data_split) == -1:
        print("Error: json file '%s' is not of the user inputted data split '%s'"%(pred_json_name,data_split))
        exit()
        
    print("Evaluating '%s'..."%pred_json_name)
    
    b_time = time.time()
    
    pred_file = open(os.path.join(PREDS_DIR,pred_json_name))
    
    
    all_preds = json.load(pred_file)
    
    # all_preds is of the form
    
    '''
    
    [
        "yducky.application.babytime-screens/screenshot_1.jpg": {
            "caption1": "in the center of the screen is a text field where the user inputs their email address",
            "caption2": "in the center of the screen is a text field where the user inputs their email address",
            "caption3": "in the center of the screen is a text field where the user inputs their email address and password"
        },
        "yuni.tvremote.controle.remote.control-screens/screenshot_1.jpg": {
            "caption1": "the screen provides the user with an error message",
            "caption2": "the screen provides the user with an error message",
            "caption3": "the screen provides the user with an error"
        }
    ]
    '''
    
    # essentially a clone of preds_file's contents but with BLEU scores included (see the below diagram)
    # each set of scores corresponds to the respective caption in "captions"
    # i.e. captions[0] corresponds to scores[0]
    # and the scores dictionary will be empty ( i.e. {} ) if the string is empty
    scores_with_preds = []
    
    
    '''
    [
        {
            "captions": [
                "in the center of the screen is a text field where the user inputs their email address",
                "in the center of the screen is a text field where the user inputs their first name",
                "in the center of the screen is a text field where the user inputs their email"
            ],
            
            "scores": [
                {"Bleu_1":0.5,"Bleu_2":0.4,"Bleu_3":0.3,"Bleu_4":0.1},
                {"Bleu_1":0.6,"Bleu_2":0.5,"Bleu_3":0.2,"Bleu_4":0.1},
                {"Bleu_1":0.3,"Bleu_2":0.2,"Bleu_3":0.1,"Bleu_4":0.0}
            ]
            
            "image_id": "0009902"
        }
    ]
    '''
    
    # lists of scores that we keep for averaging later
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    
    for key in all_preds:
        # `key` is a screenshot like 'yuni.tvremote.controle.remote.control-screens/screenshot_1.jpg'
        
        entry = all_preds[key]
        
        # entry looks like this:
        '''
            {
                "caption1": "the screen provides the user with an error message",
                "caption2": "the screen provides the user with an error message",
                "caption3": "the screen provides the user with an error"
            }
        '''
        
        # entry to append to `scores_with_preds`
        append_entry = {}
        append_entry["screenshot"] = key
        append_entry["captions"] = [entry["caption1"], entry["caption2"], entry["caption3"]]
        append_entry["scores"] = [] # list corresponding to scores
        
        image_id = screens_to_ids[key]
        
        captions_triplet = append_entry["captions"]
        
        #print(captions_triplet)
        
        
        # if this image_id has associated ground truths in gts_by_id
        if image_id in gts_by_id:

            for cap in captions_triplet:
                if cap != "": # dont evaluate bleu scores on an empty string
                    
                    bleu_1, bleu_2, bleu_3, bleu_4 = get_bleu_scores(cap, image_id, gts_by_id)
                    
                    append_entry["scores"].append({"Bleu_1":bleu_1,"Bleu_2":bleu_2,"Bleu_3":bleu_3,"Bleu_4":bleu_4})
                    bleu_1_scores.append(bleu_1)
                    bleu_2_scores.append(bleu_2)
                    bleu_3_scores.append(bleu_3)
                    bleu_4_scores.append(bleu_4)
                else:
                    # add an empty dictionary
                    append_entry["scores"].append({})
        else: # i.e. this image_id doesn't have ground truths in gts_by_id
              # this is the case for images in the test/val set that only
              # have a high level caption, for instance
            print("Image id '%s' not in gts_by_id"%image_id)
            
            # append empty dictionaries for the scores
            # instead of evaluating BLEU scores against ground truths
            # that don't exist
            
            append_entry["scores"].append([{},{},{}])
            
            
        scores_with_preds.append(append_entry)
    
    # tuple of string, int where int is the ckpt number
    csv_tuple = (("%s,%f,%f,%f,%f"%(pred_json_name,avg(bleu_1_scores),avg(bleu_2_scores),avg(bleu_3_scores),avg(bleu_4_scores))),int(pred_json_name.split("_")[-2]))
    
    scores_csv_list.append(csv_tuple)
    print("Finished evaluating bleu scores for %s in %s s"%(pred_json_name,str(time.time()-b_time)))
    
    scores_file = open(os.path.join(SCORES_DIR,pred_json_name.replace("preds","scores")), "w")
    json.dump(scores_with_preds,scores_file)
    
    print("Wrote bleu scores to %s"%scores_file.name)
    print("")
        
    pred_file.close()


# on 64 threads...
pool = ThreadPool(64)


# main loop; go through each prediction json and evaluate each of its captions
# (multithreaded variant)


# fixes unicode error in python 2 with map
#for i in range(len(prediction_jsons)):
    #prediction_jsons[i] = prediction_jsons[i].encode('ascii','ignore')

#print(prediction_jsons)
#exit()

pool.map(main_work_function, prediction_jsons)

# now we wait for all the threads to return; should be ~ 40 min

# sort the elements in `scores_csv_list` by increasing checkpoint
scores_csv_list.sort(key=sort_by_ckpt_num)

dir_id = os.path.split(os.path.dirname(SCORES_DIR))[1].replace("-","_")
scores_csv_files = open(os.path.join(SCORES_DIR,dir_id+".csv"), "w")
# make a csv file
for row_tuple in scores_csv_list:
    scores_csv_files.write(row_tuple[0]+"\n")

scores_csv_files.close()

print("\nWrote out csv '%s'"%scores_csv_files.name)

# close our GT FILE at the very end
RELEVANT_GT_FILE.close()
print("\nClosed RELEVANT_GT_FILE. Finished whole process in %s s"%str(time.time()-start_time))
