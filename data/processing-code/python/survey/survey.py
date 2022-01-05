# Script by Ali Yachnes
# ayachnes@email.wm.edu

# script to sample from various predicted captions from several different models (im2txt, neuraltalk2, seq2seq, groundtruth)
# and produce a single .csv file as input to the mechanical turk survey


# captions are sampled **RANDOMLY**, but captions with < MIN_WORDS_PER_CAPTION (4 in our case) are not considered (this includes blank captions) 

 ###############  survey thought process ###############  

# sample N screenshots randomly from the test set that have ALL 11 DESCRIPTIONS NON BLANK AND >= MIN_WORDS
# (i.e. if a screenshot has even 1 description blank, discard the screenshot)

# Note: the number 11 comes from the fact that EACH screenshot has:
# 1) im2txt low model predicted caption
# 2) im2txt high model predicted caption
# 3) im2txt combined model predicted caption
# 4) seq2seq low model predicted caption
# 5) seq2seq high model predicted caption
# 6) seq2seq combined model predicted caption
# 7) neuraltalk low model predicted caption
# 8) neuraltalk high model predicted caption
# 9) neuraltalk combined model predicted caption
# 10) groundtruth low caption (randomly sampled from one of four NONEMPTY captions) - if all low captions are empty then discard the screenshot
# 11) groundtruth high caption (given that the high level caption is NONEMPTY) - if the high level caption is empty then discard the screenshot

# ^^^ if any of the 11 captions are empty, then discard the screenshot

# once N valid (has all nonempty captions) screenshots are sampled, they must be arranged into N HITs such that:
# 1) No HIT should have 2 of the same screenshot
# 2) Each HIT should have one of each type of description for each model (i.e. it should have each number 1 to 11 above)
# 3) The order of individual descriptions should be different across HITs; (i.e. the im2txt-high description should not be the first description on each HIT) (randomization)



import os
import json
import csv
import random

random.seed(600) # seed RNG for reproducibility; i.e., running this script multiple times will produce the same output csv


# define constants

NDESCS = 11 

NUM_SCREENSHOTS_SAMPLE = 220 # NDESCS * 19 # the number of screenshots from the test set to sample

if NUM_SCREENSHOTS_SAMPLE  % NDESCS != 0:
    print("Error: NUM_SCREENSHOTS_SAMPLE must be a multiple of NDESCS; " + str(NUM_SCREENSHOTS_SAMPLE) + " is not a multiple of " + str(NDESCS) + " !")
    exit()

MIN_WORDS_PER_CAPTION = 4 # the minimum number of words for a caption; captions sampled less than this

GEMMA_PREFIX = "http://173.255.245.197:8080/GEMMA-CP/Clarity/"  # server where these images are located; this prefix will be used when adding screenshots to the mechanical turk csv



# load im2txt imageids dictionary

im2txt_ids_json = open(os.path.join("im2txt-dict","imageids-to-screens.json"), "r")

im2txt_imageids_to_screens = json.load(im2txt_ids_json) # dictionary with entries like '0001397': 'com.github.jamesgay.fitnotes-screens/screenshot_2.jpg'
                                                        # essentially maps im2txt image_ids to their screenshot name
im2txt_ids_json.close()

predictions_root = "predictions" # root predictions folder


# load im2txt low, high, both

im2txt_low = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
im2txt_high = {}
im2txt_both = {}

for fname in os.listdir(os.path.join(predictions_root, "im2txt")):
    im2txt_json = open(os.path.join(predictions_root, "im2txt", fname), "r")
    
    if fname.find("-low-") != -1:
        im2txt_low = json.load(im2txt_json)
    elif fname.find("-high-") != -1:
        im2txt_high = json.load(im2txt_json)
    elif fname.find("-both-") != -1:
        im2txt_both = json.load(im2txt_json)
    
    im2txt_json.close()



'''
DEPRECATED IM2TXT FORMAT
# [
#     {
#         "caption": "in the center of the screen is a text field where the user inputs their password",
#         "image_id": "0010086"
#     },
#     {
#        "caption": "in the center of the screen is a text field where the user inputs their email address",
#         "image_id": "0009470"
#     }
#  ]
'''

'''
im2txt_low, im2txt_high, im2txt_both are each a list like

(each entry has 1-3 captions, so the best caption is chosen based on the
 average BLEU score)
 
[
    {
        "captions": [
            "the screen allows the user to set the date",
            "the screen allows the user to set up a pin",
            "the screen allows the user to set up the device"
        ],
        "image_id": "0009187",
        "scores": [
            {
                "Bleu_1": 0.597,
                "Bleu_2": 0.517,
                "Bleu_3": 0.468,
                "Bleu_4": 0.418
            },
            {
                "Bleu_1": 0.6,
                "Bleu_2": 0.516,
                "Bleu_3": 0.464,
                "Bleu_4": 0.411
            },
            {
                "Bleu_1": 0.6,
                "Bleu_2": 0.516,
                "Bleu_3": 0.464,
                "Bleu_4": 0.411
            }
        ]
    }
    ...
]
'''

im2txt_lists = [im2txt_low, im2txt_high, im2txt_both]

for i in range(len(im2txt_lists)): # each of these is a list, now convert it to a dictionary mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
    temp_dict = {} # build a temporary dict from im2txt_lists[i], then copy it to the correct im2txt_low (high,both)

    for entry in im2txt_lists[i]:
        
        # get the key (screenshot name) from the image id
        key = im2txt_imageids_to_screens[entry["image_id"]] # i.e. key = "com.github.jamesgay.fitnotes-screens/screenshot_2.jpg"
        key = key[0:len(key)-4]+".png" # i.e. key = "com.github.jamesgay.fitnotes-screens/screenshot_2.png"
        key = key.strip()
        
        # this single entry can have up to 3 captions, so
        # find the caption with the highest average BLEU score and
        # choose it
        
        
        # list of average bleu scores, where scores[i] is the 
        # average bleu score of the ith caption
        scores = []

        # for each dictionary of 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4'
        # in this entry's `scores` list,
        for bleu_scores_4 in entry["scores"]: # this should happen 3 times for us
            # initialize the score
            curr_avg_score = 0.0
            
            
            # a few im2txt entries were malformed, as they had 
            # `scores` lists like this: [[{}, {}, {}]]
            # we skip those here (only a handful, ~4)
            if type(bleu_scores_4) == list:
                print("Invalid scores list: `%s`"%(str(entry["scores"])))
                break
            
            # if the dictionary isn't empty
            if len(bleu_scores_4) != 0:
                for score_type in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]:
                    #print(entry)
                    #print(bleu_scores_4)
                    curr_avg_score += bleu_scores_4[score_type]
                
                # divide by 4 to get the average
                curr_avg_score /= 4
                
            # append the score to the scores list; this will be the
            # ith entry where i is our current index in the scores list
            scores.append(curr_avg_score)
            
        
        # now we should have a list of 3 average scores
        # so find the best one and choose the corresponding caption
        # for our survey
        
        # the caption with the best average bleu score, which we
        # will use for the survey
        best_caption = ""
        best_score = 0.0 # the best score so far; initially 0.0
        
        for j in range(len(scores)): # for each score,
            if scores[j] > best_score:
                best_caption = entry["captions"][j]
                best_score = scores[j]
            
        # at this point best_caption corresponds to the caption that had
        # the best average bleu score in our entry, so now add it to temp_dict
        
        #print(json.dumps(entry, sort_keys=True, indent=2))
        #print(scores)
        #print(best_caption)
        
        temp_dict[key] = best_caption # i.e. mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
    
    # overwrite the im2txt list with a dictionary mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
    if i == 0:
        im2txt_low = temp_dict
    elif i == 1:
        im2txt_high = temp_dict
    elif i == 2:
        im2txt_both = temp_dict

    


assert(len(im2txt_low) > 0)
assert(len(im2txt_high) > 0)
assert(len(im2txt_both) > 0)

# at this point im2txt_low, im2txt_high, im2txt_both are each a dictionary mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption

# load seq2seq low, high, both 

seq2seq_key_low = []
seq2seq_low = [] 

seq2seq_key_high = []
seq2seq_high = []

seq2seq_key_both = []
seq2seq_both = []

for fname in os.listdir(os.path.join(predictions_root, "seq2seq")):
    if fname[len(fname)-4:len(fname)] == ".txt": # seq2seq .txt data files
        seq_file = open(os.path.join(predictions_root, "seq2seq", fname))

        seq_lines = seq_file.read().splitlines()
        
        
        if fname.find("-low-") != -1:
            for line in seq_lines:
                if line != "":
                    seq2seq_low.append(line)
            
        elif fname.find("-high-") != -1:
            for line in seq_lines:
                if line != "":
                    seq2seq_high.append(line)
           
        elif fname.find("-both-") != -1:
            for line in seq_lines:
                if line != "":
                    seq2seq_both.append(line)
        
        
        seq_file.close()
        
    elif fname[len(fname)-4:len(fname)] == ".csv": # seq2seq key file
        
        key_file = open(os.path.join(predictions_root, "seq2seq", fname))
        
        key_lines = key_file.read().splitlines()
        
        
        if fname.find("-low.csv") != -1:
            for line in key_lines:
                if line != "":
                    seq2seq_key_low.append(line)
            
            
        elif fname.find("-high.csv") != -1:
            for line in key_lines:
                if line != "":
                    seq2seq_key_high.append(line)
           
           
        elif fname.find("-both.csv") != -1:
            for line in key_lines:
                if line != "":
                    seq2seq_key_both.append(line)
            
        
        key_file.close()
            


assert(len(seq2seq_key_low) > 0)
assert(len(seq2seq_low) > 0)
assert(len(seq2seq_key_low) == len(seq2seq_low))


assert(len(seq2seq_key_high) > 0)
assert(len(seq2seq_high) > 0)
assert(len(seq2seq_key_high) == len(seq2seq_high))

assert(len(seq2seq_key_both) > 0)
assert(len(seq2seq_both) > 0)
assert(len(seq2seq_key_both) == len(seq2seq_both))



# build the 3 seq2seq dictionaries mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption

seq2seq_lists = [[seq2seq_key_low, seq2seq_low], [seq2seq_key_high, seq2seq_high], [seq2seq_key_both, seq2seq_both]]

for i in range(len(seq2seq_lists)):
    pair = seq2seq_lists[i] # pair[0] is the key list, pair[1] is the caption list
    
    temp_dict = {} # temp dictionary to build
    
    
    for j in range(len(pair[0])): # pair[0] and pair[1] are the same size (see above assert statements)
        key = pair[0][j]
        components = key.split("/")
        key = components[0] + "/" + components[1][0:len(components[1])-4].replace("hierarchy_","screenshot_") + ".png"
        key = key.strip()
        temp_dict[key] = pair[1][j] # pair[1] contains captions
        
    
    if i == 0:
        seq2seq_low = temp_dict 
    elif i == 1:
        seq2seq_high = temp_dict
    elif i == 2:
        seq2seq_both = temp_dict



assert(len(seq2seq_low) > 0)
assert(len(seq2seq_high) > 0)
assert(len(seq2seq_both) > 0)

# at this point seq2seq_low, seq2seq_high, seq2seq_both are each a dictionary mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption





# load neuraltalk2 low, high, both

neuraltalk2_low = {}
neuraltalk2_high = {}
neuraltalk2_both = {}


for fname in os.listdir(os.path.join(predictions_root, "neuraltalk2")):
    ntk2_json = open(os.path.join(predictions_root, "neuraltalk2", fname), "r")
    
    if fname.find("-low") != -1:
        neuraltalk2_low = json.load(ntk2_json)
    elif fname.find("-high") != -1:
        neuraltalk2_high = json.load(ntk2_json)
    elif fname.find("-both") != -1:
        neuraltalk2_both = json.load(ntk2_json)
    
    ntk2_json.close()
    


'''
neuraltalk2_low, neuraltalk2_high, neuraltalk2_both are each a list like 

(each entry has 1-3 captions, so the best caption is chosen based on the
 average BLEU score)

[
    {
        "captions": [
            "this screen allows the user to sign into the app",
            "the screen allows the user to select a country",
            "this screen allows the user to sign into the app"
        ],
        "scores": [
            {
                "Bleu_1": 0.2,
                "Bleu_2": 0.0,
                "Bleu_3": 0.0,
                "Bleu_4": 0.0
            },
            {
                "Bleu_1": 0.199,
                "Bleu_2": 0.0,
                "Bleu_3": 0.0,
                "Bleu_4": 0.0
            },
            {
                "Bleu_1": 0.2,
                "Bleu_2": 0.0,
                "Bleu_3": 0.0,
                "Bleu_4": 0.0
            }
        ],
        "screenshot": "com.hollar.android-screens/screenshot_1.jpg"
    },
    ...
]
'''

'''
DEPRECATED NTK2 FORMAT:

# [
#     {
#         "caption": "at the update both they there is a option for forgot with a label",
#         "path": "com.leafgreen.teen-screens/screenshot_1.jpg"
#     },
#     {
#         "caption": "in this x or charge states picture had is a new as on by which",
#         "path": "com.ToDoReminder.gen-screens/screenshot_1.jpg"
#     }
# ]


'''

neuraltalk2_lists = [neuraltalk2_low, neuraltalk2_high, neuraltalk2_both]

for i in range(len(neuraltalk2_lists)): # each of these is a list, now convert it to a dictionary mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
    temp_dict = {} # build a temporary dict from neuraltalk2_lists[i], then copy it to the correct neuraltalk2_low (high,both)


    for entry in neuraltalk2_lists[i]:
        
        # get the key (screenshot name)
        key = entry["screenshot"] # i.e. key = "com.github.jamesgay.fitnotes-screens/screenshot_2.jpg"
        key = key[0:len(key)-4]+".png" # i.e. key = "com.github.jamesgay.fitnotes-screens/screenshot_2.png"
        key = key.strip()
        
        # this single entry can have up to 3 captions, so
        # find the caption with the highest average BLEU score and
        # choose it
        
        # list of average bleu scores, where scores[i] is the 
        # average bleu score of the ith caption
        scores = []

        # for each dictionary of 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4'
        # in this entry's `scores` list,
        for bleu_scores_4 in entry["scores"]: # this should happen 3 times for us
            # initialize the score
            curr_avg_score = 0.0
            
            
            # a few entries were malformed, as they had 
            # `scores` lists like this: [[{}, {}, {}]]
            # we skip those here (only a handful, ~4)
            if type(bleu_scores_4) == list:
                print("Invalid scores list: `%s`"%(str(entry["scores"])))
                continue
            
            # if the dictionary isn't empty
            if len(bleu_scores_4) != 0:
                for score_type in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]:
                    #print(entry)
                    #print(bleu_scores_4)
                    curr_avg_score += bleu_scores_4[score_type]
                
                # divide by 4 to get the average
                curr_avg_score /= 4
                
            # append the score to the scores list; this will be the
            # ith entry where i is our current index in the scores list
            scores.append(curr_avg_score)
            
        
        # now we should have a list of 3 average scores
        # so find the best one and choose the corresponding caption
        # for our survey
        
        # the caption with the best average bleu score, which we
        # will use for the survey
        best_caption = ""
        best_score = 0.0 # the best score so far; initially 0.0
        
        for j in range(len(scores)): # for each score,
            if scores[j] > best_score:
                best_caption = entry["captions"][j]
                best_score = scores[j]
            
        # at this point best_caption corresponds to the caption that had
        # the best average bleu score in our entry, so now add it to temp_dict
        
        #print(json.dumps(entry, sort_keys=True, indent=2))
        #print(scores)
        #print(best_caption)
        
        temp_dict[key] = best_caption # i.e. mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
       
    # overwrite the neuraltalk2 list with a dictionary mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
    if i == 0:
        neuraltalk2_low = temp_dict
        print("A")
    elif i == 1:
        neuraltalk2_high = temp_dict
        print("B")
    elif i == 2:
        neuraltalk2_both = temp_dict
        print("C")


assert(len(neuraltalk2_low) > 0)
assert(len(neuraltalk2_high) > 0)
assert(len(neuraltalk2_both) > 0)



# at this point neuraltalk2_low, neuraltalk2_high, neuraltalk2_both are each a dictionary mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
 
 

# load groundtruth low, high (from test set only), AND make a list with only test set screnshots

test_screenshots = [] # list of all screenshots in the test set (i.e. a list of strings like "com.github.jamesgay.fitnotes-screens/screenshot_2.png")
groundtruth_low = {} # maps "com.kitkatandroid.keyboard-screens/screenshot_3.png" -> [low1, low2, low3, low4] (only non blank lows included in the list, so technically there can be an empty list)
groundtruth_high = {} # maps "com.kitkatandroid.keyboard-screens/screenshot_3.png" -> caption (whether caption is blank or not)


unique_csv = open(os.path.join(predictions_root, "groundtruth", "unique.csv"))
            
lines = unique_csv.read().splitlines()
            
unique_csv.close()



for i, row in enumerate(csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)):
    if i > 0: # skip the header row
        if len(row) == 7: # each row has 7 columns
            # a row is ["Filename","High","Low1","Low2","Low3","Low4","Split"]
            
            if row[6].lower() == "test": # we are only interested in test set screenshots
                key = row[0][len(GEMMA_PREFIX):len(row[0])] # row[0] is the filename like http://173.255.245.197:8080/GEMMA-CP/Clarity/com.WaterfallLiveWallpaper-screens/screenshot_1.png
                key = key.strip()
                # now key is something like "com.kitkatandroid.keyboard-screens/screenshot_3.png"
                
                test_screenshots.append(key)
                
                groundtruth_low[key] = [cap for cap in [row[2], row[3], row[4], row[5]] if cap != ""] # all nonempty lows are included in groundtruth_low dictionary
                groundtruth_high[key] = row[1] # the only high caption, whether blank or not, is included in groundtruth_high dictionary
                
                
assert(len(test_screenshots) > 0)
assert(len(groundtruth_low) > 0)
assert(len(groundtruth_high) > 0)


# now, map all captions with less than MIN_WORDS_PER_CAPTION in im2txt, seq2seq, and neuraltalk2 dictionaries to the empty caption
# so that when we sample test screenshots, we can quickly tell whether to throw out the screenshot (because there's at least one blank caption somewhere)

def num_words(caption):
        
    nwords = 0

    words = caption.strip().split()
    
    if words != ['']: #i.e. the empty string case
        nwords += len(words)

    return nwords


# each DICT is a dictionary mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption

DICT_LIST = [im2txt_low, im2txt_high, im2txt_both, seq2seq_low, seq2seq_high, seq2seq_both, neuraltalk2_low, neuraltalk2_high, neuraltalk2_both]

for i in range(len(DICT_LIST)):
    
    DICT = DICT_LIST[i]
    
    for key in DICT:
        
        # filter out any <UNK> for im2txt and UNK for neuraltalk2
        
        UNK = "" # unk token (changes for im2txt and neuraltalk2; seq2seq doesn't have one)
        
        if (i >= 0 and i <= 2):
            UNK = "<UNK>" # im2txt
        elif (i >= 3 and i <= 5):
            UNK = "<SEP>"
        elif (i >= 6 and i <= 8): # neuraltalk2
            UNK = "UNK"
        
        caption = DICT[key]
        
        words = caption.split()

        for w in range(len(words)):
            if words[w].strip().lower() == UNK.lower():
                words[w] = ""

        caption = (" ".join([wor for wor in words if wor != ""])).strip()
                

        if num_words(caption) < MIN_WORDS_PER_CAPTION: # if this caption has less than the minimum number of words (after taking out the UNK)
            DICT[key] = "" # make this caption the empty string
        else: # if the caption has at least the min number of words, then overwrite the current caption with the UNK-filtered one 
            DICT[key] = caption



# Now everything is preprocessed, captions that are too short have been mapped to the empty string, and all UNKs are gone
# now sample NUM_SCREENSHOTS_SAMPLE valid screenshots from the test set (screenshots that have ALL nonblank captions)







 ###############  survey thought process ###############  

# sample N screenshots randomly from the test set that have ALL 11 DESCRIPTIONS NON BLANK AND >= MIN_WORDS
# (i.e. if a screenshot has even 1 description blank, discard the screenshot)

# Note: the number 11 comes from the fact that EACH screenshot has:
# 1) im2txt low model predicted caption
# 2) im2txt high model predicted caption
# 3) im2txt combined model predicted caption
# 4) seq2seq low model predicted caption
# 5) seq2seq high model predicted caption
# 6) seq2seq combined model predicted caption
# 7) neuraltalk low model predicted caption
# 8) neuraltalk high model predicted caption
# 9) neuraltalk combined model predicted caption
# 10) groundtruth low caption (randomly sampled from one of four NONEMPTY captions) - if all low captions are empty then discard the screenshot
# 11) groundtruth high caption (given that the high level caption is NONEMPTY) - if the high level caption is empty then discard the screenshot

# ^^^ if any of the 11 captions are empty, then discard the screenshot




#reminder of all the dictionaries/lists at our disposal:


'''

im2txt_low = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
im2txt_high = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
im2txt_both = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption


seq2seq_low = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
seq2seq_high = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
seq2seq_both = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption


neuraltalk2_low = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
neuraltalk2_high = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption
neuraltalk2_both = {} # mapping "com.github.jamesgay.fitnotes-screens/screenshot_2.png" -> caption



test_screenshots = [] # list of all screenshots in the test set (i.e. a list of strings like "com.github.jamesgay.fitnotes-screens/screenshot_2.png")

groundtruth_low = {} # maps "com.kitkatandroid.keyboard-screens/screenshot_3.png" -> [low1, low2, low3, low4] (only non blank lows included in the list, so technically there can be an empty list)
groundtruth_high = {} # maps "com.kitkatandroid.keyboard-screens/screenshot_3.png" -> caption (whether caption is blank or not)


DICT_LIST = [im2txt_low, im2txt_high, im2txt_both, seq2seq_low, seq2seq_high, seq2seq_both, neuraltalk2_low, neuraltalk2_high, neuraltalk2_both]


'''




print("Done preprocessing.\n\nSampling " + str(NUM_SCREENSHOTS_SAMPLE) + " screenshots from the test set...")

valid_sampled_screens = [] # list of valid screens sampled

while (len(test_screenshots) > 0) and (len(valid_sampled_screens) < NUM_SCREENSHOTS_SAMPLE): # continually sample until we get the desired number of valid screenshots
    index = random.randint(0, len(test_screenshots)-1) # random index
    
    curr_screenshot = test_screenshots[index]
    
    del test_screenshots[index] # remove the current screenshot
    
    valid = True # whether this screenshot is valid
    
    for DICT in DICT_LIST:
    
        if curr_screenshot not in DICT: # if the key doesn't exist
            valid = False
            break
        elif DICT[curr_screenshot] == "": # if the mapping of the current screenshot is empty in this DICT, then 
            valid = False
            break
    
    
    
    # now check validity in the ground truths (nonempty high/low captions)
    
    if curr_screenshot in groundtruth_low:
        if len(groundtruth_low[curr_screenshot]) == 0: # i.e. ALL low level captions for this screen are empty
            valid = False
    else:
        valid = False
    
    if curr_screenshot in groundtruth_high:
        if groundtruth_high[curr_screenshot] == "": # empty high level caption
            valid = False
    else:
        valid = False
    
    if not valid: # screenshot isn't valid, so continue sampling
        continue 

    
    
    
    # screenshot is valid, so add it to the valid_sampled_screens list
    
    valid_sampled_screens.append(curr_screenshot)
    
    
    

if (len(valid_sampled_screens) < NUM_SCREENSHOTS_SAMPLE): # i.e. if we ran out of valid screenshots in the above while loop
    print("Error: not enough valid screenshots. There are only " + str(len(valid_sampled_screens)) + " which is less than " + str(NUM_SCREENSHOTS_SAMPLE) + ".")
    exit()


# once N valid (has all nonempty captions) screenshots are sampled, they must be arranged into N HITs such that:
# 1) No HIT should have 2 of the same screenshot
# 2) Each HIT should have one of each type of description for each model (i.e. it should have each number 1 to 11 above)
# 3) The order of individual descriptions should be different across HITs; (i.e. the im2txt-high description should not be the first description on each HIT)

NUM_HITS = NUM_SCREENSHOTS_SAMPLE # number of HITs is the same as the number of screenshots to sample


HITs = [] # list of HITs; HITs[0] is the first HIT, etc.
          # format HIT[0] = ["screen1","desc1","screen2","desc2",...]


for i in range(NUM_HITS):
    HITs.append([])
    for j in range(NDESCS):
        HITs[i].append(["",""]) # list that will have NDESCS entries corresponding to the NDESCS pairs of the MTurk csv (in total NDESC * 2 columns)
                                # first entry is screenshot, second entry is caption


HITs_str = "semeru-hit-id,image_url1,desc1,image_url2,desc2,image_url3,desc3,image_url4,desc4,image_url5,desc5,image_url6,desc6,image_url7,desc7,image_url8,desc8,image_url9,desc9,image_url10,desc10,image_url11,desc11\n" # string used to write out the HITs csv for mechanical turk
shuffled_HITs_str = HITs_str

lower_hit = 0 # lower bound for HIT we are currently building (inclusive)
              # current HITs range from lower_hit to lower_hit + 10 (inclusive)
              
# DICT_LIST = [im2txt_low, im2txt_high, im2txt_both, seq2seq_low, seq2seq_high, seq2seq_both, neuraltalk2_low, neuraltalk2_high, neuraltalk2_both]

lower_screen = 0 # lower bound for screen number


def caption(screenshot, dict_index): # returns a caption from the "index"th model with the given screen as a key; i.e. caption("screenshot_1.png", 0) returns a caption from im2txt_low from screenshot_1.png
    # if the index is 0 - 8, just take the caption from DICT_LIST normally
    # otherwise we need to pull the caption from ground truths
        
    if ((dict_index >= 0) and (dict_index <= 8)):
        return DICT_LIST[dict_index][screenshot] # return the caption
    else:
        if (dict_index == 9): # low level ground truth
            low_gts = groundtruth_low[screenshot] # low ground truths
            return low_gts[random.randint(0, len(low_gts)-1)] # sample a random low ground truth
                
        elif (dict_index == 10): # high level ground truth
            return groundtruth_high[screenshot] # return the high level ground truth normally



while (lower_screen < len(valid_sampled_screens)):

    if (lower_screen + NDESCS) > len(valid_sampled_screens): # i.e. NUM_SCREENSHOTS is not a multiple of NDESCS
        print("ERROR 1")
        exit()

    for sc in range(lower_screen, lower_screen+NDESCS): # for each screen, in strides of NDESCS
        if sc >= len(valid_sampled_screens):
            print("ERROR 2")
            exit()
            break
        
        curr_screen = valid_sampled_screens[sc] # current screen
        
        for hit in range(lower_screen, lower_screen+NDESCS):
            if hit >= len(HITs):
                print("ERROR 3")
                exit()
                break
            
            # HITs[hit] is the HIT we are currently modifying
            
            sc_offset = sc - lower_screen # 0 to NDESCS-1 inclusive; which screen are we currently using
            hit_offset = hit - lower_screen # 0 to NDESCS-1 inclusive; which element of the HIT are we changing (which correspond to either im2txt_low, im2txt_high, groundtruth_low, etc.)
            
            #Note: HITs[hit] has NDESCS (11 in our case) entries
            
            hit_index = (hit_offset+sc_offset)%NDESCS # index in the hit to modify; involves MODULAR ARITHMETIC so that screens wrap around after the first HIT out of NDESCS hits
            screenshot = valid_sampled_screens[sc]
            
            HITs[hit][hit_index][0] = (GEMMA_PREFIX + screenshot) # 0 is the screenshot location
            HITs[hit][hit_index][1] = caption(screenshot, hit_index) # 1 is the caption location
            
            
    lower_screen += NDESCS
    
    
    
    
    
    
    
    
    
    
    
'''  
    for hit in range(lower_hit, lower_hit+11):
        if hit >= len(HITs):
            break
        dict_index = len(HITs[hit]) # either 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, or 20
        dict_index = int(dict_index//2) # either 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

        HITs[hit].append(GEMMA_PREFIX + screen) # append the screenshot WITH GEMMA PREFIX (invariant is that we append the screenshot)
        
        # if the index is 0 - 8, just take the caption from DICT_LIST normally
        # otherwise we need to pull the caption from ground truths
        
        if ((dict_index >= 0) and (dict_index <= 8)):
            HITs[hit].append(DICT_LIST[dict_index][screen]) # append the caption
        else:
            if (dict_index == 9): # low level ground truth
                low_gts = groundtruth_low[screen] # low ground truths
                HITs[hit].append(low_gts[random.randint(0, len(low_gts)-1)]) # sample a random low ground truth
                
            elif (dict_index == 10): # high level ground truth
                HITs[hit].append(groundtruth_high[screen])
        
        
    if len(HITs[lower_hit]) == 22: # if we build a full HIT, increment the lower count by 11 and start working on the next 11 HITs
        lower_hit += 11
'''


# Done constructing HITs
print("Done building HITs...\n\nNow writing to survey.csv...")

SEMERU_HIT_ID = 1 # ID for each HIT

def format_row(row_list, semeru_id):
    
    ret_str = ""
    
    copy_list = []
    
    for pair in row_list:
        copy_list.append(pair[0]) # screenshot
        copy_list.append(pair[1]) # caption
    
    return "\"" + str(semeru_id) + "\", \"" + "\", \"".join([entry for entry in copy_list]) + "\"\n"


IDS_TO_HITS = {} # dictionary mapping numerical SEMERU IDS to a HIT list

for H in HITs:
    IDS_TO_HITS[SEMERU_HIT_ID] = H
    SEMERU_HIT_ID += 1
    



# 1) No HIT should have 2 of the same screenshot
# 2) Each HIT should have one of each type of description for each model (i.e. it should have each number 1 to 11 above)
# 3) The order of individual descriptions should be different across HITs; (i.e. the im2txt-high description should not be the first description on each HIT)\

# Now all of the HITs are built, and there are no duplicate screenshots in any HIT, and all screenshots have all 11 of their descriptions used (mixed throughout different screenshots)
# and each HIT as one of each type of description

# So #1 and #2 are fulfilled
# Now we need to do #3 by randomizing the order of entries in each HIT and by keeping track of this in a keyfile csv (each HIT has its own ID)

'''

Note: mapping before shuffling:

# 0) im2txt low model predicted caption
# 1) im2txt high model predicted caption
# 2) im2txt combined model predicted caption
# 3) seq2seq low model predicted caption
# 4) seq2seq high model predicted caption
# 5) seq2seq combined model predicted caption
# 6) neuraltalk low model predicted caption
# 7) neuraltalk high model predicted caption
# 8) neuraltalk combined model predicted caption
# 9) groundtruth low caption (randomly sampled from one of four NONEMPTY captions) - if all low captions are empty then discard the screenshot
# 10) groundtruth high caption (given that the high level caption is NONEMPTY) - if the high level caption is empty then discard the screenshot


'''

IDS_TO_ORDERS = {} # dictionary mapping semeru_id (1,2,3... int) to the order of its captions; 
                   # i.e. a non shuffled HIT would have a list like: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                   # and a shuffled HIT would have a list like: [5, 6, 3, 7, 0, 4, 2, 8, 10, 1, 9]

IDS_TO_SHUFFLED_HITS = {}

for key in IDS_TO_HITS:
    curr_HIT = IDS_TO_HITS[key]
    
    shuffle_list = [] # list to shuffle indices
    
    for i in range(len(curr_HIT)):
        shuffle_list.append(i)
    
    random.shuffle(shuffle_list)
    
    IDS_TO_ORDERS[key] = shuffle_list

    # now actually apply the shuffle
    
    shuffled_HIT = [] # copy of the original HIT but with the order shuffled
    
    for index in shuffle_list:
        shuffled_HIT.append(curr_HIT[index])
    
    IDS_TO_SHUFFLED_HITS[key] = shuffled_HIT




for key in IDS_TO_HITS:
    HITs_str += format_row(IDS_TO_HITS[key], key)

survey_csv = open("survey.csv", "w")

survey_csv.write(HITs_str)

survey_csv.close()

print("Successfully wrote to survey.csv")



for key in IDS_TO_SHUFFLED_HITS:
    shuffled_HITs_str += format_row(IDS_TO_SHUFFLED_HITS[key], key)

survey_csv = open("shuffled_survey.csv", "w")

survey_csv.write(shuffled_HITs_str)

survey_csv.close()

print("Successfully wrote to shuffled_survey.csv")



survey_key_str = "semeru-hit-id" + ("," * (NDESCS)) + "\n"

for key in IDS_TO_ORDERS:
    list_str = str(IDS_TO_ORDERS[key])
    survey_key_str += str(key) + ", " + list_str[1:len(list_str)-1] + "\n"

survey_csv = open("survey_key.csv", "w")

survey_csv.write(survey_key_str)

survey_csv.close()

print("Successfully wrote to survey_key_str.csv")

# mechanical turk csv looks like:
# image_url1	desc1	image_url2	desc2	image_url3	desc3	image_url4	desc4	image_url5	desc5	image_url6	desc6	image_url7	desc7	image_url8	desc8	image_url9	desc9	image_url10	desc10	image_url11	desc11
# Hit1_image_url1_data	Hit1_desc1_data	Hit1_image_url2_data	Hit1_desc2_data	Hit1_image_url3_data	Hit1_desc3_data	Hit1_image_url4_data	Hit1_desc4_data	Hit1_image_url5_data	Hit1_desc5_data	Hit1_image_url6_data	Hit1_desc6_data	Hit1_image_url7_data	Hit1_desc7_data	Hit1_image_url8_data	Hit1_desc8_data	Hit1_image_url9_data	Hit1_desc9_data	Hit1_image_url10_data	Hit1_desc10_data	Hit1_image_url11_data	Hit1_desc11_data
# Hit2_image_url1_data	Hit2_desc1_data	Hit2_image_url2_data	Hit2_desc2_data	Hit2_image_url3_data	Hit2_desc3_data	Hit2_image_url4_data	Hit2_desc4_data	Hit2_image_url5_data	Hit2_desc5_data	Hit2_image_url6_data	Hit2_desc6_data	Hit2_image_url7_data	Hit2_desc7_data	Hit2_image_url8_data	Hit2_desc8_data	Hit2_image_url9_data	Hit2_desc9_data	Hit2_image_url10_data	Hit2_desc10_data	Hit2_image_url11_data	Hit2_desc11_data
# Hit3_image_url1_data	Hit3_desc1_data	Hit3_image_url2_data	Hit3_desc2_data	Hit3_image_url3_data	Hit3_desc3_data	Hit3_image_url4_data	Hit3_desc4_data	Hit3_image_url5_data	Hit3_desc5_data	Hit3_image_url6_data	Hit3_desc6_data	Hit3_image_url7_data	Hit3_desc7_data	Hit3_image_url8_data	Hit3_desc8_data	Hit3_image_url9_data	Hit3_desc9_data	Hit3_image_url10_data	Hit3_desc10_data	Hit3_image_url11_data	Hit3_desc11_data





















