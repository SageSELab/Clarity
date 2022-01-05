# coding: utf-8
'''
 * Script to parse all mechanical turk csv files and remove duplicates, producing several files
 * the first file is unique.csv, which is a master record of all the unique files and their PREPROCESSED (lowercase, no punctuation, no unicode, etc.) descriptions
 * the second file is duplicate.csv, which is a spreadsheet of all the duplicates and their PREPROCESSED descriptions (since descriptions are different)
 * the third file is master-list.csv, which is a list of all 17000 clarity screenshots and a boolean 1 or 0 to indicate whether they have been used before or not (preventing sending duplicates to mech turk)
 * this script also produces preprocessed jsons and csvs for the im2txt, neuraltalk2, and seq2seq models to ensure that all models use the same preprocessed captions
 * this script disregards file/description pairs that were rejected by the SEMERU lab
 * 
 * Ali Yachnes
 * ayachnes@email.wm.edu
'''

import os
import sys
import csv
import string # for string.translate
import re # for regular expression magic
import json # for outputting neuraltalk2 input json files
import math
import random # for doing val, test, train split
import tarfile # for tar.gz writing
import time # to get elapsed time

def main(argv):
    
    random.seed(3) # this is seeded for now for reproducibility
    
    if len(argv) < 2:
        print("\nReads in all mechanical turk csv files from a given directory and produces unique.csv, duplicate.csv, and master-list.csv, disregarding rejected file/description pairs.")
        print("This script also produces preprocessed jsons and csvs for the im2txt, neuraltalk2, and seq2seq models to ensure that all models use the same preprocessed captions")
        print("This performs preprocessing on descriptions before writing out csv files and json, including:\n* casting to lowercase\n* removing punctuation\n* removing non ASCII and non alphanumeric characters\n* removing descriptions that say \"na\", \"nothing to describe\" or anything of the sort.\n* fixing common mispellings and typos")
        print("\nusage: python2.7 " + __file__ + " <directory with mechanical turk csv files> <1 = verbose, 0 = non-verbose; defaults to 0>")
        print("\nex: python2.7 " + __file__ + " ../../mechanical-turk-data/\n")
        exit()
    
    
    # Define constants
    
    CLIP_CAPTION_LENGTH = 30  # max number of words any individual caption can be (since neuraltalk2 clips captions, but im2txt and seq2seq may not, we have to make sure that all models clip captions at the same number)
    
    GEMMA_PREFIX = "http://173.255.245.197:8080/GEMMA-CP/Clarity/"  # prefix used for all filenames
    
    
    BG9_JPG_PATH = "/scratch/ayachnes/Clarity-Data/ClarityJpegs" # path where all the clarity jpegs are on bg9
    HUDSON_JPG_PATH = "/home/semeru/ClarityJpegs"  # path where all the clarity jpegs are on hudson
    SEMERU2_JPG_PATH = "/home/semeru/ClarityJpegs" # path where all the clarity jpegs are on semeru-2
    SCICLONE_JPG_PATH = "/sciclone/data10/ayachnes/ClarityJpegs" # path where all the clarity jpegs are on sciclone cluster
    BG4_JPG_PATH = "/home/scratch/ayachnes/ClarityJpegs/" # path where all the clarity jpegs are on bg4
    
    # note: percentage training data is assumed to be 1 - perc_val - perc_test
    perc_val = 0.1 # percentage validation data
    perc_test = 0.1 # percentage test data
    
    start_time = time.time() # start the clock
    
    if perc_val + perc_test >= 1:
        print("Error: validation and test percentages sum to >= 1.")
        exit()

    # Open the typos dictionary to fix common typos in submitted descriptions
    
    if not os.path.isfile("typos.txt"):
        print("Error: could not open typos.txt; ensure that typos.txt is in the working directory.")
        exit()
    
    if not os.path.isfile("num2words.txt"):
        print("Error: could not open num2words.txt; ensure that num2words.txt is in the working directory.")
        exit()
    
    def whitelist(s): #strips a string of anything not in the whitelist, returns a new whitelisted string
        ret_str = ""
        
        CHARACTER_WHITELIST= "abcdefghijklmnopqrstuvwxyz0123456789 "
        
        if len(s) == 0:
            return s
        
        for char in s:
            if char in CHARACTER_WHITELIST:
                ret_str += char  # i.e. only add characters to the description that are in the whitelist
        return ret_str
        
        
        
    #Populate typos dictionary (must be in ASCII format); for the wikipedia list, any instance of strange characters (like an a with an accent) were removed manually
    
    typos_file = open("typos.txt", "r")
    
    typos_list = typos_file.read().lower().split("\n") #read in the typos dictionary (as lowercase) as a list in the form: ['teh = the', 'tset = test', 'achived = achieved, archived'] etc.

    typos_file.close()

    typos = {} # dictionary mapping a typo to its correction
    
    multi_word_typos = {} # dictionary mapping multi word typo to its correction (i.e. "stat install" -> "start install")
    typos_leaderboard = {} # dictionary mapping a typo to how often it occurs
    
    
    for entry in typos_list:
        entry = entry.strip()
        
        mapping = entry.split(" = ")
        
        key = mapping[0].strip()
        val = mapping[1].split(",")[0].strip() # For entries like 'achived = achieved, archived', we choose the first word on the right hand side arbitrarily
        
        
        key = key.replace("-", " ") # since we won't allow hyphens when whitelisting, replace hyphens with spaces
        val = val.replace("-", " ") # since we won't allow hyphens when whitelisting, replace hyphens with spaces
        
        key = whitelist(key).strip() #whitelist removes any punctuation or character we don't want
        
        val = whitelist(val).strip()
        
        if key != val: # i.e. if whitelisting the typo and its correction results in different things (i.e. we don't want isnt -> isnt)
            
            if key.find(" ") != -1: # i.e. if this is a multi word typo like "for go to previous"
                
                multi_word_typos[key] = val # add it to the multi word typos dictionary
                
            else: 
                
                typos[key] = val # create a dictionary entry mapping a typo to its correction
        
    #Add numbers to the typos dictionary (so that if the program comes across "1" it maps it to "one")
    
    num2words_file = open("num2words.txt", "r")

    num2words_list = num2words_file.read().lower().split("\n") #read in a list mapping numbers to words (i.e. 1 = one)
    
    num2words_file.close()
    
    for entry in num2words_list:
        entry = entry.strip()
    
        mapping = entry.split(" = ")
        
        key = mapping[0].strip()
        val = mapping[1].strip()
        
        key = whitelist(key)
        val = whitelist(val)
        
        typos[key] = val # add this number to the typos dictionary (abuse of terminology, but we are mapping things in the same way so it's fine)

    ##### End of populating typos dictionary
    
    
    master_list_dir = "../../master-screen-list/"
    
    directory = argv[1]
    sql_path = os.path.join(master_list_dir, "Clarity-images-SQL.csv")

    verbose = 0

    if len(argv) == 4:
        verbose = int(argv[3]) # verbose means printing filenames with empty description lists that are excluded
                                      # and printing filenames that dont appear in the sql csv

    if sql_path[len(sql_path)-4:len(sql_path)] != ".csv":
        print("Second argument should be a .csv file!")
        exit()

    files = os.listdir(directory)


    #what a header should look like; this is used to make sure that we are parsing each row correctly, since the script prints any row that is not 33 columns and isn't the standard header row
    std_header = ['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate', 'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.image_url', 'Answer.HighLevel', 'Answer.LowLevel1', 'Answer.LowLevel2', 'Answer.LowLevel3', 'Answer.LowLevel4', 'Approve', 'Reject']

    filenames = {} # dictionary mapping filename in format "http://173.255.245.197:8080/GEMMA-CP/Clarity/com.citc.aag-screens/screenshot_4.png" 
                   # to a list containing lists of descriptions (each list should only have one list in it, but if it's a duplicate, it will have more than one list in it)
                   # i.e. "http://173.255.245.197:8080/GEMMA-CP/Clarity/com.citc.aag-screens/screenshot_4.png" mapped to [["this is a screen to ....", "desc2", "desc3", "desc4", "desc5"]]


    def clean_descriptions(descriptions, split_periods_worker = False, blacklist_row = []): # function to filter bad descriptions ( <= 2 words, "n/a", "nothing to write", etc.)
                                          # function also removes punctuation, casts to lowercase, removes trailing/leading whitespace, and replaces/removes unicode
                                          # returns true if the cleaning process resulted in all empty descriptions, and false otherwise
        
        
        # list of every dud entry encountered so far (in lowercase, without punctuation)
        duds = ["what is it to say", 
                "i got nothing", 
                "nothing to see", 
                "there are no other buttons or links on the page", 
                "there are no other buttons to describe", 
                "na", 
                "n a",
                "none", 
                "good", 
                "no data", 
                "nothing to write", 
                "nothing else", 
                "nothing other than that", 
                "na  no other buttons",
                "na no other buttons", 
                "i am aware this is needs to be blank when there are no more componenents to describe however i am not able to submit the hit without putting something in this field",
                "the hit interface will not allow this box to remain blank",
                "na this hit is not set up to allow submissions with blank text boxes",
                "hit wont submit without text here",
                "nanote this hit is not set up to allow submissions with blank fields",
                "na note this hit is not set up to allow submissions with blank fields",
                "this hit will not submit with this field left blank",
                "this hit will not submit without a message here",
                "this hit will not submit without text here",
                "this hit will not submit unless i write something here",
                "this hit will not submit without details here",
                "this hit will not submit without text in this field",
                "this hit cannot be submitted without text here",
                "na cant submit when blank",
                "there is no more functionality on this screen",
                "check",
                "leave the box blank if there are no more components to describe",
                "no other buttons shown in the page",
                "the screen remains blank in the center",
                "blank",
                "leave the box blank if there are no more components to describe",
                "0",
                "x",
                "will not let me leave blank",
                "hit wont allow me to close without filling in this text",
                "it wont let me proceed without filling these out again im sorry",
                "i cant go into more detail since i dont know what this is for",
                "lorem ipsum",
                "it wont let me proceed without filling all of these out theres no more content to describe im sorry",
                "it wont let be proceed without filling all of these out im sorry theres no more features",
                "it wont let be proceed without filling all of these out im sorry  theres no more features",
                "it wont let me proceed without filling out all of these there are no other features again im sorry",
                "it wont let me proceed without filling in every box im sorry i know were not supposed to theres no more features to describe",
                "reasons to leave blank field",
                "there is not another description",
                "there is on other description available",
                "there is no other description available",
                "there is a persistent navigation bar at the bottom for the os",
                "na i was unable to leave this box blank when trying to submit",
                "none cannot leave this field blank",
                "none unable to leave blank",
                "none unable to leave this field blank",
                "none  unable to leave field blank",
                "noneunable to leave field blank",
                "",
                "the rest of the page remaining blank",
                "the center part of the screen remains blank",
                "there are no other buttons or links on the page",
                "there are no other illustrations in the page",
                "the background of the page is grey in color",
                "no specifics",
                "there are no other buttons or links on the page",
                "the background of the page is black",
                "the background of the screen is black",
                "the are no other illustrations in the page",
                "the background of the page is white in color",
                "in the center there is a black blank space with nothing visible",
                "the background of the page is black",
                "the center of the page remains blank",
                "the background of the pge is black",
                "the background of the page is black",
                "the main body of the screen is taken up by a pink sterile background with no features",
                "the center of the screen is blank",
                "the buttons are all orange",
                "there are no buttons to press until the app has finished loading",
                "there are no buttons andor options on this screen",
                "there are no buttons and or options on this screen",
                "in the background and forming the main body of the screen is a white sterile background wallpaper with no discernible features",
                "the rest of the screen is taken up with red wallpaper background with no features",
                "there are no buttons for the user to press until the app is done loading",
                "no further input",
                "there are no buttons to describe",
                "the screen is black and empty",
                "there is no othere options and the screen is black",
                "there is no options and the screen is white",
                "the screen is white and no other options",
                "rest of screen except the popup window is blank",
                "the screen is black and no other options",
                "the screen is white and no option",
                "there are no other buttons",
                "the screen is black",
                "the screen is white",
                "from top to bottom there is a wallpaper photo which can be setted",
                "from bottom to top there is a black photo",
                "no other functionality",
                "the entire screen is black",
                "there are no options and the screen is black",
                "there are no further user interfaces",
                "top of screen is blank",
                "bottom of screen is blank",
                "blank blue screen",
                "there are no other buttons on this page",
                "the two options are the only things you can do on this screen",
                "there are no other fields on this page",
                "the screen is black and there is no other options",
                "the screen is black and there are no other options",
                "the screen is a black and there is no other option",
                "there are no other items to describe",
                "the screen is white and there is no other options",
                "the screen is white and there is a no other options",
                "the screen is red and there is no other options",
                "the screen is blackish and there is no other option",
                "no additional data",
                "there are no buttons to describe",
                "blank area on screen",
                "no further buttons or descriptions",
                "there are no other interactive buttons on the page"]
        
        
        # list of descriptions that occur twice per submission (i.e. people copied and pasted descriptions in this list within their submissions)
        deduplicates = ["from top to bottom there are eighteen countries option in one column on left",
                        "from top too bottom there are sixteen countries of alphabet a",
                        "from top to bottom there are five version information",
                        "a scrolling list shows ten choices descriptions of each category",
                        "center of screen is four emojis wide and nine emojis tall the screen scrolls for more",
                        "from top to bottom there are eighteen language options in a column on left",
                        "below the text field there is another text field where user can input their password",
                        "from top to bottom there are eighteen country names with calling code in a column",
                        "from top to bottom there are some privacy policy information",
                        "from top to bottom there are eight terms and policies in text",
                        "from middle top bottom there are nine questions about kids on left",
                        "from top to bottom there are some text information about the license of the app",
                        "from top to bottom there are twenty one country for selection on left in one column"]

        if split_periods_worker == True: # if this is worker A1USYYS8TPQ31A, who often puts multiple sentences into his low level descriptions
            
            
            low_levels = [] # table holding all low levels, split by periods
            
            num_empty = 0 # number of empty low level descriptions
            
            for k in range(1, len(descriptions)): # for each low level description
                if whitelist(descriptions[k]).strip() != "" and len(descriptions[k]) > 5 and (whitelist(descriptions[k]).strip().lower() not in duds): # if the low level description is not empty
                    for desc in descriptions[k].split("."):  # split this low level at periods
                        if desc != "":
                            low_levels.append(desc.strip())
                else:
                    num_empty += 1
            
            if num_empty > 0: # we only do this if the number of empty descriptions is > 0
            
                #print("\n\n\nSPLIT PERIODS WORKER\n\n\n")
                
                #print("BEFORE: " + str(descriptions))
            
                for k in range(min(len(low_levels), len(descriptions)-1)): # set all the low level descriptions in the descriptions table to the resulting split period low levels, to a maximum of
                    descriptions[k + 1] = low_levels[k]                    # len(low_levels) or len(descriptions) - 1)
                
            
                #print("AFTER: " + str(descriptions))
            
        
        #list of unicode characters with their replacement
        unicode_chars = [ [u"\u00b0", " degrees "], [u"\u00f3", "o"], [u"\u2153", "one third"], [u"\u00a1", "a"], [u"\u00f9", "u"], [u"\u00fc", "u"] ] 
        
        #List of unicode characters that we replace (the non punctuation unicode was taken from JsonBuilder.java)
        #char weirdO = 0xf3;
        #char[] oneThirdFracArr = {0x2153};
        #char aWithAccent = 0xa1;
        #char specialU1 = 0xf9; // ù
        #char specialU2 = 0xfc; // ü
        
        
        #Get rid of all duds in descriptions and map them to the empty string
        #Cast every description to lowercase, removing punctuation and getting rid of unicode
        for i in range(len(descriptions)):

            #Make unicode replacements

            for repl in unicode_chars: # each 'repl' is a list of [unicode, replacement]
                descriptions[i] = descriptions[i].decode("utf-8").replace(repl[0], repl[1]).encode("utf-8")
            
            
            #print(descriptions[i])
            
            #Map desc to lowercase and remove its surrounding whitespace
            descriptions[i] = descriptions[i].lower().strip() 
            
            descriptions[i] = descriptions[i].replace("\t", " ") # replace tabs with spaces
            
            descriptions[i] = descriptions[i].replace("=", " equals ") # replace = with ' equals '
            
            descriptions[i] = descriptions[i].replace("+", " plus ") # replace + with ' plus '
            
            descriptions[i] = descriptions[i].replace(" 1/2 ", " half ") # replace 1/2 with ' half '
            descriptions[i] = descriptions[i].replace(" 1/3 ", " a third ") # replace 1/3 with ' a third '
            
            descriptions[i] = descriptions[i].replace("In the bottom right hand corner is a + and - icon with one above the other and this allows the user to zoom into the map or zoom out.", "In the bottom right hand corner is a plus and minus icon with one above the other and this allows the user to zoom into the map or zoom out.") # hard coded replacement (because "minus" is tricky to replace otherwise)
            
            descriptions[i] = descriptions[i].replace("&", " and ") # replace & with ' and '
            
            descriptions[i] = descriptions[i].replace("%", " percent ") # replace % with ' percent '
            
            descriptions[i] = descriptions[i].replace("@", " at ") # replace @ with ' at '
            
            descriptions[i] = descriptions[i].replace(" #", " number") # replace # with ' number '
            
            descriptions[i] = descriptions[i].replace("# ", "number ") # replace # with ' number '
            
            

            # replace things of the form #n with number n
            number_words = re.findall(r"#[0-9]+", descriptions[i])
            
            if len(number_words) > 0:
                for entry in number_words:
                    #print(entry)
                    #print(descriptions[i])
                    descriptions[i] = descriptions[i].replace(entry, "number " + entry[1:])
                    #print(descriptions[i])
            
            
            # replace things of the form $n.n with n.n dollars ex: $3.99 -> 3.99 dollars
            dollar_words = re.findall(r'\$[0-9-]+[\.]*[0-9-]*', descriptions[i])

            if len(dollar_words) > 0:
                for entry in dollar_words:
                    descriptions[i] = descriptions[i].replace(entry, entry[1:] + " dollars")
                    
            
            descriptions[i] = descriptions[i].replace(".com", " dot com ") # replace .com with ' dot com '
            descriptions[i] = descriptions[i].replace(".net", " dot net ") # replace .net with ' dot net '
            descriptions[i] = descriptions[i].replace(".org", " dot org ") # replace .org with ' dot org '
            
            descriptions[i] = descriptions[i].replace("e-mail", "email") #before replacing hyphens with spaces, make sure e-mail is mapped to email (otherwise it would be mapped to "e mail")
            
            
            
            
            #Fix any "word" of the form word1.word2, word1-word2, word1/word2, word1!word2, word1\word2, word1{word2, etc. (any words separated by punctuation)
            
            for punc in "!\"#$%&\\()*+,-./:;<=>?@[\\]^_`{|}~": # this is string.punctuation without ' (because apostrophes are usually used for contractions)
                pattern = re.compile((r"\S+" + ("[\\" + punc + "]") + r"\S+"))
                punctuation_words = re.findall(pattern, descriptions[i])
                
                if len(punctuation_words) > 0: #if there is an instance of two words joined by punctuation,
                    for entry in punctuation_words:
                        if (entry != "n/a") and (entry != "n.a") and (entry != "n.a."): #i.e. we don't want to split n/a or n.a
                            #print("About to split on " + punc)
                            #print("Before: " + descriptions[i])
                            descriptions[i] = descriptions[i].replace(entry, " ".join(entry.split(punc)))
                            #print("After: " + descriptions[i])
                        


            # Remove any character that is not in the whitelist of characters (we only allow alphanumeric characters, spaces and tabs)
            # This will remove punctuation and any stray unicode characters we didn't catch
            descriptions[i] = whitelist(descriptions[i]).strip()
            
            
            # Fix typos
            
            nwords = 0
            
            words = [w.strip() for w in descriptions[i].strip().split(" ")] #split description into words, stripping each word of whitespace in the process
            
            for w in words: # update nwords (nwords equals the number of elements in words that arent the empty string)
                if len(w) > 0:
                    nwords += 1
            
            for j in range(len(words)):
                if len(words[j]) > 0: # i.e. if this word is not the empty string
                    if words[j] in typos: # i.e. if this word is a typo
                        if words[j] in typos_leaderboard: # add it to the typos leaderboard
                            typos_leaderboard[words[j]] += 1
                        else:
                            typos_leaderboard[words[j]] = 1                    
                       
                        words[j] = typos[words[j]] # correct it
            
            descriptions[i] = "";
            
            #print(str(nwords) + " word(s): " + str(words))
            
            for w in words: # put all the words back together into the description
                if len(w) > 0: # i.e. if this word is not the empty string
                    descriptions[i] += w + " "
            
            descriptions[i] = descriptions[i].strip() #get rid of the trailing space at the end
            
            if len(descriptions[i]) <= 5 or (descriptions[i] in duds) or (nwords <= 2): # if the description is less than 3 characters, is a dud (see the duds table), or has 2 or less words
                descriptions[i] = ""
            #elif isunique and len(descriptions[i]) <= 8 and len(descriptions[i]) > 0: #disabled for now
                #print(descriptions[i] + " : " + name)

            
            
            # Now deal with multi word typos (like the following):
            
            #below the login button where sign up new account message shown
            #for create the account
            #is given for login
            #for go previous
            
            for key in multi_word_typos:
                if descriptions[i].find(" " + key + " ") != -1: # i.e. " face book "
                    #print("Before: " + descriptions[i])
                    descriptions[i] = descriptions[i].replace(" " + key + " ", " " + multi_word_typos[key] + " ") # i.e. replace 'face book' with 'facebook' so more things map to facebook
                    #print("After: " + descriptions[i] + "\n")
                    
            descriptions[i] = descriptions[i].strip() # strip any whitespace from the ends
            
            # finally, clip the caption at CLIP_CAPTION_LENGTH words
            
            words = [w.strip() for w in descriptions[i].strip().split(" ")] #split description into words, stripping each word of whitespace in the process
            
            for j in range(len(words) - 1, -1, -1): # iterate through 'words' backwards to delete any empty word
                if words[j] == "": # if the word is empty
                    del words[j]
            
            clipped = False
            
            if len(words) > CLIP_CAPTION_LENGTH: # caption is too long
                clipped = True
                for j in range(len(words)-1, CLIP_CAPTION_LENGTH-1, - 1):
                    del words[j]
                    
                
            
            descriptions[i] = ""; # make descriptions[i] empty
            
            for w in words: # put all the words back together into the description
                if len(w) > 0: # i.e. if this word is not the empty string
                    descriptions[i] += w + " "
            
            descriptions[i] = descriptions[i].strip() #get rid of the trailing space at the end
            #if clipped == True:
               # print("Done clipping: \"" + descriptions[i] + "\"")
            #for figuring out why certain one character words pop up
            #for w in descriptions[i].split(' '):
                #w = w.strip()
                
                #if len(w) == 1 and w != "a" and w != "i" and w != "x":
                    #print(descriptions[i])
        
        
        # Now get rid of duplicate descriptions within a single submission
        
        potential_duplicates = {} # dictionary to hold each description, used to get rid of duplicates
        
        for i in range(len(descriptions)):
            if descriptions[i] in deduplicates:
                if descriptions[i] in potential_duplicates: # this description has occured before, so erase this description
                    descriptions[i] = ""
                else: # this description hasn't occured before, so add it to the potential_duplicates dictionary
                    potential_duplicates[descriptions[i]] = 1
        
        
        
        test = {}
        l = 4
        for k in range(1,len(descriptions)):
            if descriptions[k] != "":
                test[descriptions[k]] = 1
            else:
                l -= 1
        
        if len(list(enumerate(test))) < l:
            if blacklist_row[2] in garbage:
                #print(blacklist_row)
                print(("\"Tue Jul 31 14:43:01 PDT 2018\"," * 27) + "\"" + blacklist_row[2] + "\",\"" + descriptions[0] + "\",\"" + descriptions[1] + "\",\"" + descriptions[2] + "\",\"" + descriptions[3] + "\",\"" + descriptions[4] + "\"")
        
        
        for desc in descriptions:
            if desc != "":
                return False #not all of the descriptions are empty, so return false
        
        return True #all the descriptions are empty, so return true
        
    
    
    #list of HITs that we manually rejected in post (so they say "Approved" but we want to treat them as rejected)
    #rather than manually editing the batch file from MTurk, we just add a list here
    #so that the raw data can be downloaded from MTurk at any time and without modification, be run through this script
    
    # form of each entry is [HITId, WorkerId, Input.image_url]
    #                   i.e [row[0], row[15], row[27]]
    HIT_blacklist = [
    ['3MG8450X2OXM8F6HMYPMKII0L8CPU3', 'A29ZTF7JUO8F39', GEMMA_PREFIX + 'tv.ustream.ustream-screens/screenshot_6.png'],
    ['3ZICQFRS315X8I2XFUMWS8ZTXPYZZN', 'A29ZTF7JUO8F39', GEMMA_PREFIX + 'com.measuresquare.tile_calc-screens/screenshot_2.png'],
    ['3Y40HMYLL15RHO888PZ3EV43N7LUXW', 'AQ3KPKZ6UIYPF',  GEMMA_PREFIX + 'com.etrade.mobilepro.activity-screens/screenshot_5.png'],
    ['3WKGUBL7SZ9X0WX4F05XQXBMHLNL4E', 'AQ3KPKZ6UIYPF',  GEMMA_PREFIX + 'pdf.reader-screens/screenshot_2.png'],
    ['3UQ1LLR26AVC2LDLO1FO30XM9VJALE', 'AQ3KPKZ6UIYPF',  GEMMA_PREFIX + 'com.mangaquick.mangahere-screens/screenshot_3.png'],
    ['3P7RGTLO6E01IFV3313NO0K7KVEAKB', 'A26MN6JIKD4NXU', GEMMA_PREFIX + 'com.pregnancyfoodscanner-screens/screenshot_1.png'],
    ['386659BNTL43B1BZ3P0CUFCV233018', 'A26MN6JIKD4NXU', GEMMA_PREFIX + 'air.com.myheritage.mobile-screens/screenshot_5.png'],
    ['388FBO7JZRG3M3E9GK9JJEJIHHMNY3', 'A26MN6JIKD4NXU', GEMMA_PREFIX + 'com.appvv.os9launcherhd-screens/screenshot_3.png'],
    ['3DQYSJDTYLYASSBMIPKYADI01JRXEI', 'AQ3KPKZ6UIYPF',  GEMMA_PREFIX + 'com.gw.smart-screens/screenshot_2.png'],
    ['3XBXDSS8886OYYLEZB72N88KAGSXL5', 'AS6BWSIIQJDYV',  GEMMA_PREFIX + 'com.devhd.feedly-screens/screenshot_1.png'],
    ['3XAOZ9UYRZERZUGYKNUR9L3DF70Q1V', 'A18FWUMVW5APB0', GEMMA_PREFIX + 'com.natewren.linesfree-screens/screenshot_3.png'],
    ['3VI0PC2ZAY7W4JOBC3NUFPCE5LXXOM', 'A18FWUMVW5APB0', GEMMA_PREFIX + 'vn.fastsell.app-screens/screenshot_2.png'],
    ['3QI9WAYOGQYYJWW8IXVDZH9V561S6K', 'A18FWUMVW5APB0', GEMMA_PREFIX + 'com.app.studio.voicerecord-screens/screenshot_3.png'],
    ['386T3MLZLNILM1VK2A9R9PZ3G9908S', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'pixelate.color.splash.effect-screens/screenshot_2.png'],
    ['39KV3A5D18UHNWAD17055V9R0HLS7J', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.tools.wifi.calling.unlimited-screens/screenshot_3.png'],
    ['3B286OTISE467VPLLG3BNLKU56EAJC', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.bottegasol.com.migym.InShape-screens/screenshot_4.png'],
    ['31JUPBOORNRYJXK6R3VA79RRJ7JL8L', 'A187U5WJUDVPOC', GEMMA_PREFIX + 'net.slideshare.mobile-screens/screenshot_2.png'],
    ['3YO4AH2FPD7RS5VGO4QRVGK3AFUQ0S', 'A187U5WJUDVPOC', GEMMA_PREFIX + 'com.GodLiveWallpapers-screens/screenshot_2.png'],
    ['3OCZWXS7ZOUFWF14T5O7VPH0HHIL5J', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.ucy.ece464.obd.project-screens/screenshot_6.png'],
    ['3NI0WFPPI93YINCW9VCTPB3CAEJ066', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'net.hiddenscreen.faketime-screens/screenshot_2.png'],
    ['3P6ENY9P79JPM7DT94S3S6SY5MFIHJ', 'A3AJM8TBH0D9C2', GEMMA_PREFIX + 'com.handmark.sportcaster-screens/screenshot_4.png'],
    ['3GL25Y6843H84G3ZZN8HNLN9ALIXM7', 'A3AJM8TBH0D9C2', GEMMA_PREFIX + 'com.sad.mimediamanzana-screens/screenshot_5.png'],
    ['3EFNPKWBMSBZL4PEQTEVI8RS0NH030', 'A3AJM8TBH0D9C2', GEMMA_PREFIX + 'com.rhmsoft.pulsar-screens/screenshot_3.png'],
    ['3GMLHYZ0LEKGHTYIDLL5X9BCPDFUY2', 'A3AJM8TBH0D9C2', GEMMA_PREFIX + 'com.meetme.android.activities-screens/screenshot_2.png'],
    ['3EN4YVUOUCFI38XEWQVP7TVS6RNXJ3', 'A3AJM8TBH0D9C2', GEMMA_PREFIX + 'com.imperialhealthtech.pregnantly-screens/screenshot_1.png'],
    ['3HEM8MA6H9ZUGMZ05P5HAHLBIN6PQY', 'A3AJM8TBH0D9C2', GEMMA_PREFIX + 'com.hoteltonight.android.prod-screens/screenshot_6.png'],
    ['3TKXBROM5TXLDD8JX4VO4J8DIEJIJO', 'A3AJM8TBH0D9C2', GEMMA_PREFIX + 'uk.co.nickfines.RealCalc-screens/screenshot_2.png'],
    ['32L724R85L7HZOBHSH081FUPRDPIPO', 'A3AJM8TBH0D9C2', GEMMA_PREFIX + 'codeadore.textgram-screens/screenshot_3.png'],
    ['3QQUBC64ZE1EMYOZBONH00C9FBSNXX', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.cool.Wallems.BonnieWallpapers3-screens/screenshot_4.png'],
    ['37SOB9Z0SSKCI0E0FM0EGSJG3LRL3T', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.vasundhara.tattodesignsideas-screens/screenshot_1.png'],
    ['38G0E1M85MSZDZ3D8AYLQA7I7WDUVT', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.speedAnatomy.speedAnatomyLite-screens/screenshot_1.png'],
    ['3A9LA2FRWS1MC22O8QS4VGFDTZJXHY', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.gardenpatiodesigns.rahayu-screens/screenshot_1.png'],
    ['3UDTAB6HH6ML06EU72F59LUE0FR09H', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.rvparky.android2-screens/screenshot_3.png'],
    ['31D0ZWOD0AMZ5POZZ8T55G5Y6X90AW', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.xiafy.magictricks-screens/screenshot_1.png'],
    ['3R5OYNIC2CW782RSMTZCFMWY2U5PTC', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.iart.chromecastapps-screens/screenshot_1.png'],
    ['30U1YOGZGAJX45BNSK3R5NJNMSOSDC', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'victor.app.tirinhas.brasil-screens/screenshot_2.png'],
    ['3URJ6VVYUPA56H01SSLPIEFGCQSO4Y', 'A1JYF5ULFZS8L',  GEMMA_PREFIX + 'com.keyspice.comicsmaskfree-screens/screenshot_3.png'],
    ['3BO3NEOQM04ACK5F3YIBBK5USRPAI0', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.ninefolders.hd3-screens/screenshot_3.png'],
    ['3S829FDFT2O50MXMMU9E4V0F17QXDP', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.graphicalize.comicize.app-screens/screenshot_2.png'],
    ['34D9ZRXCYRHZ6Y8B8ACPFMBH1PRSAA', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.advtechgrp.android.corrlinks-screens/screenshot_3.png'],
    ['3YZ7A3YHR5G20SWUDEL87BVV9BQS54', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.maxmpz.audioplayer-screens/screenshot_3.png'],
    ['37OPIVELUUQAGPUVXG0QB84FSEAAH7', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.fujifilm.wifi-screens/screenshot_3.png'],
    ['3NQUW096N6VYX23GWV1O75M23D1L9A', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.cube.arc.blood-screens/screenshot_1.png'],
    ['3B0MCRZMBRH9GAYL5CQSACV1DRKPP7', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.google.android.apps.m4b-screens/screenshot_6.png'],
    ['3QTFNPMJC653RTOEC6B2XLSKNBQNZU', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.ap.leb-screens/screenshot_1.png'],
    ['3X0EMNLXEPCMLUVHROZRG4KJZFLPVS', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.farintuit.sitterfriends-screens/screenshot_5.png'],
    ['34KYK9TV2RV4P8D3EOCLAXVNJ7CSB5', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.music.hero.volume.master.control-screens/screenshot_2.png'],
    ['3VMHWJRYHV32Y6BID044SV30DK6XF0', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'br.com.blooti.chavinhogroove-screens/screenshot_2.png'],
    ['3N5YJ55YXGQ2QLG60EV3GS1BEQJANG', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.vervewireless.droid.foxwtxf-screens/screenshot_3.png'],
    ['36U4VBVNQO07KSMWZQTBPDIFMQ0UR0', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.flashseats.v2-screens/screenshot_3.png'],
    ['304QEQWKZP7O7TI09RYN6BP3VECO0I', 'A18FWUMVW5APB0', GEMMA_PREFIX + 'com.neonlove.heart-screens/screenshot_1.png'],
    ['3ZQA3IO31BEOEI3I5A2NKG8D06IO1L', 'A18FWUMVW5APB0', GEMMA_PREFIX + 'com.axxessweb.agencycore-screens/screenshot_3.png'],
    ['31J7RYECZLDG8X0LSIMVSZ4J51DL11', 'A1GPLSAFPS5YKW', GEMMA_PREFIX + 'com.mobincube.cric_free.sc_HFTQ3Q-screens/screenshot_1.png'],
    ['3GV1I4SEO9CBFJ1IYOYCNVV0DCTL6Z', 'A1Q6SUTBV33DQU', GEMMA_PREFIX + 'com.Mensajes.buena.manana.tarde.noche-screens/screenshot_3.png'],
    ['3PGQRAZX02702G6EOGQ5I8TO5RSSY5', 'A378E14BT458LA', GEMMA_PREFIX + 'com.crittermap.backcountrynavigator-screens/screenshot_5.png'],
    ['3VIVIU06FKZBELE4018GKB0UI1NIMJ', 'A203P669BKYX5C', GEMMA_PREFIX + 'com.cookware.lunchrecipes-screens/screenshot_5.png'],
    ['3JAOYN9IHLPV2WOB9GWEK6PZ0FU33K', 'A1SDC3D4CEEUC4', GEMMA_PREFIX + 'com.parentoscope.parent-screens/screenshot_5.png'],
    ['3NFWQRSHVE1RCKGSOQ4049357TCFGN', 'A27XJ7XOWJ05OR', GEMMA_PREFIX + 'com.andromo.dev589470.app571471-screens/screenshot_6.png'],
    ['3WPCIUYH1AVYJ5O2HK8HFSF9LA4DTJ', 'A27KTEIB2QZVKD', GEMMA_PREFIX + 'com.antiapps.polishRack2-screens/screenshot_1.png'], # this one has a good low level descriptions but its high level description just describes elements on different parts of the screen
    ['3M67TQBQQHBH1JCCZ4CYYR6RAWG9AL', 'A1MFR8AR96VI0J', GEMMA_PREFIX + 'com.androvid-screens/screenshot_1.png'],
    ['3M4KL7H8KVAYPCXE5EGT3GMMG1116D', 'A1MFR8AR96VI0J', GEMMA_PREFIX + 'com.singular.mpos.sdk-screens/screenshot_4.png'],
    ['3OPLMF3EU5AJM47AX5KTP6HM9XWNLG', 'A1MFR8AR96VI0J', GEMMA_PREFIX + 'com.pennyapp-screens/screenshot_1.png'],
    ['3ZICQFRS315X8I2XFUMWS8ZTYKCZZS', 'A1MFR8AR96VI0J', GEMMA_PREFIX + 'com.att.futureapp-screens/screenshot_2.png'],
    ['3S1L4CQSFXSQ2T3P2QCQ8NS29YZFAB', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.mangareaderupdate.mangareaderupdate-screens/screenshot_1.png'],
    ['3P888QFVX3HCVMNZQPS2LHGL4GYOQW', 'A3VZ94BSZZ9CBO', GEMMA_PREFIX + 'com.pogomaxy.mn-screens/screenshot_6.png'],
    ['3X0EMNLXEPCMLUVHROZRG4KJWVEVPK', 'A2HU6WW37B7N0G', GEMMA_PREFIX + 'com.lolostudio.gangstervegas1-screens/screenshot_1.png'],
    ['3K1H3NEY7LMUE02WXK9AG90KVXNDGZ', 'A2HU6WW37B7N0G', GEMMA_PREFIX + 'com.pikasapps.lockscreen2.wallpapers-screens/screenshot_3.png'],
    ['3OID399FXGUO0061M03SYO08UINDF0', 'AG7EI1I9DH3OK',  GEMMA_PREFIX + 'com.appnosys.nameart-screens/screenshot_6.png'],
    ['3RBI0I35XEQWIZLZY1P4A0LC9GOY3K', 'A3TUJHF9LW3M8N', GEMMA_PREFIX + 'com.marketwatch-screens/screenshot_3.png'],
    ['3RTFSSG7T8T99YGOP6BGLPUDYUFLWM', 'A8RDXT4ZILHQT',  GEMMA_PREFIX + 'com.dispatch.droid.wbns10tv-screens/screenshot_1.png'],
    ['39AYGO6AFF713J43A1ER0NZPFNMN6P', 'A8RDXT4ZILHQT',  GEMMA_PREFIX + 'com.mineworld.gunsaddon-screens/screenshot_1.png'],
    ['3VADEH0UHCK6T2EDXTUM5H0CXRKSPC', 'A2HU6WW37B7N0G', GEMMA_PREFIX + 'joansoft.dailybible-screens/screenshot_4.png'],
    ['363A7XIFV49FYQPF25HUQ9VOX9EVA4', 'A2HU6WW37B7N0G', GEMMA_PREFIX + 'com.libiitech.princesssalon2-screens/screenshot_1.png'],
    ['3UDTAB6HH6ML06EU72F59LUEXVJ90B', 'A2ODRHHGI19CQP', GEMMA_PREFIX + 'com.linkedin.android-screens/screenshot_1.png'],
    ['3MIVREZQVHLT5V2KSX09E0ZCLTTQKK', 'A296XYM32AH03K', GEMMA_PREFIX + 'com.alibaba.aliexpresshd-screens/screenshot_4.png'],
    ['3FCO4VKOZ40GDD5V0CG0GE5CW65E7Q', 'A296XYM32AH03K', GEMMA_PREFIX + 'com.apalon.alarmclock.smart-screens/screenshot_1.png'],
    ['37OPIVELUUQAGPUVXG0QB84FPU3HA0', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.emogoth.android.phone.mimi-screens/screenshot_1.png'],
    ['3XBXDSS8886OYYLEZB72N88K7WLLXF', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.copart.membermobile-screens/screenshot_1.png'],
    ['3P4C70TRMR4DCCQOA17YZM8P46PLGN', 'A1S4F2L2O88XE4', GEMMA_PREFIX + 'com.netpulse.mobile.chuzefitness-screens/screenshot_2.png'],
    ['3Q2T3FD0ONVWOIWLFY1TG5Y506AM3I', 'A7F90XA1ZQNBI',  GEMMA_PREFIX + 'com.HealthAndFitnessGuide.EasyRecipesForFreeOffline-screens/screenshot_4.png'],
    ['3MDWE879UHPMKTH45ICUT2QXOZGB9Z', 'A7F90XA1ZQNBI',  GEMMA_PREFIX + 'com.bodybuilding.store-screens/screenshot_1.png'],
    ['37M4O367VJ5Z2XJPTHPMV2WPE21M58', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.autozone.mobile-screens/screenshot_2.png'],
    ['33J5JKFMK6LN9XUD7R8AXEIAA77Q39', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.irihon.katalkcapturer-screens/screenshot_1.png'],
    ['3XAOZ9UYRZERZUGYKNUR9L3DCNTQ1H', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.tv.remote.controle.tv-screens/screenshot_1.png'],
    ['3N5YJ55YXGQ2QLG60EV3GS1BB6CNAF', 'A21N78IUBH0N5R', GEMMA_PREFIX + 'com.richtechie.hplus-screens/screenshot_4.png'],
    ['3O4VWC1GEWT6NAQ0C1O10D6OWCFJ3B', 'ANDCYE10TTEDU',  GEMMA_PREFIX + 'com.logitech.ueboom-screens/screenshot_1.png'],
    ['3XDSWAMB22FBMXQW0KJBQHM4GKKQC7', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.groundspeak.geocaching.intro-screens/screenshot_1.png'],
    ['3Z56AA6EK4NVL1J3Y0ZRN8APAXCM6O', 'ANDCYE10TTEDU',  GEMMA_PREFIX + 'com.cars.android-screens/screenshot_2.png'],
    ['34ZTTGSNJXB2351CMCXC18JKTCPQHG', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.unifiedapps.businesscardmaker-screens/screenshot_1.png'],
    ['3KLL7H3EGDOU8DXT8BRM7VISA44VHF', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'testformularysearch.mmit.com.formulary-screens/screenshot_1.png'],
    ['3WGCNLZJKFVXALCTF1O79MWET851DO', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.ilikeyou-screens/screenshot_2.png'],
    ['3NSM4HLQNRHFGY0F52K6AMBLKZLQQC', 'A24ZU6XMYAN18O', GEMMA_PREFIX + 'com.jrj.loudalarm-screens/screenshot_1.png'],
    ['3QXFBUZ4ZK3GV2DIJFZZQ948HXIGUC', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.oldguide.inter.tipstekken-screens/screenshot_1.png'],
    ['3MYASTQBG7YLRPWA0GEMAN4M1KWQD7', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.garmin.android.apps.connectmobile-screens/screenshot_2.png'],
    ['3Y3CZJSZ9KGQZDWHYPL3H1QZKP0R51', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.jawbone.upopen-screens/screenshot_1.png'],
    ['3OZ4VAIBEX2QZJSJWBU7J8Y7NW8JV8', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'philm.vilo.im-screens/screenshot_2.png'],
    ['37AQKJ12TXB50UKZ3A8WKWXVODJTT7', 'A3TUJHF9LW3M8N', GEMMA_PREFIX + 'com.crosschx.main-screens/screenshot_1.png'],
    ['307L9TDWJYF260HLRZGTTKN55WKN3J', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.fanmaster.fan.blueremote-screens/screenshot_3.png'],
    ['3AA88CN98PQ2EXT6J7H3RM2EK2AKYH', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.cnn.mobile.android.phone-screens/screenshot_5.png'],
    ['3YCT0L9OMMW6QH20HEHU1SVHXCJNSB', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.thegrizzlylabs.geniusscan.free-screens/screenshot_4.png'],
    ['3EGKVCRQFWF1LUPCDU4ASJJOX81YBB', 'A24ZU6XMYAN18O', GEMMA_PREFIX + 'com.giftcertificatesandmore.consumer.avalon-screens/screenshot_1.png'],
    ['324N5FAHSXYLA3Y1EWUJKNYK2L8VKJ', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.webascender.callerid-screens/screenshot_4.png'],
    ['3WRAAIUSBJM72FUA2KICEUS9DKMAXM', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.FNBPA.mobilebanking-screens/screenshot_2.png'],
    ['36MUZ9VAE6PWUM65RPJMG6F8IH8EDJ', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.jb.gosms.pctheme.zt.pineapple.flower-screens/screenshot_2.png'],
    ['3WRKFXQBOBUB823IJ4WRE6N1SYLYIK', 'A24ZU6XMYAN18O', GEMMA_PREFIX + 'com.camerasideas.trimmer-screens/screenshot_5.png'],
    ['304QEQWKZP7O7TI09RYN6BP3SU5O04', 'ADN0H5B7VANUM',  GEMMA_PREFIX + 'com.discord-screens/screenshot_1.png'],
    ['3PGQRAZX02702G6EOGQ5I8TO1WNSY6', 'ADN0H5B7VANUM',  GEMMA_PREFIX + 'com.dominospizza-screens/screenshot_4.png'],
    ['3LXX8KJXPWW63N6D9NHGZMQWVYZO9G', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.cootek.smartinputv5.skin.keyboard_theme_pop_art_red-screens/screenshot_1.png'],
    ['31MCUE39BK9WW80Z4V3Y30E84KY3GR', 'A24ZU6XMYAN18O', GEMMA_PREFIX + 'mega.privacy.android.app-screens/screenshot_1.png'],
    ['388CL5C1RJARC8LZU7PGKMKQJ62LHM', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'kr.co.mflare.google.iwing-screens/screenshot_1.png'],
    ['39RRBHZ0AUOGZYXNEXXT72MX6ULVZ5', 'ADN0H5B7VANUM',  GEMMA_PREFIX + 'com.droid4you.application.wallet-screens/screenshot_5.png'],
    ['38RHULDV9Y27JX8I43E3T92LHQLIW5', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.riatech.diabeticrecipes-screens/screenshot_3.png'],
    ['3R0WOCG21MWDJI3RHPYRKO1B452DUZ', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.riatech.cookbook-screens/screenshot_4.png'],
    ['33EEIIWHK7U7L6WWR1EGG9AT1R0QV6', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mapbox.mapboxandroiddemo-screens/screenshot_3.png'],
    ['33IXYHIZB559RC8PU0Z5KNO4LQJE2A', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.pictarine.photoprint-screens/screenshot_5.png'],
    ['3WUVMVA7OBQW7AUP326XBF8KLKKAZJ', 'A24ZU6XMYAN18O', GEMMA_PREFIX + 'com.appsgodev.evantubehdevantuberawvideo-screens/screenshot_1.png'],
    ['3BAKUKE49HZRBVVYX2JREBDI89VR1J', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.flashalerts.flashalertforallapps-screens/screenshot_3.png'],
    ['3A520CCNWNNYWF3T7FSKQ572XEBEAH', 'A24ZU6XMYAN18O', GEMMA_PREFIX + 'com.abclocal.kabc.news-screens/screenshot_4.png'],
    ['329E6HTMSWP1YUWIUMOD1S61B1FK3B', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.testa.lovebot-screens/screenshot_1.png'],
    ['3R5OYNIC2CW782RSMTZCFMWYZAYPTY', 'A24ZU6XMYAN18O', GEMMA_PREFIX + 'com.toxic.apps.chrome-screens/screenshot_4.png'],
    ['3L55D8AUFAKKVZHIRU0533P6FT1YCI', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.nobluffdating.com.app-screens/screenshot_1.png'],
    ['38EHZ67RIMFXIU89AD8DZZNE4BFMGQ', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.rubisoft.lesbianradar-screens/screenshot_4.png'],
    ['3L4YG5VW9NFR2GS5VU82AOXNI57DDP', 'A24ZU6XMYAN18O', GEMMA_PREFIX + 'com.vervewireless.droid.foxwtxf-screens/screenshot_6.png'],
    ['36QZ6V15890JL7M9EFTGFNNB58CSU8', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.desaxedstudios.bassbooster-screens/screenshot_5.png'],
    ['3ABAOCJ4R8ROVJ2ND9DPTDRGAYOQMP', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.asus.filemanager-screens/screenshot_5.png'],
    ['3BFF0DJK8XZ4LU7KEA36AR19MU5STH', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.hudway.online-screens/screenshot_1.png'],
    ['3ZQX1VYFTDS6PIN34VB2ZQVLBSHO8R', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.cisco.bce-screens/screenshot_2.png'],
    ['3JMNNNO3B1R38CUGF0TB6TV4SUOW28', 'ADSGMZ4LJ0660',  GEMMA_PREFIX + 'com.andruids.musicbox-screens/screenshot_1.png'],
    ['3YGE63DIN8KWC2R20DPXN52PVW4W0O', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.calculator.vault-screens/screenshot_1.png'],
    ['3URJ6VVYUPA56H01SSLPIEFG96LO4K', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.caremark.caremark-screens/screenshot_4.png'],
    ['3NCN4N1H1G479VLY6SVZB3LHTOXBNY', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.babygirlkidnames.babyboynames-screens/screenshot_3.png'],
    ['31MBOZ6PAOE0V0AJ8FBF9VNA6EXLCD', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.edmunds-screens/screenshot_4.png'],
    ['3Y3N5A7N4GWX4LT94JUJ6ZUI975MYO', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.jane.android-screens/screenshot_2.png'],
    ['3WA2XVDZEM4X9DRQH2B8A05G1VLE6R', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.savage.android.savage-screens/screenshot_1.png'],
    ['34O39PNDK6VSOTDQZZCGKNQRULMRB2', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'vmrmedia.berriospr-screens/screenshot_5.png'],
    ['3OEWW2KGQJYI6DJUL3MILI9MMJEODX', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.google.vr.expeditions-screens/screenshot_2.png'],
    ['3CMIQF80GNDM6GHVQT684HUUKY9Q6F', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.cars.quickoffers-screens/screenshot_2.png'],
    ['3IJ95K7NDXZ30CT9MENDCECE91PGNK', 'A196N792OOTIB5', GEMMA_PREFIX + 'com.ebates-screens/screenshot_3.png'],
    ['3QX22DUVOO4G0RY4Z87ZZ0QORQ3VMO', 'A3DUFPS0143F03', GEMMA_PREFIX + 'com.webmap-screens/screenshot_1.png'],
    ['3K1H3NEY7LMUE02WXK9AG90KVXNGD2', 'ADSGMZ4LJ0660', GEMMA_PREFIX + 'com.codemindedsolutions.wink.meetme.freedating-screens/screenshot_1.png'],
    ['3BJKPTD2QCZSKF77ZL31KHWD0SMTRU', 'A196N792OOTIB5', GEMMA_PREFIX + 'com.pac_12.android_player-screens/screenshot_4.png'],
    ['3QMELQS6Y5YMIHWV38V8974ZGKBR6H', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.perfectcorp.ycn-screens/screenshot_1.png'],
    ['3Q7TKIAPOTXA0SGX8B6QT15SRE9DL5', 'A196N792OOTIB5', GEMMA_PREFIX + 'com.candl.athena-screens/screenshot_2.png'],
    ['3W9XHF7WGKI6XBUC91U4J5AZ50STK1', 'A3DUFPS0143F03', GEMMA_PREFIX + 'com.mobincube.android.sc_3DJS18-screens/screenshot_1.png'],
    ['3U74KRR67M875HFF6EMKBXR3ZVXNT0', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.rsupport.mvagent-screens/screenshot_1.png'],
    ['3ZURAPD288AU85QP67JXXMZZ5LL1FZ', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.onedebit.chime-screens/screenshot_2.png'],
    ['38F60IALAG44KLN858KM0LVQR2N0T3', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.zinio.mobile.android.libraries-screens/screenshot_1.png'],
    ['3ICOHX7ENCY2V25BFDVR1ZVPOSZE0Q', 'A196N792OOTIB5', GEMMA_PREFIX + 'com.AutoRepairInvoice-screens/screenshot_1.png'],
    ['3V7ICJJAZA3LNNBSONG3CUFH272B43', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.adhoclabs.burner-screens/screenshot_1.png'],
    ['3ZQA3IO31BEOEI3I5A2NKG8DXMC1OL', 'ADSGMZ4LJ0660', GEMMA_PREFIX + 'butterly.bubble-screens/screenshot_1.png'],
    ['388FBO7JZRG3M3E9GK9JJEJIEXFYN0', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.dungelin.heartrate-screens/screenshot_4.png'],
    ['31SIZS5W592FSZFFIH96FCLQGLNQRD', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.nationalprostaff.mobile-screens/screenshot_3.png'],
    ['3SBNLSTU6USU5HX1NCXZ1GGQ6NHDZO', 'A3TV4KEZVXVQO9', GEMMA_PREFIX + 'com.att.mobiletransfer-screens/screenshot_3.png'],
    ['39I4RL8QGJ4VE0EEOFQZY5IFXK7H4P', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.servicemagic.consumer-screens/screenshot_4.png'],
    ['32ZCLEW0BZ7DUZVNAZL8DG9P187JPN', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.j2.efax-screens/screenshot_4.png'],
    ['3B623HUYJ4DLNAFWBCSBJV5M8HK8S8', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mcu.iVMS-screens/screenshot_1.png'],
    ['3PN6H8C9R4DT49PTBKH0KNPHX2ADAN', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'video.player.audio.player.music-screens/screenshot_3.png'],
    ['3QQUBC64ZE1EMYOZBONH00C9CRLNXJ', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mytowntonight.aviationweather-screens/screenshot_3.png'],
    ['3OYHVNTV5TLQKJP4DK85PV4C6SBOKA', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.hotornot.app-screens/screenshot_6.png'],
    ['3OLZC0DJ8J2U488DET9777J7AHJIVV', 'A1XEYF0O7WJ42S', GEMMA_PREFIX + 'com.ew.coloring.flowers-screens/screenshot_2.png'],
    ['3TLFH2L6Y9BBGVQMKVO0J9O5O06T2F', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'brychta.stepan.quantum_en-screens/screenshot_1.png'],
    ['3N7PQ0KLI5CYCU48Y0DA3XTWR4JE3L', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.google.vr.cyclops-screens/screenshot_5.png'],
    ['3ZZAYRN1I6EPN2FR7TMXQMR8P52TOL', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.iart.chromecastapps-screens/screenshot_5.png'],
    ['31S7M7DAGGDHHHCUU165Y5NEU0XTL4', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.mhunters.app-screens/screenshot_1.png'],
    ['3KTCJ4SCVGO0EFI6SUNASCJ82MWM1Q', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.andersonsolutions.weathertime-screens/screenshot_6.png'],
    ['385MDVINFC23E8SOMLZ35AHLU5AJWI', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.nbcnews.today-screens/screenshot_4.png'],
    ['3UUIU9GZC5S3FS992EQYHGRH8AXT5G', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.kumandgo.rewards-screens/screenshot_4.png'],
    ['3G3AJKPCXLFH0V43YFUNK5DINGKY45', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.jb.gosms.pctheme.zt.pineapple.flower-screens/screenshot_5.png'],
    ['3IQ9O0AYW6MFRG4O9Y9S6PVMDW3TIQ', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.clinical.quicklabreference-screens/screenshot_6.png'],
    ['3HUR21WDDUC8YXSDIQWSHGAGGBPYXK', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.ram.transparentlivewallpaper-screens/screenshot_3.png'],
    ['3OND0WXMHW2D62B7DL2C7RU6A91EHS', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.lamudi.android-screens/screenshot_6.png'],
    ['3HYV4299H0JKDRAEAH86UE177QB8E7', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.creatorfactory.citizen.us.audio-screens/screenshot_1.png'],
    ['32LAQ1JNT9CK07ZO3FY6PSJX7RQTUY', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.damiapp.softdatacable-screens/screenshot_3.png'],
    ['3AQN9REUTF3U0RNWRQVGN97O0TDDYX', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.jb.gosms.pctheme.color.lms-screens/screenshot_1.png'],
    ['34KYK9TV2RV4P8D3EOCLAXVNGN5BSA', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.tio.pge-screens/screenshot_1.png'],
    ['3H781YYV6T53BB22DFJ1LAXUL3WTEH', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'bbox.anime.walpaper-screens/screenshot_2.png'],
    ['3PUV2Q8SV4RYNJULHYHWFY9NFKVDBI', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.criticalhitsoftware.policeradio-screens/screenshot_1.png'],
    ['33BFF6QPI1YSTUQZJ07GP30WY8OW3J', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.kauf.sticker.funfacechangerproeffects-screens/screenshot_2.png'],
    ['3OPLMF3EU5AJM47AX5KTP6HM5IBLNV', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mywebgrocer.ShopRiteCircular-screens/screenshot_1.png'],
    ['3Q2T3FD0ONVWOIWLFY1TG5Y506B3M0', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'mp3songs.mp3player.mp3cutter.ringtonemaker-screens/screenshot_6.png'],
    ['3JY0Q5X05JTXQKCNZAAIMU3H8P3GGC', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.bediryazilim.violin-screens/screenshot_2.png'],
    ['374UMBUHN5COEDIOHOD6VMXR0RJTCO', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mmobile.app.event-screens/screenshot_4.png'],
    ['3ZCC2DXSD7RXY2INNMIU0UHPIHJYYQ', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.makeuptuto.natural-screens/screenshot_5.png'],
    ['3WPCIUYH1AVYJ5O2HK8HFSF9LRVTDO', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.hyperspeed.rocketclean-screens/screenshot_1.png'],
    ['3L2OEKSTW9XIJWR5AIK01HTNP2G8YM', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.skimble.workouts-screens/screenshot_3.png'],
    ['3XH7ZM9YX2H900YT1FYKTHV16LJR9S', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.whisperarts.kids.breastfeeding-screens/screenshot_1.png'],
    ['3566S7OX5D6HTCBJOOLL9UMI7X217V', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.crone.skinsforminecraftpepro-screens/screenshot_5.png'],
    ['30QQTY5GMK7X292HQCV5KWRFIUZU7N', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.delgeo.desygner-screens/screenshot_1.png'],
    ['37VUR2VJ6ACN5T6FF5TWPGEW88T1CO', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.alt12.babybump-screens/screenshot_1.png'],
    ['3XAOZ9UYRZERZUGYKNUR9L3DCNU1QT', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.pandora.android-screens/screenshot_3.png'],
    ['3ATYLI1PRTPV9ZJMVZ8TOG4ZR3BOJB', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.gotv.crackle.handset-screens/screenshot_1.png'],
    ['35F6NGNVM86I2WNSQ75VF05DZGST7V', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.andromo.dev528355.app507036-screens/screenshot_5.png'],
    ['3XBYQ44Z6PRXSBOTYFD77G4US71TWX', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.joeware.android.gpulumera-screens/screenshot_4.png'],
    ['306996CF6W74VOE915X1EW8EQNT1BH', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.parkeon.whoosh-screens/screenshot_2.png'],
    ['3OEWW2KGQJYI6DJUL3MILI9MMJEDOM', 'A3SGXMKL929VX3', GEMMA_PREFIX + 'com.bumble.app-screens/screenshot_1.png'],
    ['3SR6AEG6W5GBC7SYDME6EUMMSL6YHR', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.amagumogames.neybers-screens/screenshot_3.png'],
    ['3AXFSPQOYQL13M6TWWP31MVLZ2KJFK', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.autogravity-screens/screenshot_2.png'],
    ['3XUY87HIVP1XA44VLG68C47B0XRMMY', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.navfree.android.OSM.ALL-screens/screenshot_1.png'],
    ['3ABAOCJ4R8ROVJ2ND9DPTDRGAYOMQL', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.czene.hairbook-screens/screenshot_3.png'],
    ['3X2YVV51PURWJRCKWWTX1ALZ0OAW1R', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'ch.bitspin.timely-screens/screenshot_1.png'],
    ['3XJOUITW8UES8ES7M6E1FRM846KTQT', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'mobi.wifi.toolbox-screens/screenshot_2.png'],
    ['3D7VY91L65K1RD0YURGZYOWHOYNMB9', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.match.three.bubbleshooter.free-screens/screenshot_1.png'],
    ['3YGYP13641WUP22N2PHPBIQLBA0RNR', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.scripps.android.foodnetwork-screens/screenshot_6.png'],
    ['3IJ95K7NDXZ30CT9MENDCECE91PNGR', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.parkwhiz.driverApp-screens/screenshot_2.png'],
    ['3O71U79SRBC08ZH05D2UOD6HSM9MSA', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mercadolibre-screens/screenshot_3.png'],
    ['3VGET1QSZ0MAUDRMLCA62KCC3AAW7O', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.nilvav.certificatemaker-screens/screenshot_2.png'],
    ['3XEDXEGFX3B5H2XLBZ6UZU8U8PVK0G', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.editorchoice.videomakerfree-screens/screenshot_3.png'],
    ['351S7I5UG9JDREJAUK8G9R4U1T6JNR', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.hcom.android-screens/screenshot_2.png'],
    ['3P888QFVX3HCVMNZQPS2LHGL5Y3OQ2', 'AK511N3RDUZOD', GEMMA_PREFIX + 'ipnossoft.rma.oriental-screens/screenshot_1.png'],
    ['3AFT28WXLFPZEOZGDHNTCFPZEOMIOT', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.scvngr.levelup.app-screens/screenshot_4.png'],
    ['39WICJI5ATFWNPV4UNRNO7UEFAT3ZH', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.HealthAndFitnessGuide.FoodsToEatWhenPregnant-screens/screenshot_4.png'],
    ['3W0XM68YZPI5ORI37IAUDZR4DH21K1', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.seta.tollroaddroid.app-screens/screenshot_3.png'],
    ['3CESM1J3EIQRQDDH225EW6CG8ZQW6P', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'rah.fakegps.withjoystick-screens/screenshot_2.png'],
    ['3UY4PIS8QR86WX364V2A5R887C71N6', 'A34US36IOLDD1U', GEMMA_PREFIX + 'com.eventshigh.nearme.app-screens/screenshot_4.png'],
    ['3TTPFEFXCT79TNIS73JEV32TTBE6H0', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'org.carolinas.android.dir-screens/screenshot_1.png'],
    ['33EEIIWHK7U7L6WWR1EGG9AT1R0VQB', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.shazam.android-screens/screenshot_4.png'],
    ['38LRF35D5LJF1Q1UO0SF7FFZDSDU3I', 'AORVYERJ0JD04', GEMMA_PREFIX + 'com.cyou.cma.clauncher.theme.v53b358255ffea3993ab2be42-screens/screenshot_1.png'],
    ['3R15W654VDG4P1K2DXCAT0CRATYLQI', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.erkutaras.tyresizecalculator-screens/screenshot_3.png'],
    ['3909MD9T2Z4TS6K1IVEC46ITUUOEFU', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'co.speechnotes.speechnotes-screens/screenshot_1.png'],
    ['3FSEU3P2NRNF40UFVWY6K2VVC7PRRF', 'AORVYERJ0JD04', GEMMA_PREFIX + 'riddle.me.that.riddles.logo.quiz.icomania-screens/screenshot_2.png'],
    ['31KPKEKW4A04OZTR5MT10RP4LVMB0N', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.droid27.sensev2flipclockweather-screens/screenshot_6.png'],
    ['3TY2U1TEB7XH8OSPFMGOGKNDS91JJN', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.finestandroid.voiceeffect-screens/screenshot_1.png'],
    ['34YWR3PJ28XB12WSAILJPOGW4YB0XM', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.sonic.sonicdrivein-screens/screenshot_1.png'],
    ['3BPP3MA3TC7FSYUX46HAZJNDRQALE6', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.coolmobilesolution.fastscannerfree-screens/screenshot_4.png'],
    ['3BFF0DJK8XZ4LU7KEA36AR19MU5TSI', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.qihoo.security-screens/screenshot_6.png'],
    ['3CO05SML7VSR46AG1JFR06U83HPR0G', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.hbmakeup.styles-screens/screenshot_4.png'],
    ['3M93N4X8HKAA7RBF2GP28LE0OSEJS3', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.omniluxtrade.dinnerrecipes-screens/screenshot_4.png'],
    ['3ZXNP4Z39R8UJJFNHY34T9VW536L7K', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.ikeyboard.theme.panda_night-screens/screenshot_4.png'],
    ['3RTFSSG7T8T99YGOP6BGLPUDYUFWLX', 'AORVYERJ0JD04', GEMMA_PREFIX + 'com.harleynelson.emtrainer.free-screens/screenshot_6.png'],
    ['3VDVA3ILID20M5MKHRPFF7ZB60L1GY', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.realbyteapps.moneymanagerfree-screens/screenshot_1.png'],
    ['385MDVINFC23E8SOMLZ35AHLU5AWJV', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.at.meme.maker-screens/screenshot_3.png'],
    ['38Z7YZ2SB3P2Z0CWSUFX1KKZTP4IQ1', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.ikeyboard.emoji.sexyemoji-screens/screenshot_3.png'],
    ['3D1TUISJWINURTJD93Z2BLHOWAAIU6', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.fidelity.wi.activity-screens/screenshot_1.png'],
    ['3QX22DUVOO4G0RY4Z87ZZ0QORQ3MVF', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.emoji.ikeyboard-screens/screenshot_3.png'],
    ['3QXFBUZ4ZK3GV2DIJFZZQ948HXIUGQ', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.marvel.unlimited-screens/screenshot_5.png'],
    ['3VIVIU06FKZBELE4018GKB0UJO7IME', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.goldstar-screens/screenshot_1.png'],
    ['3CESM1J3EIQRQDDH225EW6CG8ZR6W0', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'twitch.angelandroidapps.tracerlightbox-screens/screenshot_2.png'],
    ['3DTJ4WT8BD2ZXNB1J78J7YYB6ZIZE3', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.animalsound140-screens/screenshot_6.png'],
    ['3MDKGGG61QAZYW9N7X039M5H458T6W', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.photoaffections.freeprints-screens/screenshot_5.png'],
    ['3EN4YVUOUCFI38XEWQVP7TVS37GJXB', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.arielvila.comicreader-screens/screenshot_2.png'],
    ['3MD8CKRQZZAY6CB2NRPXIB942QVRJN', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.genie9.gcloudbackup-screens/screenshot_2.png'],
    ['32K26U12DNBDWXSRMF8WGA3UICBVD6', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.alphainventor.filemanager-screens/screenshot_5.png'],
    ['3L2OEKSTW9XIJWR5AIK01HTNP2GY8C', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'mobi.infolife.ezweather.widget.oxygen-screens/screenshot_2.png'],
    ['3ECKRY5B1QJBDSDNFQYASDW3YSPZIC', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'mobi.infolife.uninstaller-screens/screenshot_3.png'],
    ['3SNR5F7R92GD2XMRJ3KX73VLAMGEIL', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'bible.kingjamesbiblelite-screens/screenshot_1.png'],
    ['31JUPBOORNRYJXK6R3VA79RRGNCL87', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.hearst.android.wyff-screens/screenshot_6.png'],
    ['3SZYX62S5GNGHUC2PBNXCKZZJD775J', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'net.openvpn.openvpn-screens/screenshot_2.png'],
    ['32W3UF2EZO84XS3Y2NCYNELZKS3C4C', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.cutegirls.stuffs-screens/screenshot_3.png'],
    ['3CMIQF80GNDM6GHVQT684HUUKYA6QW', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.banggood.client-screens/screenshot_2.png'],
    ['3I7SHAD35MJEP1X8HA4JTMAL58WM7N', 'A194IH478R25PK', GEMMA_PREFIX + 'com.meamobile.printicular-screens/screenshot_2.png'],
    ['3MXX6RQ9EVSNRHC27SY47EK6JBHP42', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.endless.healthyrecipes-screens/screenshot_3.png'],
    ['3RQVKZ7ZRK6OIXTF7SLZEDWNZ3G72I', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.ionicframework.sqmobile302915-screens/screenshot_1.png'],
    ['386659BNTL43B1BZ3P0CUFCVZJW01U', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.pilottravelcenters.mypilot-screens/screenshot_2.png'],
    ['3LVTFB9DE55O396126FUG87JECCQGH', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.calea.echo-screens/screenshot_5.png'],
    ['382GHPVPHSEGKS7F9U1XUBH5EFQ345', 'A18FWUMVW5APB0', GEMMA_PREFIX + 'com.treemolabs.apps.cbsnews-screens/screenshot_2.png'],
    ['3RDTX9JRTYOZIHEES10EO04159Q97C', 'A152S7LOEE5B38', GEMMA_PREFIX + 'com.sonos.acr-screens/screenshot_2.png'],
    ['3XWUWJ18TLDZ0KED3KOGUO5ZQ5XUUQ', 'A152S7LOEE5B38', GEMMA_PREFIX + 'com.ikesloveandsandwiches.app-screens/screenshot_6.png'],
    ['36818Z1KV30VMHNWIVBO7E356PMA3N', 'A152S7LOEE5B38', GEMMA_PREFIX + 'com.soundcloud.android-screens/screenshot_1.png'],
    ['3DWNFENNE3IR5694CFTKAIYUACBJ4W', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'co.cristiangarcia.dueodirecto-screens/screenshot_1.png'],
    ['3QTFNPMJC653RTOEC6B2XLSKKRJZNS', 'A152S7LOEE5B38', GEMMA_PREFIX + 'com.evzapp.cleanmaster-screens/screenshot_2.png'],
    ['3P6ENY9P79JPM7DT94S3S6SY229HI5', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'whitecastle.ordering-screens/screenshot_1.png'],
    ['38G0E1M85MSZDZ3D8AYLQA7I4C6VUG', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.expres.android-screens/screenshot_3.png'],
    ['3JUDR1D0D6EYOFYU3RU5E4DI4T82QB', 'A152S7LOEE5B38', GEMMA_PREFIX + 'com.capstone.nik.mixology-screens/screenshot_2.png'],
    ['3SMIWMMK61SCSO3IYK3HCCQWBL8WUR', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.gsa.gscookiefinder-screens/screenshot_5.png'],
    ['3VAOOVPI3ZFTY44PLS4ECEDX0NBLLT', 'A2CUFLOIO8ZAW4', GEMMA_PREFIX + 'com.onepersonco.revolution-screens/screenshot_4.png'],
    ['3VW0145YLYZ79WYAIJTGWCFUW3WMJU', 'A152S7LOEE5B38', GEMMA_PREFIX + 'com.darkhorse.digital-screens/screenshot_3.png'],
    ['3LAZVA75NIEV33LEOE21PZIIPSQ2O3', 'A152S7LOEE5B38', GEMMA_PREFIX + 'com.hw.blaze-screens/screenshot_1.png'],
    ['3CZH926SIC1JUQNCYEITD2L254G4EX', 'A3FXP547MXY6PL', GEMMA_PREFIX + 'mangatutorial.drawanime-screens/screenshot_2.png'],
    ['3B6F54KMR2Z3CAVBCRJW4LIEUBF1SI', 'A20KWU6IJJX5AT', GEMMA_PREFIX + 'com.mobilplug.lovetest-screens/screenshot_2.png'],
    ['3NKW03WTLMUK0VW1HBJC2BT7802QWG', 'AZDN5BDGK03D3', GEMMA_PREFIX + 'com.babyfish.bests7rts-screens/screenshot_2.png'],
    ['3EFNPKWBMSBZL4PEQTEVI8RSX3A30P', 'AZDN5BDGK03D3', GEMMA_PREFIX + 'com.swyftmedia.android.THEYSAYProject-screens/screenshot_2.png'],
    ['3XT3KXP24ZL9PITCDL1ZVF38TOT6IT', 'A20KWU6IJJX5AT', GEMMA_PREFIX + 'com.zentertain.photocollage-screens/screenshot_1.png'],
    ['3T8DUCXY0NTMGFBL543FTWWJU6H9TO', 'A2SC7SYQ22RJBR', GEMMA_PREFIX + 'hr.palamida-screens/screenshot_3.png'],
    ['30OITAWPBQQWB6KQ5AMQFDTVJCM9HE', 'A2SC7SYQ22RJBR', GEMMA_PREFIX + 'com.projeto.learnsing-screens/screenshot_3.png'],
    ['3FI30CQHVK6Z2AF67B9I9SZVYY8B6O', 'A2SC7SYQ22RJBR', GEMMA_PREFIX + 'com.getaround.android-screens/screenshot_1.png'],
    ['3UQVX1UPFS4A0MTVMT0QZYM0RPA02B', 'A2SC7SYQ22RJBR', GEMMA_PREFIX + 'com.northpark.beautycamera-screens/screenshot_4.png'],
    ['3S1WOPCJFGG9X86X1L5XJ4ALN15EJX', 'A3FXP547MXY6PL', GEMMA_PREFIX + 'com.bose.monet-screens/screenshot_1.png'],
    ['372AGES0I4SZ1WGR4V4Y9KHJDOAXRH', 'AG7EI1I9DH3OK', GEMMA_PREFIX + 'com.whereismytrainnyc.whereismytrain-screens/screenshot_3.png'],
    ['3YKP7CX6G22I6ATBQLEAF6ZRT9SB7N', 'AG7EI1I9DH3OK', GEMMA_PREFIX + 'com.arcane.incognito-screens/screenshot_1.png'],
    ['3SMIWMMK61SCSO3IYK3HCCQWBL8UWP', 'A3LHA3XDJVCTL1', GEMMA_PREFIX + 'com.atistudios.italk.ja-screens/screenshot_1.png'],
    ['302U8RURJZOMI9J4B9MZCFFOWGDNVG', 'A2XFLT2I2TJPUP', GEMMA_PREFIX + 'net.andromo.dev58853.app253634-screens/screenshot_3.png'],
    ['371Q3BEXDHW076GJK2SOWF2Q7QRZS4', 'A2XFLT2I2TJPUP', GEMMA_PREFIX + 'org.EPA.gd.meeting2017-screens/screenshot_6.png'],
    ['3ACRLU860N13FP8LDTSGLGR8FWWBE9', 'A20KWU6IJJX5AT', GEMMA_PREFIX + 'com.hp.orbit-screens/screenshot_1.png'],
    ['3D17ECOUOEIZSTAWMCIVWDA22VG13Q', 'AG7EI1I9DH3OK', GEMMA_PREFIX + 'org.EPA.gd.meeting2017-screens/screenshot_5.png'],
    ['3EN4YVUOUCFI38XEWQVP7TVS37GXJP', 'AG7EI1I9DH3OK', GEMMA_PREFIX + 'si.pilcom.apps.cakemakerkids-screens/screenshot_1.png'],
    ['307L9TDWJYF260HLRZGTTKN55WL3N0', 'A20KWU6IJJX5AT', GEMMA_PREFIX + 'com.vivint.vivintsky-screens/screenshot_2.png'],
    ['3PA41K45VNRKA4UL9QWBAQ4GAD87P5', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.fotoable.locker-screens/screenshot_2.png'],
    ['3VGET1QSZ0MAUDRMLCA62KCC3AB7W0', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.rcplatform.selfiecamera-screens/screenshot_2.png'],
    ['37Y5RYYI0PSB2BG4JK43ZUMFZQTXS5', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'wiki.algorithm.algorithms-screens/screenshot_1.png'],
    ['3D7VY91L65K1RD0YURGZYOWHOYNBMY', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.dazhonghua.wallpapaer.Angel-screens/screenshot_3.png'],
    ['3R16PJFTS3EMGQFMU0TWBXY7P1C4KH', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.duapps.antivirus-screens/screenshot_5.png'],
    ['33TGB4G0LP4CHBTJ8K9T9ZI1197XTU', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.yinzcam.nba.spurs-screens/screenshot_5.png'],
    ['3A9LA2FRWS1MC22O8QS4VGFDQFCXHK', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.app.radiostorm-screens/screenshot_1.png'],
    ['3MXX6RQ9EVSNRHC27SY47EK6JBI4PI', 'A3VOG7TQCAYWAJ', GEMMA_PREFIX + 'com.yahoo.mobile.client.android.weather-screens/screenshot_1.png'],
    ['3J5XXLQDHMYFE5QUTQ2K31HIRZMV38', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.arcsoft.perfect365-screens/screenshot_4.png'],
    ['3F6045TU7DB3W0277YYY2WV70ZE99O', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.dunkinbrands.otgo-screens/screenshot_1.png'],
    ['31YWE12TE0ZPJDWCVH6S43QJCCGX7I', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.lyrebirdstudio.pip_collage-screens/screenshot_2.png'],
    ['3LN50BUKPVYJPPJNDKXLTVMH5XCPLV', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.multiverse.jarvis-screens/screenshot_2.png'],
    ['3CVBMEMMXB3SXNWXQDO61H2POMY7H0', 'A2COCSUGZV28X', GEMMA_PREFIX + 'com.momocode.shortcuts-screens/screenshot_2.png'],
    ['3PMR2DOWOOOJY7UJNDPQ98FPSBH45W', 'A2COCSUGZV28X', GEMMA_PREFIX + 'com.mcdonalds.app-screens/screenshot_5.png'],
    ['3PEG1BH7AEE61P0V9F3JJB4DZTSBKT', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.ncr.bluecoast-screens/screenshot_2.png'],
    ['3XJOUITW8UES8ES7M6E1FRM846KQTQ', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.foxsports.android-screens/screenshot_2.png'],
    ['3VO4XFFP1595AGV093B0AVUQF9U7QW', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.emoji.coolkeyboard-screens/screenshot_1.png'],
    ['3L7SUC0TTUH07QXOWBJAE70YXUQM0N', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.andromo.dev551559.app531086-screens/screenshot_1.png'],
    ['3NRZ1LDP7WT3UE0RSLOU1AXFK6KPZE', 'A3MIJJTT8QIC6', GEMMA_PREFIX + 'com.aws.android-screens/screenshot_1.png'],
    ['3UEDKCTP9VDJZDX5WYR3E9IHG327K4', 'A3MIJJTT8QIC6', GEMMA_PREFIX + 'com.music.player.mp3player.white-screens/screenshot_6.png'],
    ['3DW3BNF1GH51F2W057XGUIPT7LEV80', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.magicjack.connect-screens/screenshot_5.png'],
    ['3E9VAUV7BW1W2KKK7G4EX8ZIFQGYAG', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.advanced.rootchecker-screens/screenshot_3.png'],
    ['3T5ZXGO9DEBOUQ16SY3J1FNPP26ZQF', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'org.aztest.iqtest-screens/screenshot_4.png'],
    ['3EHVO81VN58EM1HV1X7X07BCL0XH1C', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'softmint.babyapp-screens/screenshot_1.png'],
    ['3421H3BM9A4S2CFGTAPBRQ9AW4PJ9S', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.carpart.pro-screens/screenshot_4.png'],
    ['3GKAWYFRAPG0RDVV03ZXAIECWOAPDF', 'A0450532L5P3GX6JEY8C', GEMMA_PREFIX + 'com.sketchpunk.ocomicreader-screens/screenshot_2.png'],
    ['3ZVPAMTJWNQOFAL1FL4ULYHOAYEGR8', 'A0450532L5P3GX6JEY8C', GEMMA_PREFIX + 'com.edreams.travel-screens/screenshot_2.png'],
    ['3H6W48L9F4CZ0JVMHY4N263X34GWPV', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.mega_mc.mcpeskinstudio-screens/screenshot_4.png'],
    ['3HJ1EVZS2O6NLY9DK6XA24SF6TA3RO', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.eastcoders.caralarm-screens/screenshot_5.png'],
    ['3SSN80MU8CBDES3WEBV188V5IWGXKO', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'scan.dtc.obd.readcode.elm327.oht.carsys-screens/screenshot_1.png'],
    ['3GONHBMNHVLX9B3E05W2K63KF19MZF', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.irihon.katalkcapturer-screens/screenshot_3.png'],
    ['3FULMHZ7OUKH6EFPFC6CQAQBE66M43', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.joshclemm.android.quake-screens/screenshot_3.png'],
    ['359AP8GAGG71GFLH4LA5QQ59BUT7CR', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.citc.aag-screens/screenshot_4.png'],
    ['3EFNPKWBMSBZL4PEQTEVI8RSX3A03M', 'AE06KCFQQPAZF', GEMMA_PREFIX + 'com.coderoaps.android.kitchen.designsideas-screens/screenshot_1.png'],
    ['3BFNCI9LYKDD3FPYBSZ4XN1F5HG73T', 'AE06KCFQQPAZF', GEMMA_PREFIX + 'com.kapye.greenlight-screens/screenshot_1.png'],
    ['38B7Q9C28GSGEH099RVMIM45AY696D', 'AG7EI1I9DH3OK', GEMMA_PREFIX + 'com.whereismycar-screens/screenshot_6.png'],
    ['3VZYA8PITOL2ZJCI4C2OX5PCBZ150H', 'AG7EI1I9DH3OK', GEMMA_PREFIX + 'in.fulldive.applicationslauncher-screens/screenshot_2.png'],
    ['3EQVJH0T408FVLT43GR76DU7DJOTHX', 'AG7EI1I9DH3OK', GEMMA_PREFIX + 'com.keeratipong.skineditorminecraft-screens/screenshot_2.png'],
    ['3PZDSVZ3J54NOW0PRDLC3PFBJWH4NM', 'AG7EI1I9DH3OK', GEMMA_PREFIX + 'com.buzzfeed.android-screens/screenshot_6.png'],
    ['3421H3BM9A4S2CFGTAPBRQ9AW4Q9JJ', 'A2MV1VT0WIL4QU', GEMMA_PREFIX + 'com.microsoft.windowsintune.companyportal-screens/screenshot_4.png'],
    ['338GLSUI43YU2PPJJQYHTNM8VLXSF9', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.srems.protocol-screens/screenshot_6.png'],
    ['3YLPJ8OXX80S0QETTJ8L1R69LAQX4Y', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.dilmeapps.kickme-screens/screenshot_1.png'],
    ['3YD0MU1NC2ODN0OJK7EEKVFLBR8A7S', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.gydala.allcars-screens/screenshot_2.png'],
    ['3X7837UUADL5KTDIV3MZ7GI863I6J5', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'org.carolinas.android.dir-screens/screenshot_4.png'],
    ['3OWZNK3RYLCQG85BK0EAO5A77EE2UG', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.gloto.telemundo-screens/screenshot_6.png'],
    ['3HEADTGN2PF7X7BW4G3GLZKYXD2RV8', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'cz.hipercalc-screens/screenshot_1.png'],
    ['3ULIZ0H1VAS268X00V6OBA8MGR715G', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.hd.factory.fantasy.fix-screens/screenshot_4.png'],
    ['3RZS0FBRWKXQMLPH074MQCWUBOYPCF', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.loves.finder-screens/screenshot_4.png'],
    ['336OE47KI27C53SI3ADM8YSFPSHVWF', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.hdmi.read-screens/screenshot_5.png'],
    ['363A7XIFV49FYQPF25HUQ9VOX9EAVJ', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.ambientdesign.artrage.oils-screens/screenshot_3.png'],
    ['3Z3R5YC0P3AVHP2Y8Q373SIUX4BTFZ', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.eapp.lock_wallpapers_free-screens/screenshot_1.png'],
    ['3LB1BGHFL2J1HGJTDGVVSDPA3F1TYW', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.innovative.logomaker.free.design-screens/screenshot_3.png'],
    ['3KL228NDMV92S686P1VXKMVAF6KGKF', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.internetdesignzone.quotes-screens/screenshot_2.png'],
    ['3L1EFR8WWTSCXATKAKYQCSHI3X99FG', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.cp.mpos-screens/screenshot_6.png'],
    ['36QZ6V15890JL7M9EFTGFNNB58CUSA', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.match.three.candyswap2.free-screens/screenshot_1.png'],
    ['3WYZV0QBFJ0CLPZ28YI895CFV27XB4', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'yamayka.apps.NailsTutorial-screens/screenshot_2.png'],
    ['3Z33IC0JC091T7FAAZ3UUEK4RRWV9P', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.oneabsolute.mealplanner-screens/screenshot_3.png'],
    ['311HQEI8RS3EO8WRWZUD1O6UKCF7ZQ', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.coolmobilesolution.fastscannerfree-screens/screenshot_2.png'],
    ['3NI0WFPPI93YINCW9VCTPB3C7UC06S', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.mybabygames.nurseryrhythms-screens/screenshot_3.png'],
    ['3MDKGGG61QAZYW9N7X039M5H4596TA', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'tm.app.worldClock-screens/screenshot_4.png'],
    ['3H4IKZHALB59A4AV9I082YLBADBNNZ', 'AOSB3RSWQQENC', GEMMA_PREFIX + 'com.klonengam.nickiminajprank.calli-screens/screenshot_1.png'],
    ['379OL9DBSS1IX1AFFAQE1DOY98Z9YD', 'AH5C27EW5D73O', GEMMA_PREFIX + 'com.avis.androidapp-screens/screenshot_1.png'],
    ['34XASH8KLQ93V718DWY0T816522PMX', 'AJODB4SKR8C3H', GEMMA_PREFIX + 'com.raccoonfinger.craft.bath-screens/screenshot_1.png'],
    ['3FCO4VKOZ40GDD5V0CG0GE5CW667EK', 'AJODB4SKR8C3H', GEMMA_PREFIX + 'com.united.mobile.android-screens/screenshot_1.png'],
    ['3OCZWXS7ZOUFWF14T5O7VPH0EXBL55', 'A290BY40QP2YXX', GEMMA_PREFIX + 'com.northpark.drinkwater-screens/screenshot_1.png'],
    ['3QI9WAYOGQYYJWW8IXVDZH9V2MV6SL', 'A20KWU6IJJX5AT', GEMMA_PREFIX + 'com.cardinalcommerce.greendot-screens/screenshot_2.png'],
    ['3ZRKL6Z1E8Q0GPLEUA4ZB8MKW0XSG8', 'A20KWU6IJJX5AT', GEMMA_PREFIX + 'com.emojifamily.emoji.keyboard-screens/screenshot_1.png'],
    ['3WKGUBL7SZ9X0WX4F05XQXBME1GL40', 'A3D91TO5INWQRO', GEMMA_PREFIX + 'com.goodrx.doctors-screens/screenshot_4.png'],
    ['3I7SHAD35MJEP1X8HA4JTMAL58X7M9', 'A20KWU6IJJX5AT', GEMMA_PREFIX + 'com.tomandchee.tomandchee.android.app-screens/screenshot_2.png'],
    ['3L60IFZKF35PQC293MAISURJ2PUHHC', 'A3D91TO5INWQRO', GEMMA_PREFIX + 'com.fourchars.lmpfree-screens/screenshot_3.png'],
    ['3YOAVL4CA04H5LE8U1W6YCMKTAP4Z3', 'A3D91TO5INWQRO', GEMMA_PREFIX + 'com.passportparking.mobile.parkboston-screens/screenshot_6.png'],
    ['3ATYLI1PRTPV9ZJMVZ8TOG4ZR3BJO6', 'A20KWU6IJJX5AT', GEMMA_PREFIX + 'com.ikeyboard.theme.Diffusion-screens/screenshot_4.png'],
    ['3PR3LXCWSFMTXWK5AE4CIZHP7259X7', 'A2TTAG8NJ9EOIE', GEMMA_PREFIX + 'com.problemio-screens/screenshot_1.png'],
    ['37G6BXQPLQ8QJVAAWBP6G5M71WYEQP', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.autovindecoder-screens/screenshot_1.png'],
    ['3RSBJ6YZECDF4XXQC9H89IC7YWUOF8', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.disneydigitalbooks.disneystorycentral_goo-screens/screenshot_1.png'],
    ['3UXQ63NLAA9HLVIDUFFKYBHSOTXLB6', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.fs.anycast-screens/screenshot_2.png'],
    ['3VZYA8PITOL2ZJCI4C2OX5PCBZ105C', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.zalebox.living.room.decorating.ideas.s1-screens/screenshot_1.png'],
    ['32FESTC2NHD3EQXDHT3G0IJTJ5QUCG', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.banggood.client-screens/screenshot_1.png'],
    ['3LCXHSGDLTT2WBPNOFEBB518JKISER', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.jesso.jesbenthomas.Swastika-screens/screenshot_2.png'],
    ['3WRAAIUSBJM72FUA2KICEUS9DKMXA9', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.skindeep.mobile-screens/screenshot_3.png'],
    ['3R0WOCG21MWDJI3RHPYRKO1B452UDG', 'A2TTAG8NJ9EOIE', GEMMA_PREFIX + 'com.mynamecubeapps.myname-screens/screenshot_1.png'],
    ['3SA4EMRVJVP6STU1ORB2VBUT2Z1P0M', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.ml.BreitHedge-screens/screenshot_3.png'],
    ['3IZVJEBJ6A7VK8IMDPPLVA6YP1V6ZQ', 'A59E02K7PD6LG', GEMMA_PREFIX + 'com.figure1.android-screens/screenshot_5.png'],
    ['32CAVSKPCECJRNNF1WEWJM02F801UY', 'A2TTAG8NJ9EOIE', GEMMA_PREFIX + 'com.microsoft.launcher-screens/screenshot_4.png'],
    ['3B623HUYJ4DLNAFWBCSBJV5M8HKS8S', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.pillidapp-screens/screenshot_2.png'],
    ['3PR3LXCWSFMTXWK5AE4CIZHP724X9U', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'mp3.music.download.player.music.search-screens/screenshot_5.png'],
    ['3JGHED38EDEEJIMM0DSUNHXSEIB7YZ', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'pl.patrykgoworowski.cornieoldie-screens/screenshot_1.png'],
    ['3UUSLRKAULQBL7RV4H5GAWNRWU57DR', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'de.glossycon-screens/screenshot_4.png'],
    ['3BAWBGQGYLMXFORHV25AGITYW18V7D', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'au.com.realestate.app-screens/screenshot_1.png'],
    ['3JHB4BPSFKW9OZJTO09KORLWAZHQ9Q', 'A3IB7YJHP04OHI', GEMMA_PREFIX + 'com.bbt.myfi-screens/screenshot_1.png'],
    ['3P520RYKCHTIF5OY2JG8MCDJRO4U58', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.appx.pingguo.launcher-screens/screenshot_1.png'],
    ['38XPGNCKHTN0W19YT473D69O5ZJ4V3', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'pl.planmieszkania.android-screens/screenshot_1.png'],
    ['3S829FDFT2O50MXMMU9E4V0FYNJXDB', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.inveno.noticias-screens/screenshot_6.png'],
    ['3AA88CN98PQ2EXT6J7H3RM2EK2AYKV', 'A3DF6MHEX7JWMM', GEMMA_PREFIX + 'tv.peel.app-screens/screenshot_4.png'],
    ['3GMLHYZ0LEKGHTYIDLL5X9BCMT8YUS', 'ATA9AZBKH2LNZ', GEMMA_PREFIX + 'com.leguide.lego.friends-screens/screenshot_3.png'],
    ['3MJ9GGZYO3T61ZRDEVXJO4YD0BMA2C', 'A20KWU6IJJX5AT', GEMMA_PREFIX + 'wb.mobile.cx.client.droid-screens/screenshot_2.png'],
    ['3GL25Y6843H84G3ZZN8HNLN971BXMT', 'A3KFX4FS1SANOS', GEMMA_PREFIX + 'com.jsdev.pfei-screens/screenshot_4.png'],
    ['3TX9T2ZCB9OTENXS472PT45BD3NZWJ', 'A3M25WRZYKV82I', GEMMA_PREFIX + 'com.andromo.dev535138.app530249-screens/screenshot_1.png'],
    ['31D0ZWOD0AMZ5POZZ8T55G5Y3D20AI', 'A38J5E3EDQYDZ1', GEMMA_PREFIX + 'com.mobivate.colourgo-screens/screenshot_6.png'],
    ['3V0TR1NRVAPG4D60I9G7HJVBKPIA48', 'A3THGBIR7JXV1L', GEMMA_PREFIX + 'com.economist.darwin-screens/screenshot_3.png'],
    ['3X4Q1O9UBH92P4IKUQJWLQZQ08C7OO', 'A3THGBIR7JXV1L', GEMMA_PREFIX + 'com.farintuit.sitterfriends-screens/screenshot_1.png'],
    ['36D1BWBEHNO73002BYNOXVTDUSB2MO', 'A3THGBIR7JXV1L', GEMMA_PREFIX + 'unclaimed.money-screens/screenshot_2.png'],
    ['341H3G5YF0106XW7B094M9W7CY90ZJ', 'A3THGBIR7JXV1L', GEMMA_PREFIX + 'com.amikulich.pregnancycalculator-screens/screenshot_5.png'],
    ['3A3KKYU7P34TFGY9PICVL292YZ5MWP', 'A2K8VGDGGFHPJI', GEMMA_PREFIX + 'com.adp.run.mobile-screens/screenshot_6.png'],
    ['3JAOYN9IHLPV2WOB9GWEK6PZ0FU33K', 'A1SDC3D4CEEUC4', GEMMA_PREFIX + 'com.parentoscope.parent-screens/screenshot_5.png'],
    ['3VCK0Q0PO516P8IO8CYARMPY2K0N0O', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.happytree.apps.contractiontimer-screens/screenshot_1.png'],
    ['35NNO802AVJHW3Z4C2NGXQPUOEHNIJ', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'forderror.forddtc.elm327.fordtroublecode.oht.fordscan.fordsysscan-screens/screenshot_1.png'],
    ['372AGES0I4SZ1WGR4V4Y9KHJDOARXB', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.antivirus.tablet-screens/screenshot_1.png'],
    ['3Q9SPIIRWJ9SDFPKSFMQCBE24IGWAF', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.acmeaom.android.myradar-screens/screenshot_6.png'],
    ['3UY4PIS8QR86WX364V2A5R887C6N1R', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.eggheadgames.quicklogicproblems-screens/screenshot_1.png'],
    ['35U0MRQMUJU40UJGCOMCR4FTMQIOVW', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.topoflearning.best.medical.vibering.abbreviation.words.dictionary-screens/screenshot_2.png'],
    ['3VP28W7DUN7R31BXEISPPGJBI0XZFL', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.digitalcosmos.shimeji-screens/screenshot_4.png'],
    ['3E24UO25QZDJL44FBGE4FCZU5XRO65', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.apps.zc.memessongsplus-screens/screenshot_1.png'],
    ['3FVBZG9CLJ1EFCWD5E3XM2S2G8RH09', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.obdautodoctor-screens/screenshot_4.png'],
    ['31ODACBENU2RTEFSX69B5MQM2N6QS1', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.microsoft.office.excel-screens/screenshot_1.png'],
    ['39TX062QX1B7IUVPT532RME37AV3XK', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.sherpaoutdoorapp.noaaweatherbuoys-screens/screenshot_3.png'],
    ['3Q7TKIAPOTXA0SGX8B6QT15SRE9LDD', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.nikon.spoton-screens/screenshot_3.png'],
    ['3HEADTGN2PF7X7BW4G3GLZKYXD2VRC', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.barilab.handmirror.googlemarket-screens/screenshot_3.png'],
    ['3ZG552ORAMRQTNUBPQOFKRCQLLMV2X', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.apalon.weatherlive.free-screens/screenshot_6.png'],
    ['34F34TZU7WMH2SUH81AWH31WQYG2JK', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.mexico.autosusados-screens/screenshot_3.png'],
    ['31J7RYECZLDG8X0LSIMVSZ4J2H6L1N', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.ambientdesign.artrage.oils-screens/screenshot_5.png'],
    ['3EKZL9T8Y89Y94RTHUWHH3U3PXOHC2', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.barventure.app1-screens/screenshot_1.png'],
    ['3XBXDSS8886OYYLEZB72N88K7WLXLR', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'org.contentarcade.apps.airfryer-screens/screenshot_1.png'],
    ['32L724R85L7HZOBHSH081FUPOTIIPA', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.qrcodescanner.barcodescanner-screens/screenshot_4.png'],
    ['3X52SWXE0XSG6UMICTIHI64Q44BCW9', 'A5N4AMMLOVZNS',  GEMMA_PREFIX + 'com.astrapaging.vff-screens/screenshot_5.png'],
    ['3Z8UJEJOCZ0HHYDRQPXOBJSL0QE39D', 'A2VNBZ7P595W77', GEMMA_PREFIX + 'com.tplink.skylight-screens/screenshot_2.png'],
    ['3H5TOKO3D96FHBUXSWZV1ETPO6S46C', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.funreality.software.nativefindmyiphone.lite-screens/screenshot_1.png'],
    ['3VJ4PFXFJ3UFLB0FXF7PUNT5J25UAE', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'scan.dtc.obd.readcode.elm327.oht.carsys-screens/screenshot_4.png'],
    ['3IVEC1GSLPMAD7CLPXAICKRRYS1J1J', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.froogloid.kring.google.zxing.client.android-screens/screenshot_3.png'],
    ['3PKVGQTFIH7O11619RQ0SYOSFU4RYH', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.gtp.nextlauncher.theme.magic-screens/screenshot_1.png'],
    ['33QQ60S6AS5JKY2X5DAW5HHSAGTU0N', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.abclocal.wabc.news-screens/screenshot_3.png'],
    ['3BAKUKE49HZRBVVYX2JREBDI89W1RU', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.codified.hipyard-screens/screenshot_6.png'],
    ['3M93N4X8HKAA7RBF2GP28LE0OSESJC', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.kmart.android-screens/screenshot_1.png'],
    ['31S7M7DAGGDHHHCUU165Y5NEU0XLTW', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.eggheadgames.logicproblems-screens/screenshot_6.png'],
    ['3MJ9GGZYO3T61ZRDEVXJO4YD0BM2A4', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.omniluxtrade.veganrecipes-screens/screenshot_6.png'],
    ['3D5G8J4N5ARKDMOO858BLELGLYZVTP', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.luxypro-screens/screenshot_3.png'],
    ['3L21G7IH47J08W7KBXBLM76FBWB1YW', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.couchgram.privacycall-screens/screenshot_6.png'],
    ['3TFJJUELSHCUUEOBY9TAUZO10E72C6', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.polyvore-screens/screenshot_1.png'],
    ['3ZQA3IO31BEOEI3I5A2NKG8DXMBO17', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.jb.gokeyboard.theme.ztlove2.getjar-screens/screenshot_1.png'],
    ['3PIOQ99R7Y9M5UU46JCUGTD5I94NUR', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'example.matharithmetics-screens/screenshot_1.png'],
    ['3CIS7GGG656IBOHRDK9BQ6JW4H3UE9', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.goodreads-screens/screenshot_1.png'],
    ['3FW4EL5A3LBHWWW91G44IMFFONU22X', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.codesector.speedview.free-screens/screenshot_5.png'],
    ['39WICJI5ATFWNPV4UNRNO7UEFASZ3C', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.yinzcam.nfl.steelers-screens/screenshot_1.png'],
    ['3V8JSVE8YYDZSX1WNL60TRP905EYEB', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.fairfax.domain-screens/screenshot_3.png'],
    ['3RKHNXPHGWJLXO9196KEO1W1OEZUKT', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.androidlab.videoroad-screens/screenshot_5.png'],
    ['3P0I4CQYVYUHFJJL2YR8D6Y7TZKOW6', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.johnugwuadi.simplemacro-screens/screenshot_1.png'],
    ['39HYCOOPKO8U6AYI8N75TEKHRJZDM7', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.desertstorm.recipebook-screens/screenshot_4.png'],
    ['3I4E7AFQ2KMBMROURGUSIQAMQBSJTS', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.dotwdg.countrynoise-screens/screenshot_5.png'],
    ['34R0BODSP1M1Q9RCCJ13IURG50AE5B', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'jp.co.a_tm.android.plus_momo_gingham_check-screens/screenshot_1.png'],
    ['33QQ60S6AS5JKY2X5DAW5HHSAGU0UU', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'net.blastapp-screens/screenshot_3.png'],
    ['3OREP8RUT2Y1Z34CJOI4LJCKSCBBGQ', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.formasystems.fuelbook-screens/screenshot_1.png'],
    ['3JUDR1D0D6EYOFYU3RU5E4DI4T7Q2Y', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.wmur.android.weather-screens/screenshot_1.png'],
    ['33KGGVH24U4B8RHA61PJ3TZ69QGX1L', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'mobi.infolife.ezweather.widget.milky-screens/screenshot_1.png'],
    ['3RIHDBQ1NELBIQ3KUJQVKZZFJBSMHP', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.livesoccertv-screens/screenshot_6.png'],
    ['3FO95NVK5CNR5FUSALYBAC0RY98SR4', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.andromo.dev439549.app477615-screens/screenshot_1.png'],
    ['3KA7IJSNW6S7CGXBM78RFSQCT3YPB8', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mywebgrocer.ShopRiteCircular-screens/screenshot_6.png'],
    ['31GN6YMHLPFWDBBE9F8HXB88QONSW7', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.ba.mobile-screens/screenshot_2.png'],
    ['37SOB9Z0SSKCI0E0FM0EGSJG01L3LY', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mineworld.pvpskins-screens/screenshot_3.png'],
    ['3ZUE82NE0AOCJ5AA5SSCCWM7JRQ8FP', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.hushed.release-screens/screenshot_1.png'],
    ['38DCH97KHHPYUBW2AC0XDLZZ64TJQE', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mobilligy.android-screens/screenshot_1.png'],
    ['3AFT28WXLFPZEOZGDHNTCFPZEOMOIZ', 'A383I2LLYX9LJM', GEMMA_PREFIX + 'com.yahoo.mobile.client.android.sportacular-screens/screenshot_5.png'],
    ['301KG0KX9C8LBMZRK17B5QLHD6C2HF', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.chatbooks-screens/screenshot_5.png'],
    ['3IV1AEQ4DR0SO7W6OIJXRUEZCY7J83', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.coloring.book.animals-screens/screenshot_3.png'],
    ['3MJ28H2Y1EVN20416X4ZN6LU92GO5P', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.sony.tvsideview.phone-screens/screenshot_5.png'],
    ['3HYV4299H0JKDRAEAH86UE177QBE8D', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.sony.tvsideview.phone-screens/screenshot_2.png'],
    ['3XU80RHWHZ312OQJ986G4G9BSFM44R', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.faithcomesbyhearing.android.bibleis-screens/screenshot_4.png'],
    ['3OWZNK3RYLCQG85BK0EAO5A77EDU27', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.squatschallenge-screens/screenshot_2.png'],
    ['3LN3BXKGC0ITN62NEFE08XP52DTGWB', 'AK7HDEEL3I05V', GEMMA_PREFIX + 'sw.tytdroid-screens/screenshot_1.png'],
    ['32ZCLEW0BZ7DUZVNAZL8DG9P187PJT', 'A2SC7SYQ22RJBR', GEMMA_PREFIX + 'com.fitnesschallenges-screens/screenshot_2.png'],
    ['34OWYT6U3W4W71VABS4BFPUAJP19I7', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.memory.brain.training.games-screens/screenshot_1.png'],
    ['3X4Q1O9UBH92P4IKUQJWLQZQ08BO74', 'A2SC7SYQ22RJBR', GEMMA_PREFIX + 'mobi.supo.optimizer-screens/screenshot_1.png'],
    ['33N1S8XHHM80DMHP7JD40EFHHQF1ZN', 'A1S4F2L2O88XE4', GEMMA_PREFIX + 'com.oldguide.inter.tipstekken-screens/screenshot_4.png'],
    ['34F34TZU7WMH2SUH81AWH31WQYFJ20', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.airgoat.goat-screens/screenshot_6.png'],
    ['3NFWQRSHVE1RCKGSOQ4049357A3GFD', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.wevideo.mobile.android-screens/screenshot_3.png'],
    ['36FQTHX3Z3E05RSOO9BK2PNBO76B3I', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.InspirationalBibleQuotesWallpapers.apps-screens/screenshot_2.png'],
    ['3G9UA71JVVHOOTRNEDRRDUI41E27J5', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.baritastic.view-screens/screenshot_2.png'],
    ['3V0TR1NRVAPG4D60I9G7HJVBKPI4A2', 'A4KV3O5TLCMQV', GEMMA_PREFIX + 'com.amour.chicme-screens/screenshot_4.png'],
    ['33K3E8REWWITJR1V5MYYI3MENWMX85', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.nikon.spoton-screens/screenshot_6.png'],
    ['3IKMEYR0LWICDC0ZQMA8II195NG2KJ', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.easymobs.pregnancy-screens/screenshot_4.png'],
    ['3LN3BXKGC0ITN62NEFE08XP52DTWGR', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.j2.myfax-screens/screenshot_2.png'],
    ['3S829FDFT2O50MXMMU9E4V0FYNJDXR', 'A2COCSUGZV28X', GEMMA_PREFIX + 'com.days30.absworkout-screens/screenshot_2.png'],
    ['3KTZHH2ONI2IPF2RRFWP7M6GGS2M8A', 'A3DS9DP2JE8I4Z', GEMMA_PREFIX + 'tk.alexapp.freestuffandcoupons-screens/screenshot_1.png'],
    ['371DNNCG44PM2ASB0CX1T018A0Y8TX', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.microsoft.bing-screens/screenshot_5.png'],
    ['3YCT0L9OMMW6QH20HEHU1SVHXCJSNG', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.libiitech.mermaidsalon-screens/screenshot_2.png'],
    ['31HLTCK4BLIG8H2IO594MV6RV4RGV1', 'A2COCSUGZV28X', GEMMA_PREFIX + 'com.ovuline.fertility-screens/screenshot_1.png'],
    ['3X55NP42EO329LTY7ETLX9S05BLP3H', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.enjoy.app.cartoon-screens/screenshot_3.png'],
    ['3MDWE879UHPMKTH45ICUT2QXOZH9BY', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.pof.android-screens/screenshot_1.png'],
    ['3DGDV62G7OWDO3XF3EFGEZN8ZXLP26', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.gau.go.weatherex.theme.gowidget.purplenightskin-screens/screenshot_1.png'],
    ['3IWA71V4TI36FDI7C710YPQNH1X6XT', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'proffesionals.dog.whistle.cat.repelent-screens/screenshot_1.png'],
    ['3P4ZBJFX2VQN966S3V8IQCPT1YTWFS', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'prank.agescanner.fake-screens/screenshot_1.png'],
    ['3K8CQCU3KEOZXYJ91JRVYHY23PFWNZ', 'A3CPGN8DF2Z85S', GEMMA_PREFIX + 'com.logopit.logoplus-screens/screenshot_2.png'],
    ['3KWGG5KP6JPK1IS48RCU982Z6JNMCG', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.Girly.eddymnewspictures-screens/screenshot_2.png'],
    ['39TX062QX1B7IUVPT532RME37AUX3D', 'A3JTZB1TE9EG4R', GEMMA_PREFIX + 'com.dominospizza-screens/screenshot_5.png'],
    ['3XD2A6FGFNHL1FY6G4YPJR0XSN39SZ', 'A3DS9DP2JE8I4Z', GEMMA_PREFIX + 'com.nanoequipment.kfc-screens/screenshot_2.png'],
    ['3JHB4BPSFKW9OZJTO09KORLWAZI9QA', 'A3DS9DP2JE8I4Z', GEMMA_PREFIX + 'com.dccomics.comics-screens/screenshot_1.png'],
    ['39AYGO6AFF713J43A1ER0NZPFNN6N9', 'A3DS9DP2JE8I4Z', GEMMA_PREFIX + 'tr.com.ea.a.a.mm-screens/screenshot_2.png'],
    ['3ZICQFRS315X8I2XFUMWS8ZTU5RZZ9', 'A9J8FIAW2DA7Y', GEMMA_PREFIX + 'com.dailydevotionapp-screens/screenshot_2.png'],
    ['338431Z1FL2C7N2Z34H2Q7QQ1K5OR3', 'A9J8FIAW2DA7Y', GEMMA_PREFIX + 'com.dianxinos.dxbs-screens/screenshot_5.png'],
    ['37MQ8Z1JQEJ7XHPGU1BZRQGK32P2YE', 'AZ2ISX1NXM4AL', GEMMA_PREFIX + 'com.instructure.candroid-screens/screenshot_1.png'],
    ['3IWA71V4TI36FDI7C710YPQNH1WX6J', 'A3IM1NBW1G1TCU', GEMMA_PREFIX + 'com.Livewallpaper.Ceiling-screens/screenshot_5.png'],
    ['3FDWKV9VCNPGNC94UIXU3EO5DJUUMY', 'A3IM1NBW1G1TCU', GEMMA_PREFIX + 'com.leguide.lego.friends-screens/screenshot_2.png'],
    ['391FPZIE4C9UVY8T3LHHB9G9WXVUHP', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.google.android.apps.blogger-screens/screenshot_1.png'],
    ['367O8HRHKGVK49SUZ92Y2JPH6VOS4L', 'A2COCSUGZV28X', GEMMA_PREFIX + 'com.pocketsupernova.pocketvideo-screens/screenshot_6.png'],
    ['31GN6YMHLPFWDBBE9F8HXB88QONWSB', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.mogulsoftware.android.BackPageCruiserSafe-screens/screenshot_2.png'],
    ['3LG268AV38TQVWPA9QE6LVWCXIZRE2', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.lmn.cnews-screens/screenshot_4.png'],
    ['3HXCEECSQMGX3SSSJ8KDE1QROBNZYI', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.joulespersecond.seattlebusbot-screens/screenshot_5.png'],
    ['35XW21VSVG1G2HZF511FO0RSSHJSLE', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'modernpenandpaper.com.fix21dayrecipes1-screens/screenshot_1.png'],
    ['3VEI3XUCZRKUSNE0I4UHF7VGBP1RPN', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.nilvav.certificatemaker-screens/screenshot_5.png'],
    ['366FYU4PTGC48SCFJ659KJAY2Q5EKW', 'A3GAVW48D7KGCN', GEMMA_PREFIX + 'mobi.infolife.ezweather.locker.fingerprint-screens/screenshot_1.png'],
    ['36BTXXLZ2VV83USHKK2TC9KLKT64RA', 'A3GAVW48D7KGCN', GEMMA_PREFIX + 'com.herzick.houseparty-screens/screenshot_1.png'],
    ['3SV8KD29L4F2JN3BFTJM5TBGQWEZKN', 'A3GAVW48D7KGCN', GEMMA_PREFIX + 'com.lfantasia.android.outworld-screens/screenshot_1.png'],
    ['30Y6N4AHYPJL3QBADV3MFDERX6YDRV', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.jb.gokeyboard.theme.dlglkeyboardnew2017-screens/screenshot_2.png'],
    ['3ULIZ0H1VAS268X00V6OBA8MGR751K', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.crowdcompass.appa3sdF2hjb1-screens/screenshot_1.png'],
    ['36AZSFEYZ4NKH0U78JHQLKFUFRZVBZ', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.flightradar24free-screens/screenshot_4.png'],
    ['338431Z1FL2C7N2Z34H2Q7QQ1K5RO6', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.kidscrape.king-screens/screenshot_5.png'],
    ['3K8CQCU3KEOZXYJ91JRVYHY23PFNWQ', 'A2COCSUGZV28X', GEMMA_PREFIX + 'com.appxy.tinyscanner-screens/screenshot_4.png'],
    ['31YWE12TE0ZPJDWCVH6S43QJCCH7XT', 'A2COCSUGZV28X', GEMMA_PREFIX + 'com.fiservcardvalet.mobile.android-screens/screenshot_5.png'],
    ['36MUZ9VAE6PWUM65RPJMG6F8IH8DEI', 'A2COCSUGZV28X', GEMMA_PREFIX + 'mobi.infolife.ezweather.widget.blackglass-screens/screenshot_1.png'],
    ['3WKGUBL7SZ9X0WX4F05XQXBME1H4LK', 'A2COCSUGZV28X', GEMMA_PREFIX + 'com.zrgiu.antivirus-screens/screenshot_5.png'],
    ['3ZTE0JGGCEF81LDGL7R71CR41J2OCX', 'A2X0RVFZQYDDWN', GEMMA_PREFIX + 'com.petsmart.consumermobile-screens/screenshot_5.png'],
    ['38G0E1M85MSZDZ3D8AYLQA7I4C6UVF', 'A2COCSUGZV28X', GEMMA_PREFIX + 'com.lavendrapp.lavendr-screens/screenshot_1.png'],
    ['3I01FDIL6MV4Z6UPY5OLE56JLEJ2D6', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.iexamguru.dmv_practice_test-screens/screenshot_1.png'],
    ['3V7ICJJAZA3LNNBSONG3CUFH2734BX', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.aopeng.fashionmia-screens/screenshot_3.png'],
    ['31ANT7FQN8PDAJIE2K993XOTXG2H5U', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.gloto.telemundo-screens/screenshot_1.png'],
    ['3X52SWXE0XSG6UMICTIHI64Q4L1WCH', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.pteam.camera-screens/screenshot_3.png'],
    ['38XPGNCKHTN0W19YT473D69O5ZIV4T', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.omniluxtrade.veganrecipes-screens/screenshot_2.png'],
    ['33KGGVH24U4B8RHA61PJ3TZ69QH1XQ', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.ui.LapseIt-screens/screenshot_1.png'],
    ['3IVKZBIBJ0WEGLGPEGMHW8YLB0ASH7', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'app.calleridfaker.com-screens/screenshot_2.png'],
    ['30ZKOOGW2WTZCE9HVRX5JLO8858A1V', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.azwstudios.theholybible.em-screens/screenshot_4.png'],
    ['3RWO3EJELHW6ZIFJKAF29GD37R81P2', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.crazygame.inputmethod.keyboard7-screens/screenshot_1.png'],
    ['36U4VBVNQO07KSMWZQTBPDIFJ6TURM', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.Adanel.QC-screens/screenshot_3.png'],
    ['3QGTX7BCHPPZ14I886FG34SYX1P5Z8', 'A1LJGHG6RYGLMB', GEMMA_PREFIX + 'com.mi.AthleanX-screens/screenshot_1.png'],
    ['351S7I5UG9JDREJAUK8G9R4U6V6NJ4', 'A3TUJHF9LW3M8N', GEMMA_PREFIX + 'com.nilvav.certificatemaker-screens/screenshot_1.png'],
    ['3Q2T3FD0ONVWOIWLFY1TG5Y558A3M8', 'A3CB7509BB86ZX', GEMMA_PREFIX + 'com.mobileguru.candyfever2.free-screens/screenshot_1.png'],
    ['3Z8UJEJOCZ0HHYDRQPXOBJSL593399', 'A3CB7509BB86ZX', GEMMA_PREFIX + 'dil.soup_recipe-screens/screenshot_2.png'],
    ['37VHPF5VYCQ5GTQ0EQ2B4Q14RGY8CN', 'A3CB7509BB86ZX', GEMMA_PREFIX + 'com.gogii.textplus-screens/screenshot_1.png'],
    ['3HEADTGN2PF7X7BW4G3GLZKY2F1RVG', 'A3CB7509BB86ZX', GEMMA_PREFIX + 'com.fi7227.godough-screens/screenshot_4.png'],
    ['3B9J25CZ250NZD59SO8GLH153A5CSR', 'AVETOUO04K4I4', GEMMA_PREFIX + 'com.microsoft.office.outlook-screens/screenshot_2.png'],
    ['3QGTX7BCHPPZ14I886FG34SYY8J5ZH', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'org.lds.ldsmusic-screens/screenshot_2.png'],
    ['3O0M2G5VC6P2ZUWVQ327LOKRJ9Z940', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.faithcomesbyhearing.android.bibleis-screens/screenshot_5.png'],
    ['3XABXM4AJ1S9AU0JJ836OVQLVVZ8QS', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.guardian-screens/screenshot_6.png'],
    ['3RDTX9JRTYOZIHEES10EO041ABP97K', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.azwstudios.theholybible.kjv-screens/screenshot_6.png'],
    ['3EPG8DX9LKD5N0G2LXHEC6QKO9C5PW', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.aries.software.california_dmv_test-screens/screenshot_3.png'],
    ['3DTJ4WT8BD2ZXNB1J78J7YYBB1IEZR', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.mcproteam.videoplayer-screens/screenshot_1.png'],
    ['391JB9X4ZYV2KG4S04TSXRF7GUVMK1', 'AVETOUO04K4I4', GEMMA_PREFIX + 'com.gamoper.fruitjourney.free-screens/screenshot_1.png'],
    ['3LEG2HW4UFA1XQ8LPBJB25942TYF22', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.lifelyus.android-screens/screenshot_3.png'],
    ['3ZTE0JGGCEF81LDGL7R71CR46L2COU', 'A39KVTVXJ5FU61', GEMMA_PREFIX + 'buscar.pareja-screens/screenshot_4.png'],
    ['3W31J70BASJPBUS1NFZEUVAVMGSCKA', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.outfit7.talkingpierrefree-screens/screenshot_1.png'],
    ['31MBOZ6PAOE0V0AJ8FBF9VNABGXLCM', 'AVETOUO04K4I4', GEMMA_PREFIX + 'com.apalon.ringtones-screens/screenshot_3.png'],
    ['367O8HRHKGVK49SUZ92Y2JPHBXNS4T', 'AVETOUO04K4I4', GEMMA_PREFIX + 'co.triller.droid-screens/screenshot_1.png'],
    ['3ZXV7Q5FJBBEV80NM48HEIIQEKBCF8', 'A27BYNQVXYIC7H', GEMMA_PREFIX + 'wp.wattpad.covers-screens/screenshot_2.png'],
    ['360ZO6N6J16I3KL1W7237I1R50JM97', 'A18R117CBYN9GN', GEMMA_PREFIX + 'com.Hypnosis-screens/screenshot_3.png'],
    ['33J5JKFMK6LN9XUD7R8AXEIAF97Q3I', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.sgiggle.production-screens/screenshot_1.png'],
    ['3S1L4CQSFXSQ2T3P2QCQ8NS2EHQAF4', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.paycor.perform-screens/screenshot_2.png'],
    ['3ZZAYRN1I6EPN2FR7TMXQMR8U72TOU', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.jb.gokeyboard.theme.ztlove2.getjar-screens/screenshot_1.png'],
    ['3LOJFQ4BOX2ZGMYFNBUPE1SD7G4DKB', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'mobi.infolife.ezweather.widget.sense3style-screens/screenshot_1.png'],
    ['391FPZIE4C9UVY8T3LHHB9G91ZVHUL', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.akasoft.topplaces-screens/screenshot_2.png'],
    ['3QREJ3J433KIEYM70SSDXE0IGP6KLY', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.riatech.salads-screens/screenshot_1.png'],
    ['3C8QQOM6JPOR83WJ0P71KYL5OLHLIO', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.gloto.telemundo-screens/screenshot_2.png'],
    ['3QMELQS6Y5YMIHWV38V8974ZLMBR6Q', 'A1EZRTSWXP7LJJ', GEMMA_PREFIX + 'com.urbandroid.lux-screens/screenshot_5.png'],
    ['3HO4MYYR12BEY9OC72QDEIRJSLFU6X', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.waplog.social-screens/screenshot_2.png'],
    ['3YCT0L9OMMW6QH20HEHU1SVH2EJNSK', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.adianteventures.converters.usd2mxn-screens/screenshot_2.png'],
    ['3UQ1LLR26AVC2LDLO1FO30XMBDCAL9', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.green.banana.app.lockscreenpassword-screens/screenshot_3.png'],
    ['37PGLWGSJTTGOXE1FPV05Y8QZLCIKI', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.rms.gamesforkids.painting.cars-screens/screenshot_4.png'],
    ['3P520RYKCHTIF5OY2JG8MCDJWQ45US', 'A330E8HHBLEGMP', GEMMA_PREFIX + 'com.irisstudio.photomixer-screens/screenshot_1.png'],
    ['3IZPORCT1FW2F6GCZRMC6YTPU0RHRH', 'A1EZRTSWXP7LJJ', GEMMA_PREFIX + 'me.meecha-screens/screenshot_1.png'],
    ['3W9XHF7WGKI6XBUC91U4J5AZA2SKT1', 'ALJBVO9XK6O99', GEMMA_PREFIX + 'com.guidebook.apps.PAXEast2016.android-screens/screenshot_5.png'],
    ['32K26U12DNBDWXSRMF8WGA3UNEBDVX', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.dwgsee.dwgviewer-screens/screenshot_3.png'],
    ['3DZKABX2ZIS3R5MDMJDLW4LC2EYVCE', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.kiddoware.kidspictureviewer-screens/screenshot_1.png'],
    ['3OKP4QVBP2KWGTZKDAI8Q8SEFWQGA9', 'A2ECRNQ3X5LEXD', GEMMA_PREFIX + 'com.g8n8.pregnancytracker-screens/screenshot_6.png'],
    ['3IQ9O0AYW6MFRG4O9Y9S6PVMIY3ITO', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.goodwill.app-screens/screenshot_1.png'],
    ['3421H3BM9A4S2CFGTAPBRQ9A16PJ91', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.sevencsolutions.babysleeper-screens/screenshot_2.png'],
    ['36AZSFEYZ4NKH0U78JHQLKFUKTYVB7', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'ee.mtakso.client-screens/screenshot_3.png'],
    ['3KLL7H3EGDOU8DXT8BRM7VISF64HVA', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'samsungupdate.com-screens/screenshot_4.png'],
    ['35O6H0UNLS391PD4QKCUFA48FA6J5A', 'A2ECRNQ3X5LEXD', GEMMA_PREFIX + 'love.swan.bird.colorful-screens/screenshot_1.png'],
    ['3VJ4PFXFJ3UFLB0FXF7PUNT5O45UAN', 'A2ECRNQ3X5LEXD', GEMMA_PREFIX + 'com.omniluxtrade.paleorecipes-screens/screenshot_2.png'],
    ['35XW21VSVG1G2HZF511FO0RSXJJLSG', 'A2ECRNQ3X5LEXD', GEMMA_PREFIX + 'ali.alhadidi.gif_facebook-screens/screenshot_2.png'],
    ['3XUY87HIVP1XA44VLG68C47B5ZQMM6', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.hotwire.hotels-screens/screenshot_3.png'],
    ['3G5RUKN2ECQOL26188H3KXQRAQUN99', 'AH56J7I291XL7', GEMMA_PREFIX + 'dailybibleverse.bible.verse-screens/screenshot_6.png'],
    ['3IYI9285WSNU0AMJ5UZ2TGAI7RSJCI', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.vervewireless.droid.foxwttg-screens/screenshot_6.png'],
    ['338GLSUI43YU2PPJJQYHTNM80NWSFH', 'A1MFR8AR96VI0J', GEMMA_PREFIX + 'com.talixa.pocketherbalist-screens/screenshot_4.png'],
    ['3J6BHNX0U9F82YPSC58SA647LK6KN0', 'AY4B7SUM68J3G', GEMMA_PREFIX + 'com.itcode.reader-screens/screenshot_4.png'],
    ['363A7XIFV49FYQPF25HUQ9VO2BEAVS', 'A1M9NT8B1KNVZK', GEMMA_PREFIX + 'com.fashiongeek.mehndidesigns2016-screens/screenshot_4.png'],
    ['3U74KRR67M875HFF6EMKBXR34XXNT9', 'A346Y4O6PXRWWG', GEMMA_PREFIX + 'com.instaeditor.cartoonavtar-screens/screenshot_5.png'],
    ['3DZKABX2ZIS3R5MDMJDLW4LC2EZCVW', 'AY4B7SUM68J3G', GEMMA_PREFIX + 'com.eliferun.music-screens/screenshot_2.png'],
    ['37Y5RYYI0PSB2BG4JK43ZUMF4STSX9', 'AY4B7SUM68J3G', GEMMA_PREFIX + 'com.ncaa.mmlive.app-screens/screenshot_3.png'],
    ['3E22YV8GG1T9DPM8PXD06NQ6FUCNP4', 'A1MFR8AR96VI0J', GEMMA_PREFIX + 'com.livemapsguide.mapliveguide-screens/screenshot_1.png'],
    ['3H1C3QRA016MPU0RRTOBW0XQ2JWECS', 'A299AYFO0RJGDI', GEMMA_PREFIX + 'com.bbt.myfi-screens/screenshot_1.png'],
    ['3XDJY5RK5S80JMKYOEXYHK75WU9U4C', 'A299AYFO0RJGDI', GEMMA_PREFIX + 'com.togotechnologies.RoyalFarms4-screens/screenshot_2.png'],
    ['3CVDZS288HNR4UCEZT2VHEN28YEMFZ', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.lyrebirdstudio.art_filter-screens/screenshot_5.png'],
    ['31ANT7FQN8PDAJIE2K993XOT2I2H53', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.andromo.dev528355.app507036-screens/screenshot_2.png'],
    ['306W7JMRYYLM6OYU0Q6GT6VM9VYB8J', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.modoohut.dialer-screens/screenshot_3.png'],
    ['32XN26MTXZ6G18F3WZIVEUL92R00L8', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.ImaginationUnlimited.PotoPro-screens/screenshot_1.png'],
    ['3GMLHYZ0LEKGHTYIDLL5X9BCRV8UYX', 'A2VDF42IP5PE5O', GEMMA_PREFIX + 'com.hp.babynames-screens/screenshot_2.png'],
    ['3P4C70TRMR4DCCQOA17YZM8P98PLGW', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'au.org.nps.medicinelistplus-screens/screenshot_3.png'],
    ['3S1WOPCJFGG9X86X1L5XJ4ALS35JEB', 'A1S8DYWNS59XWB', GEMMA_PREFIX + 'com.idmobile.usameteo-screens/screenshot_4.png'],
    ['3T2HW4QDUVU5UY9AJI8P4MWFBMGC9G', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.instagram.android-screens/screenshot_3.png'],
    ['3UAU495MIIF4NFUG7YC7VIDADL9UOM', 'A1MFR8AR96VI0J', GEMMA_PREFIX + 'com.urbandictionary.android-screens/screenshot_1.png'],
    ['3J5XXLQDHMYFE5QUTQ2K31HIW1MV3H', 'A2VDF42IP5PE5O', GEMMA_PREFIX + 'fun.kids.drawingfnafsisterlocation-screens/screenshot_1.png'],
    ['3UUIU9GZC5S3FS992EQYHGRHDCXT5P', 'A1MFR8AR96VI0J', GEMMA_PREFIX + 'com.app_awireless.layout-screens/screenshot_5.png'],
    ['38G0E1M85MSZDZ3D8AYLQA7I9E6UVO', 'A2VDF42IP5PE5O', GEMMA_PREFIX + 'com.carfax.mycarfax-screens/screenshot_3.png'],
    ['3VW0145YLYZ79WYAIJTGWCFU15WJM0', 'A2VDF42IP5PE5O', GEMMA_PREFIX + 'com.instatrendyyy-screens/screenshot_6.png'],
    ['3H5TOKO3D96FHBUXSWZV1ETPNV3460', 'A1P7U2ULSAE7YL', GEMMA_PREFIX + 'com.campbellskitchen.android-screens/screenshot_1.png'],
    ['3SBX2M1TKDA8RTDH75DT7JAGOJE4Q7', 'A9ADJ4VENB2JX',  GEMMA_PREFIX + 'com.barilab.handmirror.googlemarket-screens/screenshot_3.png'],
    ['3PIOQ99R7Y9M5UU46JCUGTD5NB4UN7', 'A18R117CBYN9GN', GEMMA_PREFIX + 'seesaw.shadowpuppet.co.seesaw-screens/screenshot_2.png'],
    ['359AP8GAGG71GFLH4LA5QQ59GWSC74', 'A18R117CBYN9GN', GEMMA_PREFIX + 'com.andromo.dev589470.app571471-screens/screenshot_2.png'],
    ['39XCQ6V3KYRXDUYX61I566Z3T4M65R', 'A18R117CBYN9GN', GEMMA_PREFIX + 'com.fitness22.workout-screens/screenshot_5.png'],
    ['3MZ3TAMYTLA2B1RW594X6AU4UD6RIK', 'A18R117CBYN9GN', GEMMA_PREFIX + 'com.pocketpoints.pocketpoints-screens/screenshot_3.png'],
    ['3ZQX1VYFTDS6PIN34VB2ZQVLGUHO80', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.rsoftr.android.earthquakestracker.add-screens/screenshot_3.png'],
    ['33TGB4G0LP4CHBTJ8K9T9ZI16B7TXZ', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.champssports.champssports-screens/screenshot_4.png'],
    ['3QTFNPMJC653RTOEC6B2XLSKPTJNZP', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.rubenmayayo.reddit-screens/screenshot_2.png'],
    ['371QPA24C2B4RA8Q1ROMEQE01WST17', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.topvideo.cutecut-screens/screenshot_1.png'],
    ['34R0BODSP1M1Q9RCCJ13IURGA2AE5K', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'me.tombailey.skinsforminecraftpe-screens/screenshot_2.png'],
    ['3126F2F5F8Q5JJ9VWYAHG0WX12BEP6', 'A3C4WN7F2VKC8B', GEMMA_PREFIX + 'com.mathrawk.privatetexting-screens/screenshot_3.png'],
    ['3WRBLBQ2GRV0FG12L8LF12G16AEG0I', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'com.jiubang.fastestflashlight-screens/screenshot_5.png'],
    ['39N6W9XWRDAXCBXNTGWOTUALINTYG1', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'com.ridecharge.android.taximagic-screens/screenshot_1.png'],
    ['3EN4YVUOUCFI38XEWQVP7TVS89GXJY', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'xyz.markapp.renthouse-screens/screenshot_3.png'],
    ['3BVS8WK9Q0IFVUA79CI76VP0CR3IBY', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'com.estmob.android.sendanywhere-screens/screenshot_4.png'],
    ['3VO4XFFP1595AGV093B0AVUQKBTQ7N', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'com.nomanprojects.mycartracks-screens/screenshot_5.png'],
    ['3ZURAPD288AU85QP67JXXMZZANKF1L', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'com.gotv.crackle.handset-screens/screenshot_2.png'],
    ['33NKDW9FFX5VVUR8MYE3KPIXIP7XCK', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'com.ew.coloring.princess-screens/screenshot_2.png'],
    ['3EKZL9T8Y89Y94RTHUWHH3U3UZOCH6', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'com.musicplayer.music-screens/screenshot_2.png'],
    ['3UEBBGULPFBAH2HN8VTH8O4WLKIUF0', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'show.wifi.password.hienthi.matkhau.wifi.chua.ketnoi-screens/screenshot_2.png'],
    ['3S37Y8CWI8NDBQ93JECZZ8S2HAKW4D', 'A1DHNW63TISLN0', GEMMA_PREFIX + 'com.totaltrivia-screens/screenshot_2.png'],
    ['3SR6AEG6W5GBC7SYDME6EUMMXN6YH0', 'A1EJ4HA4K85PBJ', GEMMA_PREFIX + 'com.etrade.mobilepro.activity-screens/screenshot_3.png'],
    ['335VBRURDJNKRWTOF9EKUAWIWYU9E7', 'A1MFR8AR96VI0J', GEMMA_PREFIX + 'com.myfitnesspal.android-screens/screenshot_1.png'],
    ['3NKW03WTLMUK0VW1HBJC2BT7D22WQV', 'A29Y7MCUQWQHRI', GEMMA_PREFIX + 'com.punchh.grubburger-screens/screenshot_1.png'],
    ['3X52SWXE0XSG6UMICTIHI64Q4XBWCF', 'A3RPNKORMJ2PWO', GEMMA_PREFIX + 'com.twothumbsapp.futbolLigaMexicana-screens/screenshot_4.png'],
    ['3B6F54KMR2Z3CAVBCRJW4LIEUNPS17', 'A330UTE0AHOQ2B', GEMMA_PREFIX + 'com.garmin.android.apps.phonelink-screens/screenshot_2.png'],
    ['3WGCNLZJKFVXALCTF1O79MWETKFD1Y', 'A1K4QLD3R5Z0B6', GEMMA_PREFIX + 'com.rinrada.android.mustang-screens/screenshot_5.png'],
    ['3MIVREZQVHLT5V2KSX09E0ZCKBOQKE', 'AUH4A0OGHF2CK', GEMMA_PREFIX + 'com.p1.chompsms-screens/screenshot_1.png'],
    ['3E9ZFLPWOYFM8XD8E8APVSGSPAMIXS', 'ACWU6HLK6HSTB', GEMMA_PREFIX + 'com.accuweather.android-screens/screenshot_5.png'],
    ['3UZUVSO3P7IZYGCI4IIPZW22RV0ME9', 'A2ZLWXY1QIOT1T', GEMMA_PREFIX + 'com.comcast.cvs.android-screens/screenshot_1.png'],
    ['3HRWUH63QUP5KF4P5I4M8HLPJYBN5L', 'A3JNZX2U13B12Q', GEMMA_PREFIX + 'com.energizedsw.flashcardmathfree-screens/screenshot_6.png'],
    ['3ACRLU860N13FP8LDTSGLGR8F2WEBO', 'A35OXR5UZO3TE8', GEMMA_PREFIX + 'com.devotional.collection-screens/screenshot_3.png'],
    ['3L2OEKSTW9XIJWR5AIK01HTNP8G8YY', 'A1SDC3D4CEEUC4', GEMMA_PREFIX + 'com.timeanddate.countdown-screens/screenshot_4.png'],
    ['31GECDVA9J9TWYYBBKSAYCD3K3Y66B', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.recetasgratisnet.recetasmexicanas-screens/screenshot_1.png'],
    ['3CESM1J3EIQRQDDH225EW6CG7O26WO', 'A3MK12K0EAQITB', GEMMA_PREFIX + 'com.ivideon.client-screens/screenshot_4.png'],
    ['3FDWKV9VCNPGNC94UIXU3EO5D22UM8', 'A3COB27QFIGEN', GEMMA_PREFIX + 'org.dayup.stocks-screens/screenshot_5.png'],
    ['3ZUE82NE0AOCJ5AA5SSCCWM7JAZ8F0', 'A5N4AMMLOVZNS', GEMMA_PREFIX + 'com.waplogmatch.social-screens/screenshot_2.png'],
    ['36QZ6V15890JL7M9EFTGFNNB5RKSUI', 'A5N4AMMLOVZNS', GEMMA_PREFIX + 'com.monster.android.Views-screens/screenshot_5.png'],
    ['3IVEC1GSLPMAD7CLPXAICKRRYB9J1T', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.autometer.DashLink-screens/screenshot_1.png'],
    ['34XASH8KLQ93V718DWY0T8165LAPM7', 'A2HC9OWINJFDEZ', GEMMA_PREFIX + 'riddle.me.that.riddles.logo.quiz.icomania-screens/screenshot_2.png'],
    ['31MCUE39BK9WW80Z4V3Y30E8436G3E', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.meijer.mobile.meijer-screens/screenshot_3.png'],
    ['3HEA4ZVWVD91UQYJ1I80E0L3SQK55C', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.comcast.cvs.android-screens/screenshot_1.png'],
    ['3UY4PIS8QR86WX364V2A5R887VFN12', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.visualcv.app-screens/screenshot_1.png'],
    ['3OPLMF3EU5AJM47AX5KTP6HM51JLN5', 'A685WMI1QRZHE', GEMMA_PREFIX + 'com.sad.mimediamanzana-screens/screenshot_3.png'],
    ['3AFT28WXLFPZEOZGDHNTCFPZE7UIO3', 'A2CE4O5C1UBFTC', GEMMA_PREFIX + 'com.musictogether.hello_everybody-screens/screenshot_1.png'],
    ['32XN26MTXZ6G18F3WZIVEUL9X890LA', 'A6YCO8SLGR8K7', GEMMA_PREFIX + 'com.problemio-screens/screenshot_5.png'],
    ['3FBEFUUYRKSUJ0AE1X9MEHFPGZWA63', 'AS8EPK2ZG96VG', GEMMA_PREFIX + 'com.whereismycar-screens/screenshot_3.png'],
    ['3O71U79SRBC08ZH05D2UOD6HS5IMSL', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.Smartemoji.ColorEmoji-screens/screenshot_1.png'],
    ['3XH7ZM9YX2H900YT1FYKTHV164S9RL', 'A685WMI1QRZHE', GEMMA_PREFIX + 'com.otakuclub.animegirls-screens/screenshot_1.png'],
    ['3BDORL6HKK0BYY5WDZ8BV7W9CPVCR5', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.autozone.mobile-screens/screenshot_1.png'],
    ['3SCKNODZ0X3K7JI8TBJJ61ZLAHF7NJ', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.facebook.Socal-screens/screenshot_1.png'],
    ['3G57RS03HHS9VTFWJZ62GTIR8GT52B', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.cricut.designspace-screens/screenshot_4.png'],
    ['3ZUE82NE0AOCJ5AA5SSCCWM7JA08F1', 'A5N4AMMLOVZNS', GEMMA_PREFIX + 'net.metapps.meditationsounds-screens/screenshot_3.png'],
    ['338GLSUI43YU2PPJJQYHTNM8V46SFK', 'AELOP89FTAQRW', GEMMA_PREFIX + 'com.kellerwilliams.eventapp-screens/screenshot_2.png'],
    ['388FBO7JZRG3M3E9GK9JJEJIEGONY0', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.doapps.android.mln.MLN_40dfe94a4610c0022409deeeaf622414-screens/screenshot_2.png'],
    ['3V0TR1NRVAPG4D60I9G7HJVBK8QA4I', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.mingle.justsayhi-screens/screenshot_2.png'],
    ['3H5TOKO3D96FHBUXSWZV1ETPOP164P', 'A3HLJ37GGU8AGN', GEMMA_PREFIX + 'com.cnn.mobile.android.phone-screens/screenshot_5.png'],
    ['33K3E8REWWITJR1V5MYYI3MENFW8XS', 'ACW4OZL1VFTK8', GEMMA_PREFIX + 'tv.picpac-screens/screenshot_2.png'],
    ['3LN3BXKGC0ITN62NEFE08XP52W3GWN', 'ASUR9I0R9LRCO', GEMMA_PREFIX + 'videoplayer.mediaplayer.hdplayer-screens/screenshot_2.png'],
    ['31S7M7DAGGDHHHCUU165Y5NEUJ6LT7', 'A685WMI1QRZHE', GEMMA_PREFIX + 'com.ivideon.client-screens/screenshot_4.png'],
    ['3QO7EE372OASCSYQMKNGFXGMYIUBQX', 'A6MAGHAZSDMT0', GEMMA_PREFIX + 'audio.mp3.music.player-screens/screenshot_4.png'],
    ['3IH9TRB0FBMEIWSF7FPI0JCRLWMI1I', 'A5N4AMMLOVZNS', GEMMA_PREFIX + 'net.babygender.predictor-screens/screenshot_4.png'],
    ['3G57RS03HHS9VTFWJZ62GTIR8GU52C', 'A5N4AMMLOVZNS', GEMMA_PREFIX + 'com.carnival.android-screens/screenshot_2.png'],
    ['3N7PQ0KLI5CYCU48Y0DA3XTWRNU3EN', 'AWHJJ1YYPK65H', GEMMA_PREFIX + 'com.siplay.tourneymachine_android-screens/screenshot_4.png'],
    ['3NZ1E5QA6ZO3J6FS2SZDHMLV2M65BD', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.disneydigitalbooks.disneystorycentral_goo-screens/screenshot_1.png'],
    ['3UV0D2KX1M6B04N8TP2ZVK62LKEF47', 'A2ECRNQ3X5LEXD', GEMMA_PREFIX + 'com.fsm.audiodroid-screens/screenshot_3.png'],
    ['3KI0JD2ZU15C0YCGUUX24QDZFRQ769', 'A685WMI1QRZHE', GEMMA_PREFIX + 'com.anbu.ringtonemaker-screens/screenshot_3.png'],
    ['34O39PNDK6VSOTDQZZCGKNQRU4VRBD', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.homedesign.ideas-screens/screenshot_3.png'],
    ['3RWSQDNYL99LYKKXD6YIMO3T6EDFFP', 'A3RYJY5PJ0VXVJ', GEMMA_PREFIX + 'com.gohopscotch.android.ticketpop-screens/screenshot_4.png'],
    ['386659BNTL43B1BZ3P0CUFCVZ26107', 'A3RYJY5PJ0VXVJ', GEMMA_PREFIX + 'com.meridian.vdot-screens/screenshot_6.png'],
    ['3HXK2V1N4K27BQ4G0VPTKQ9GYI9G2S', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.scoompa.collagemaker.video-screens/screenshot_3.png'],
    ['3O71U79SRBC08ZH05D2UOD6HSYKMS9', 'A2VP4CMG2YJKB6', GEMMA_PREFIX + 'com.Livewallpaper.SimpleInteriorDesign-screens/screenshot_2.png'],
    ['3A520CCNWNNYWF3T7FSKQ572XQMAEC', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.MaMouer.lockscreen-screens/screenshot_2.png'],
    ['3DTJ4WT8BD2ZXNB1J78J7YYB6BSZE1', 'A1R5B8KT4EIIKB', GEMMA_PREFIX + 'com.simplifynowsoftware.pregnancysafetytips.free-screens/screenshot_5.png'],
    ['3BO3NEOQM04ACK5F3YIBBK5UPJTAIL', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.ubercab.eats-screens/screenshot_1.png'],
    ['3EKTG13IZUQD4MMAL45TCRSM04CMLV', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.innovative.logomaker.free.design-screens/screenshot_1.png'],
    ['3X52SWXE0XSG6UMICTIHI64Q4XCCWW', 'A1K4QLD3R5Z0B6', GEMMA_PREFIX + 'com.gpsnav.evo.gps2-screens/screenshot_6.png'],
    ['3YCT0L9OMMW6QH20HEHU1SVHXOUSNF', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.amour.chicme-screens/screenshot_4.png'],
    ['334ZEL5JX62O822CITH7GHVMNYZSOU', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.eapp.lock_wallpapers_free-screens/screenshot_3.png'],
    ['30P8I9JKOI8ISKDY79QDIYF257OV5X', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'co.ucvr.ultracover-screens/screenshot_1.png'],
    ['3AQN9REUTF3U0RNWRQVGN97O05OYDH', 'A1K4QLD3R5Z0B6', GEMMA_PREFIX + 'com.dinixe.eyesmakeups-screens/screenshot_2.png'],
    ['3G5RUKN2ECQOL26188H3KXQR505N9Z', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.olo.noodles-screens/screenshot_6.png'],
    ['3ROUCZ907FH9AAAJBC0YWCLLPFGOOO', 'A149PNINK87HZJ', GEMMA_PREFIX + 'com.airbnb.lottie-screens/screenshot_1.png'],
    ['3MD8CKRQZZAY6CB2NRPXIB9418QRJH', 'A9SZHZ6B4IMI1', GEMMA_PREFIX + 'com.moovn.rider-screens/screenshot_4.png'],
    ['3PKVGQTFIH7O11619RQ0SYOSECZRYB', 'A37FVCKAUOBI15', GEMMA_PREFIX + 'com.sears.android-screens/screenshot_1.png'],
    ['37SOB9Z0SSKCI0E0FM0EGSJG07KL3R', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.diycarrepairfasf-screens/screenshot_1.png'],
    ['39N6W9XWRDAXCBXNTGWOTUALDRUGYN', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.mathrawk.calleridfaker-screens/screenshot_2.png'],
    ['3L1EFR8WWTSCXATKAKYQCSHI339F9Y', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.ncaa.finalfourmens-screens/screenshot_4.png'],
    ['3BA7SXOG1JD9MVFJWNS6TL0QL4D8R9', 'A1S1K7134S2VUC', GEMMA_PREFIX + 'com.eapp.lock_wallpapers_free-screens/screenshot_3.png'],
    ['3B623HUYJ4DLNAFWBCSBJV5M76W8SX', 'A1S1K7134S2VUC', GEMMA_PREFIX + 'com.eapp.lock_wallpapers_free-screens/screenshot_1.png'],
    ['3MXX6RQ9EVSNRHC27SY47EK6I0T4P6', 'A3MK12K0EAQITB', GEMMA_PREFIX + 'com.atlogis.northamerica.free-screens/screenshot_1.png'],
    ['3D06DR52256W83V6ODG33DCB6ZBMAP', 'A176JIQ517D8UV', GEMMA_PREFIX + 'com.google.android.apps.ads.express-screens/screenshot_2.png'],
    ['3HEM8MA6H9ZUGMZ05P5HAHLBFM7PQU', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.stockmarket.pennystocks-screens/screenshot_3.png'],
    ['3Y3N5A7N4GWX4LT94JUJ6ZUI9QEMYZ', 'AATTSED5GB8Z4', GEMMA_PREFIX + 'com.hdmi.read-screens/screenshot_3.png'],
    ['3T8DUCXY0NTMGFBL543FTWWJUPOT9H', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.huffingtonpost.android-screens/screenshot_3.png'],
    ['3UYRNV2KITMO7XNR3GBPK1VGL1LN8M', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'yamayka.apps.BeautifulHairstyle-screens/screenshot_1.png'],
    ['39O0SQZVJNU5MHA0MI00FL4VBE47R7', 'AH56J7I291XL7', GEMMA_PREFIX + 'softin.my.fast.fitness-screens/screenshot_2.png'],
    ['33NKDW9FFX5VVUR8MYE3KPIXD6GCX1', 'A3OHY0XZ967UL1', GEMMA_PREFIX + 'com.jb.gokeyboard.theme.tmekeyboardred-screens/screenshot_2.png'],
    ['3XDSWAMB22FBMXQW0KJBQHM4G3TQCI', 'ATA9AZBKH2LNZ', GEMMA_PREFIX + 'com.gau.go.launcherex.theme.smokecolors.yang-screens/screenshot_4.png'],
    ['3IZPORCT1FW2F6GCZRMC6YTPPH0RHT', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'net.activetheory.paperplanes-screens/screenshot_1.png'],
    ['3MD8CKRQZZAY6CB2NRPXIB94294RJY', 'AVGD7RELGEJOB', GEMMA_PREFIX + 'com.hailocab.consumer-screens/screenshot_4.png'],
    ['3MA5N0ATTCYYPSY646ZF6PHY9DJWK5', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.forbeautiful.girly-screens/screenshot_1.png'],
    ['3JMQI2OLFZS0OSJIKOM56T9HWSJNDT', 'A349TSPKFHIWE1', GEMMA_PREFIX + 'com.pregbuddy-screens/screenshot_1.png'],
    ['3E24UO25QZDJL44FBGE4FCZU5G1O6H', 'AWHJJ1YYPK65H', GEMMA_PREFIX + 'com.copyharuki.englishspanishdictionaries-screens/screenshot_2.png']
    ]
    
    
    
    
    #blacklist of screenshots that are either in landscape, are blank, or have some other problem
    #these are excluded from unique.csv and duplicate.csv (as if they are rejected)
    screenshot_blacklist = [GEMMA_PREFIX + "com.budgestudios.googleplay.StrawberryShortcakeHolidayHair-screens/screenshot_1.png",
                            GEMMA_PREFIX + "mobile.fourkites.com.carrierLink-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.buildabear.honeygirls.selfie-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.dinosaur-screens/screenshot_1.png", 
                            GEMMA_PREFIX + "com.abg.VRVideoPlayer-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kidstatic.musicinstruments-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.honeywell.mobile.android.totalComfort-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.kolesnik.pregnancy-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.cutegirls.stuffs-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.andromo.dev498527.app566087-screens/screenshot_5.png",
                            GEMMA_PREFIX + "se.appfamily.superpuzzlefree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fun.photo.apps.bodybuilding-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.mbek.waterfallphotoframes-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sesameworkshop.alphabetkitchen.play-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.babbel.mobile.android.en-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.toongoggles.a-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.atranssexualdate-screens/screenshot_2.png",
                            GEMMA_PREFIX + "co.cristiangarcia.dueodirecto-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.mustafademir.drumkit-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.appxy.tinyscanner-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.hasbro.riskbigscreen-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mattel.bestjobever-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cg.flyingdinosaur2017simulator-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.planner5d.planner5d-screens/screenshot_5.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlerscello-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tv.kidoodle.android-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.mp.Injusticepolicecargosquad-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.arrowstar.FunnyFoodsLite-screens/screenshot_1.png",
                            GEMMA_PREFIX + "young.ooh.notcrysound-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ohnineline.sas.kids-screens/screenshot_3.png",
                            GEMMA_PREFIX + "net.aljazeera.english-screens/screenshot_4.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlersbanjo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.caynax.a6w-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.disneydigitalbooks.disneycolorandplay_goo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.disney.datg.videoplatforms.android.abcf-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.chatews.newyork-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.dorna.officialmotogp-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.kassanity.dashysquarelite-screens/screenshot_1.png",
                            GEMMA_PREFIX + "net.visiting.card.maker-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.cadTouch.androidTrial-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.tinylabproductions.tropicalislands-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.planner5d.planner5d-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.YOZHStudio.RailRoad-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.gamestar.pianoperfect-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.appsministry.fixiki-screens/screenshot_4.png",
                            GEMMA_PREFIX + "net.metaquotes.metatrader4-screens/screenshot_2.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlersbanjo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.gamegarden.fk-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ally.MobileBanking-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.sim.policescifibikeriderpark-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.edujoy.toddler.games-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.psvn.KidsBasketball-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.metajunky.glitterpaint-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.PSVStudio.KidsAirport-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tw.mobileapp.qrcode.banner-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mexico.autosusados-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mico-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.medicaljoyworks.clinicalsense-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.birthdayparty-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.abto.morenails-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.owlchirp.puzzlecars-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.jns.info.vallentine.wishes-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.google.android.apps.books-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.tinylabproductions.motocrosswinter-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tutotoons.app.monstersisters2homespa.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.honeywell.mobile.android.totalComfort-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.breakcoder.volleyballscoreboard-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.KidsGames.SuperHippo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.FairWare.PixelStudio-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.lge.tv.remoteapps-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.kidstatic.kidsanimalpiano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.DigitalLemon.LearnColor-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.planner5d.planner5d-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.k3games.babyfashiontailor2.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hasbro.FurbaccaAPPSTORE-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.jesso.jesbenthomas.Swastika-screens/screenshot_5.png",
                            GEMMA_PREFIX + "de.wonderkind.peekaboo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ethiopian.arada-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.appsministry.kikoriki-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.raycom.kfvs-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.iabuzz.Beepzz2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.DressFashion.retry-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fusion.recetascomdo-screens/screenshot_4.png",
                            GEMMA_PREFIX + "net.visiting.card.maker-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.icatch.SkyThunderRC-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.escort.androidui.root-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hearst.android.wyff-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.R.K.Games.JW_Puzzle-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.agnitus.playearlytobedearlytorisebook-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.instatrendyyy-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.topoflearning.free.vibering.medical.advanced.biology.exam.review-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.deltatre.atp.tennis.android-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ehawk.antivirus.applock.wifi-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.imayi.dinosaurdiggerfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "app.spider.snow.partylight-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.hearst.android.wgal-screens/screenshot_4.png",
                            GEMMA_PREFIX + "jp.nagoya_studio.shakocho-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.fox.android.fsp-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.professionalbanjo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.k3games.jungleanimalsalon.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.psvn.traumatologist-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hiddenobjectsecrets.mansion3-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.motocross-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.eyup.electrickguitar-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tomdxs.dronefpv-screens/screenshot_1.png",
                            GEMMA_PREFIX + "net.bucketplace-screens/screenshot_3.png",
                            GEMMA_PREFIX + "ru.sort1-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.nbcuni.telemundostation.telemundo52-screens/screenshot_4.png",
                            GEMMA_PREFIX + "pl.infinzmedia.electricscreenfree-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.edujoy.Dinosaur_pet-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.comica.comics.google-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kids.free.audiobook.nursery.rhymes.songs.offline-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.hearing.healthcare.siemens.touchcontrol-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.coolniks.niksgps-screens/screenshot_4.png",
                            GEMMA_PREFIX + "net.metapps.meditationsounds-screens/screenshot_4.png",
                            GEMMA_PREFIX + "pl.mkonferencja.exodus-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.andromo.dev535138.app530249-screens/screenshot_6.png",
                            GEMMA_PREFIX + "br.com.blooti.chavinhogroove-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.yinzcam.nfl.sf-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.candlify.vrplayer-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.toilet-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.androbaby.firstanimalsforbaby-screens/screenshot_1.png",
                            GEMMA_PREFIX + "id.mcdonalds.delivery-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.getaround.android-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.application.goldteetphotoeditor-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.racergame.cityracinglite.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.workoutapps.fatburnworkout-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.PSVGamestudio.PhotoAdventureForKids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.developdroid.mathforkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.gm.chevrolet.nomad.ownership-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.happyconz.blackbox-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.embarazosemanaasemana.controldiaadia-screens/screenshot_6.png (page in Spanish)",
                            GEMMA_PREFIX + "com.pgatourlive.pga-screens/screenshot_6.png",
                            GEMMA_PREFIX + "br.com.ctncardoso.ctncar-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.anuntis.fotocasa-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.friendmapper-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.gamegarden.ft-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.apps.zc.memessongsplus-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.sesameworkshop.elmoloves123s-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.coolniks.niksgps-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.developdroid.mathforkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.spreaker.android-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.applicaster.il.babyfirsttv-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.california.cyber.developers.gps.speedometer.tripmeter-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.buildabear.honeygirls.selfie-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.kellerwilliams.eventapp-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.babbel.mobile.android.en-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.landoncope.games.toddlersingandplay2free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.recycledjeanscraftideas.pitlord-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.BarbieMagicalFashion-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.samiapps.sami.sleepBabyOwl-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.yahoo.mobile.client.android.finance-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.proframeapps.videoframeplayer-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.emisoradominicana.tv-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.racergame.cityracing3d-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.HBO-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lego.legotv-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hyvee.android-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.j2.myfax-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.cmgdigital.android.wsocweather-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.organized-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.wsandroid.suite-screens/screenshot_3.png",
                            GEMMA_PREFIX + "dolphin.video.players-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.budgestudios.BarbieSuperstar-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.idw.TMNTreader-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.androidlab.videoroad-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.TransformersRescueBotsHeroAdventures-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.ThomasAndFriendsGoGoThomas-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.rtve.clan-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.samrendra.Bowler-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cadTouch.androidTrial-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.crayolallc.carcreator_android-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.paypal.here-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nascar.raceviewmobile-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.falkolife.obdcods-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.postoffice-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cg.railroadbusredemptionroad-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.dating.apps.cupid-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cubamessenger.cubamessengerapp-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.bubadu.buildergame-screens/screenshot_1.png",
                            GEMMA_PREFIX + "meavydev.ARDrone-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cg.calculatorcashregisterkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fic.foxsports-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.royalcaribbean.iq-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hiddenobjectsecrets.mansion-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVGamestudio.piratestreasure2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.catchsports.catchsports-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.samiapps.ksvp-screens/screenshot_1.png",
                            GEMMA_PREFIX + "zok.android.numbers-screens/screenshot_1.png",
                            GEMMA_PREFIX + "net.sourceforge.opencamera-screens/screenshot_5.png",
                            GEMMA_PREFIX + "mrigapps.andriod.fuelcons-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.mgae.numnoms.flavorfusion-screens/screenshot_1.png",
                            GEMMA_PREFIX + "baby.com.BubblePOPKids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tomdxs.drone-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.yahoo.mobile.client.android.sportacular-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.gamestar.pianoperfect-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.cyberlink.powerdirector.DRA140414_02-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.sonymobile.androidapp.audiorecorder-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.onlinico.rubeslab-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.goodbarber.explore419-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.livechat.dating-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.affinity.san_francisco_football-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.saga.user.my_ouote-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.fareness.fareness-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.americanwell.android.member.amwell-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.interactive8.readmestories-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.intellijoy.drawwithshapes.lite-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ibm.events.android.atlantaunited-screens/screenshot_6.png",
                            GEMMA_PREFIX + "co.filld.frontend-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.eniseistudio.caraccessories-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.bulky.sports.snow.street.basketaball.holiday-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.animocabrands.google.mhminismania-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mattel.eahdragons-screens/screenshot_1.png",
                            GEMMA_PREFIX + "pl.rosmedia.music-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.qrcodescanner.barcodescanner-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.tutotoons.app.rockstaranimalhairsalon.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lge.tv.remoteapps-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.LearnMangaDrawing.rahayu-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.duckduckmoosedesign.ibs-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.drum2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.honeywell.mobile.android.totalComfort-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.duckduckmoosedesign.bat-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.androidlab.videoroad-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ea.game.simcitymobile_row-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.outfit7.talkingnewsfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.StrawberryShortcakeCandyGarden-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.oki.cow-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.bydeluxe.d3.android.program.starz-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.flipps.fitetv-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.FairWare.PixelStudio-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.digitalkidsapp.cartoonforkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sean.candleprox-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pescapps.kidspaint-screens/screenshot_2.png",
                            GEMMA_PREFIX + "ru.car2.cardashboard-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.nchsoftware.pocketwavepad_free-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.sonymobile.androidapp.audiorecorder-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.pilottravelcenters.mypilot-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.indigokids.mim-screens/screenshot_2.png",
                            GEMMA_PREFIX + "hr.podlanica-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.webruli.musicalinstruments-screens/screenshot_1.png",
                            GEMMA_PREFIX + "yamayka.apps.Makeup-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.webruli.ukulele-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.geekslab.screenshot-screens/screenshot_6.png",
                            GEMMA_PREFIX + "business.card.maker.scopic-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.partycity-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.bewild.dating.android-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.instamag.activity-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.klonengam.nickiminajprank.calli-screens/screenshot_4.png",
                            GEMMA_PREFIX + "to.fax.android-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.weebly.android-screens/screenshot_2.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.professionaltrombone-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.bediryazilim.littlepiano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.one20.ota-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.simcoachgames.rc21x-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.wheelsvision-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.GameStudio.PuppiesFirePatrol-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.samrendra.Bowler-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.Andrey.Pimpmyride-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tomdxs.symago-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.kevinbradford.games.pklg2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.buildabear.honeygirls.video-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.mobile.bizo.reverse-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.hiddenobjectsecrets.gardensecrets-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mcpeppergames.freeAmazingCarWashGame-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.oki.piano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.appgame7.jigsaw.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pg.helicopterrescuepracticesim-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.medm.atbaki-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.mustafademir.piano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.duckduckmoosedesign.trucks2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "net.metapps.sleepsounds-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.appsministry.fixiki-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.tomdxs.symafpv-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kevinbradford.games.barnyardgamesfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.KnowledgeAdventure.SchoolOfDragons-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.razmobi.zootimeforkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.shahvilla.vhs-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.doobeedoo.balloonpoppingfortodlers-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kevinbradford.games.firstgrade-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.artomob.artteacher-screens/screenshot_3.png",
                            GEMMA_PREFIX + "tv.kidoodle.android-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.mobi2fun.lionkingdom-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.softbolt.redkaraoke-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.famousbluemedia.piano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mybabygames.nurseryrhythms-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.algoriddim.djay_free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pdffiller-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVGameStudio.HippoFirePatrol-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.att.tv-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nchsoftware.pocketwavepad_free-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.buildabear.honeygirls.video-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.beansprites.mysteryhauntedhollowFREE-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.korrisoft.ringtone.maker-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.appsministry.fixiki-screens/screenshot_2.png",
                            GEMMA_PREFIX + "appinventor.ai_homestudioapp.angelsexist-screens/screenshot_3.png",
                            GEMMA_PREFIX + "org.dayup.stocks-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.gokids.drums-screens/screenshot_1.png",
                            GEMMA_PREFIX + "br.com.brainweb.ifood-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.iggnovation.hackanyone-screens/screenshot_2.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlersxylophone-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.miniclip.dinopets-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mark.calligraphy-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlersclarinet-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.autolauncher.motorcar.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.TeenOutfit.retry-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.project.vivareal-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lemurmonitors.bluedriver-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.jvr.dev.magnifying.glass.light.camera-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.itcode.reader-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.kids.preschool.learning.games-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlerssaxophone-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVStudio.KidsPolicemanStation-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVStudio.HippoPrincessAndTheWizard-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.emergency-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.food-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.iabuzz.Puzzle4KidsFood-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.MyLittlePonyHarmonyQuest-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.monstertruckwinter-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVStudio.kidscafewithhippo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.highlights.apps.highlightsshapes-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.madmax-screens/screenshot_1.png",
                            GEMMA_PREFIX + "bwebmedia.sandralearnshousecraft-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.psvn.CandyBAR-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lego.ninjago.skybound-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.seaworld-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.olympic-screens/screenshot_1.png",
                            GEMMA_PREFIX + "se.appfamily.balloonpopfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.bimiboo.colors-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.icetruck-screens/screenshot_1.png",
                            GEMMA_PREFIX + "se.appfamily.coloring.dinosaurs.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.lovehealthy-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pescapps.BurgerKids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pescapps.kids_paint_free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mp.tractorfarmingsimulator-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.easylabs.xylophone-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.originatorkids.EndlessAlphabet-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.turner.surelyyouquest-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.edujoy.carwash-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.policeracing-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tivola.dogworld.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.christmas-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.psvn.doors100-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.ThomasAndFriendsExpressDelivery-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.eyuponer.colorfulpiano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.razmobi.monstertrucks2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.superheroes-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mygdx.clock-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lescapadou.cursivefree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lego.bricksmore-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.dj-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.egert.piano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.eyup.drum-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.internetdesignzone.nurseryrhymes-screens/screenshot_1.png",
                            GEMMA_PREFIX + "co.romesoft.toddlers.puzzle.truck-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cg.policedonutrestaurantpd-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ChordFunc.MiReDo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tabtale.carebears-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.psvn.KidsBicycle-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kidsworld.surpriseeggs-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.motocrossadvanced-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.concert-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nick.android.nickjr-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sesameworkshop.elabcs.play-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pescapps.connect4-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.archiactinteractive.LamperVRCardboard-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.miumiu-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.MissHollywood2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.dreamworkscolor-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lego.city.my_city2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tomdxs.dronefpv-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.imib.cctv-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.pearlgames.villagecityislandsim2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.playtoddlers.happydaycarestories.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.puzzletime.jigsaw-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cyberlink.powerdirector.DRA140225_01-screens/screenshot_1.png",
                            GEMMA_PREFIX + "in.fulldive.applicationslauncher-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.han.dominoes-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.androidbuttonaccordion.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ragassoft.shapespuzzlesforkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sagosago.Boats.googleplay.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.greenguard.global-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.draw.trippy-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.dinosaurII-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pjmasks.moonlightheroes-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.appquiz.baby.ballons-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVStudio.HippoCristmassCalendar-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.MonsterHighFrightfulFashion-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.robotifun.smart.baby.LITE-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mb.tractorfarmersimulator2016-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.YovoGames.aefishing2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.songV-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVGamestudio.BeachFamilyBusiness-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.candy-screens/screenshot_1.png",
                            GEMMA_PREFIX + "net.kidjo.app.android-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.StudioTwoBeans.HorseQuest-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.touchzing.bestnurseryrhymes-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tv.benandholly.party-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.appquiz.baby.musical-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVGamestudio.CarServiceHippo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.StrawberryShortcakeBakeShare-screens/screenshot_1.png",
                            GEMMA_PREFIX + "pl.muse.blocks.castle-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.eyewind.colorfit.mandala-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.gm.despegar-screens/screenshot_1.png",
                            
                            
                            # beginning of landscape/blank/foreign language screens
                            # that were previously a part of the dataset but then were taken out by combing
                            # through every image in the dataset
                            
                            GEMMA_PREFIX + "com.smackall.animator-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pg.NewYorkPoliceSimulator-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nchsoftware.videopad_free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.jacek.miszczyk.pregnancytestLite-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.pg.shipsimulatorgo2017-screens/screenshot_1.png",
                            GEMMA_PREFIX + "co.romesoft.kids.car.wash.salon.autoBodyShop-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.intuit.turbotax.mobile-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kevinbradford.games.preschoolgames-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.highlights.apps.highlightsmonsterday.mx-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.carpart.classic-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.artomob.artteacher-screens/screenshot_1.png",
                            GEMMA_PREFIX + "uk.co.fusionlogic.babysbouncycastle-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.doubledigital.fishschooling-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.weebly.android-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.oki.shapesnew-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.babbel.mobile.android.eng-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.travelsafety-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.discovery.tlcgo-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.autolauncher.motorcar.free-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.medicaljoyworks.clinicalsense-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.arent.snakespuzzles-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kevinbradford.games.fourthgrade-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.topoflearning.free.vibering.medical.advanced.biology.exam.review-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.mobile.bizo.slowmotion-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.kidstatic.toddlerpiano-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.mattel.DCsuperherogirls-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.planner5d.planner5d-screens/screenshot_4.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlersflute-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.andymstone.metronome-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlersdrum-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hbo.hbonow-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cibgHippo.Babyshop-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lego.legotv-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.tiltangames.dinopuzzle-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.autolauncher.motorcar.free-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.spreaker.android.studio-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.flexgames.pixasso-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.pg.dinosaur.jeep.driving.zone.sim-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.turner.trutv-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.paypal.here-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.q2developer.q2developercreateyourownmusic-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.disney.disneymoviesanywhere_goo-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.ElfizMedia.Djdubstepmscmkrpd-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.dokdoapps.mybabydrum-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kodak.kodakprintmaker-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.clearchannel.iheartradio.connect-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.mustafademir.realpiano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.candyland-screens/screenshot_1.png",
                            GEMMA_PREFIX + "net.aljazeera.english-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.toongoggles.a-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.undercoverdesigns.oneminuteultrasound.app-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pgatourlive.pga-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.western-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tamer.android.prayertimes-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.turner.tbs.android.networkapp-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.puzzlegame.superheroesandprincessesforkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "codematics.universal.tv.remote.control-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.intellijoy.pack-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.topoflearning.free.vibering.medical.advanced.biology.exam.review-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.goemonfactory.pianotracer-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.spreaker.android.studio-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.pg.blockyairplaneparking-screens/screenshot_1.png",
                            GEMMA_PREFIX + "ru.car2.cardashboard-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.coolniks.niksgps-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.developdroid.mathforkids-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.farm-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tomdxs.dronefpv-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.flexgames.pixasso-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.babyfirsttv.peekaboo_i_see_you-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlersviolin-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lge.tv.remoteapps-screens/screenshot_4.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.professionaltrumpet-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.qweqweq.kookwekker-screens/screenshot_2.png",
                            GEMMA_PREFIX + "co.romesoft.girls.highHeels.shoesDesigner-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.Mensajes.buena.manana.tarde.noche-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.brainpop.brainpopjuniorandroid-screens/screenshot_3.png",
                            GEMMA_PREFIX + "ideamk.com.surpriseeggs-screens/screenshot_2.png",
                            GEMMA_PREFIX + "br.com.blooti.chavinhogroove-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fox.android.fsp-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.ohnineline.sas.kids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.buffalowildwings.blazinrewards-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.braces.photo.editor.girls-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.echolake.daveysmystery-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nchsoftware.videopad_free-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.kevinbradford.games.secondgradefree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.discovery.tlcgo-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.tivola.wildlife.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mybabygames.nurseryrhythms-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.cadTouch.androidTrial-screens/screenshot_4.png",
                            GEMMA_PREFIX + "net.metaquotes.metatrader4-screens/screenshot_1.png",
                            GEMMA_PREFIX + "net.bucketplace-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cyberlink.powerdirector.DRA140414_02-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.smackall.animator-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.remind4u2.sounds.of.letters.alphabet.kids-screens/screenshot_2.png",
                            GEMMA_PREFIX + "net.metapps.sleepsounds-screens/screenshot_6.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.professionalsaxophone-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.dlink.mydlink-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pg.blockyemergencyparking-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.barbie.lifehub-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hd.factory.anime-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.libii.talentedpetsshow-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.activision.skylanders.creator-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.spreaker.android-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.PSVStudio.HippoLaboratory-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hasbro.tf360appstore-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pawsinc.garfielddaily-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.lego.common.legolife-screens/screenshot_3.png",
                            GEMMA_PREFIX + "wsj.reader_sp-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.league.theleague-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.dorna.officialmotogp-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.famousbluemedia.piano-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.mark.calligraphy-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.ics.creditcardreader-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.famousbluemedia.piano-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.sweefitstudios.drawgraffiti-screens/screenshot_6.png",
                            GEMMA_PREFIX + "net.greysox.tayoshuttle-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nchsoftware.pocketwavepad_free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.clearchannel.iheartradio.connect-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.newkidsgames.xylophonekidspiano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tutotoons.app.bffworldtriphollywood2.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.shopping-screens/screenshot_1.png",
                            GEMMA_PREFIX + "holoduke.soccer_gen-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.professionalmandolin-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.cutebabydrum-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.crunchyroll.crunchyroid-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.foxandsheep.nightynight-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.agnitus.playtwinkletwinklelittlestarsbook-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.af.screenmanager-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.selfhealing.kamin-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.iabuzz.puzzle4kidsAnimals-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mobincube.android.sc_33JPWH-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pg.realcitycardrivingsim-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.energysh.drawshow-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tutotoons.app.sweetbabygirldaycare5.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tomdxs.symafpv-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.o_taiji.digitimer-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.pg.formula.racing.car.cargo.plane-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.midasapps.roomcreator-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.flexgames.stickypixels-screens/screenshot_3.png",
                            GEMMA_PREFIX + "baby.com.DisneyCarToys-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.smackall.animator-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.k3games.burgershop.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVStudio.PuppyPolicePatrol-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nchsoftware.videopad_free-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.kidsworld.surprisetoys-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.YovoGames.trainwash-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.TheSmurfGames-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tophatter.jewelry-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.learntomaster.vtlts-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.newson.vinson-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tivola.doghotel-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.abc-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.disney.disneymoviesanywhere_goo-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.chaves.adventures.super.world-screens/screenshot_1.png",
                            GEMMA_PREFIX + "by.alfasoft.WashCarGame-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.professionaltuba-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.clearchannel.iheartradio.connect-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.flightview.flightview_free-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.mediadjz.pianomixer-screens/screenshot_1.png",
                            GEMMA_PREFIX + "co.infinum.ptvtruck.usa-screens/screenshot_1.png",
                            GEMMA_PREFIX + "mobile.eaudiologia-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sagosago.Friends.googleplay-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.farmers.ifarmers-screens/screenshot_2.png",
                            GEMMA_PREFIX + "org.pbskids.jurassicjr-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.intretech.bounty-screens/screenshot_2.png",
                            GEMMA_PREFIX + "net.kidjo.app.android-screens/screenshot_2.png",
                            GEMMA_PREFIX + "net.metapps.watersounds-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.robotifun.tangram-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.buildabear.honeygirls.video-screens/screenshot_1.png",
                            GEMMA_PREFIX + "air.com.animangaplus.doodlejump01lite-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.duapps.cleaner-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nickonline.android.nickapp-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.eumlab.android.prometronome-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.discovery.tlcgo-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.br.meow.cute.kitty.puppy.cat-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.barrydrillercom.android-screens/screenshot_5.png",
                            GEMMA_PREFIX + "net.visiting.card.maker-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lego.starwars.thenewyodachronicles-screens/screenshot_1.png",
                            GEMMA_PREFIX + "co.romesoft.toddlers.memory.music-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.bathing-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mobile.bizo.slowmotion-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.flightview.flightview_free-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.creaturecorp.kidscashregistergrocery-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.labdogstudio.pairs-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.route4me.routeoptimizer-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.dpj.drawmine2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.breakcoder.volleyballscoreboard-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tutotoons.app.zoeysmakeupsalon.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.monstertruckpolice-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mattel.everafterhigh-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cateater.stopmotionstudio-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.professionalcello-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.touchzing.animalsoundsafari-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fotoable.guitar-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.care-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mobi2fun.speedbusrace-screens/screenshot_2.png",
                            GEMMA_PREFIX + "tv.kidoodle.android-screens/screenshot_3.png",
                            GEMMA_PREFIX + "jp.nagoya_studio.shakocho-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.tomdxs.dronefpv-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.tinylabproductions.rc-screens/screenshot_1.png",
                            GEMMA_PREFIX + "vStudio.Android.Camera360-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.pawsinc.garfielddaily-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.movile.playkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "mobile.eaudiologia-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.apartments.mobile.android-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.developdroid.mathforkids-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.tomdxs.drone-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.korrisoft.ringtone.maker-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kids.preschool.learning.games-screens/screenshot_3.png",
                            GEMMA_PREFIX + "chilon.consult.pingpong2-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.bediryazilim.realtrumpet-screens/screenshot_2.png",
                            GEMMA_PREFIX + "co.romesoft.toddlers.memory.digger-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kids.preschool.learning.games-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.inertiasoftware.crossstitchworld-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.wondershare.filmorago-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tivola.doghotel-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.icatch.SkyThunderRC-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.datebytesapps.dating-screens/screenshot_2.png",
                            GEMMA_PREFIX + "mesi.macroshooting-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.professionalflute-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.babbel.mobile.android.en-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.helloworld.realcardriving2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.learn.bibliavalera-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.danpatrick-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.edujoy.Word_Search-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pg.dutydriverarmy4X4offroad-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.appquiz.memory.training-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fiservcardvalet.mobile.android-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.cpqas.legobatman3-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.applicaster.il.babyfirsttv-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.YovoGames.Kindergarten-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.spreaker.android-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.whatisone.afterschool-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.msgi.msggo-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.tinylabproductions.pirates-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.disney.frozenletitroll_goo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.professionalviolin-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.policemotocross-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.yinzcam.nfl.eagles-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.razmobi.kidscarsgame-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.gamenica.nurettink.pianoandnotesfortoddlers-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.one20.ota-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.mobile.bizo.reverse-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.spreaker.android-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.imayi.dinotruckfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tinylabproductions.trains-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pgatourlive.pga-screens/screenshot_4.png",
                            GEMMA_PREFIX + "business.card.maker.scopic-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.wondershare.filmorago-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.fundevs.app.mediaconverter-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.cubicfrog.edukitchen-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.vizorg.obd2_code-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.cruiseinfotech.sixpackphotoeditor-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nchsoftware.videopad_free-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.danpatrick-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.nchsoftware.videopad_free-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.cgh.hairstyles-screens/screenshot_4.png",
                            GEMMA_PREFIX + "net.metapps.meditationsounds-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.undercoverdesigns.oneminuteultrasound.app-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.sweefitstudios.drawgraffiti-screens/screenshot_1.png",
                            GEMMA_PREFIX + "tr.com.alyaka.alper.toddlersfrenchhorn-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.medicaljoyworks.clinicalsense-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.FairWare.PixelStudio-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nimbleminds.rhymingwordsfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.FoxieGames.DinoWorldOnline-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sweefitstudios.drawgraffiti-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.workSPACE.Fraksl-screens/screenshot_1.png",
                            GEMMA_PREFIX + "baby.com.MightyToys-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.caillouhouseofpuzzles-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.redfrog.mlp-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.discovery.discoverygo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.weebly.android-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.caynax.a6w-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.neulion.android.tablet.nfl.wnfln-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cisco.bce-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.learntomaster.vtlts-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.bediryazilim.violin-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fox.now-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.srbd.ml2fdt.vr.environment-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.midasapps.roomcreator-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tutotoons.app.sweetbabygirlnewborn2.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.spreaker.android-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.flexgames.stickypixels-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.korrisoft.ringtone.maker-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.kauf.imagefaker.photofunfunnypicscreator-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.spreaker.android-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.PSVStudio.HippoWashing-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ea.gp.fifamobile-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.samremote.view-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.mustafademir.little_piano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lescapadou.tracingfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.intretech.bounty-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.medicaljoyworks.clinicalsense-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.nchsoftware.pocketwavepad_free-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.lge.tv.remoteapps-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.topoflearning.free.vibering.medical.advanced.biology.exam.review-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.mark.calligraphy-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.jakkspacific.animalbabies-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.intellijoy.abc.trains.lite-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fourabrothers.momandbabycenter-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ats.etrackcertified-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.caynax.a6w-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.lucktastic.scratch-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.developdroid.mathforkids-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.tabtale.pjparty-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.dorna.officialmotogp-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.earlystart.android.monkeyjunior-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.homes.homesdotcom-screens/screenshot_2.png",
                            GEMMA_PREFIX + "mobi.funbabyapps.babypiano-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.rtve.clan-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.learntomaster.vtlts-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.hasbro.lpsyourworld-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.rmsgamesforkids.xylophone-screens/screenshot_3.png",
                            GEMMA_PREFIX + "net.aljazeera.english-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.disney.disneymoviesanywhere_goo-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.breakcoder.volleyballscoreboard-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.tutotoons.app.fairyland3unicornfamily.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "mobi.funbabyapps.babyxylophone-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.libiitech.jungledoctor-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.tutotoons.app.sweetbabygirlfirstlove.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.activision.peanuts-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mb.trainsimanimaltransport-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.beatronik.djstudiodemo-screens/screenshot_2.png",
                            GEMMA_PREFIX + "jp.co.ofcr.cm00-screens/screenshot_1.png",
                            GEMMA_PREFIX + "mobile.eaudiologia-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.jcbsystems.dollhouse1-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fox.android.fsp-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.loudcrow.marvelavengers-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kids.free.audiobook.nursery.rhymes.songs.offline-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.discovery.tlcgo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.artomob.artteacher-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.kidsdevgames.surpriseeggsforkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.missingames.kids.painting.lite-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tomdxs.symago-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.paypal.here-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.icatch.SkyThunderRC-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.af.screenmanager-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.hutchgames.hotwheels-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hbo.hbonow-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.planner5d.planner5d-screens/screenshot_2.png",
                            GEMMA_PREFIX + "net.bytefreaks.opencvfacerecognition-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.fami.cam-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.scdroid.atranalyzer-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.rmsgamesforkids.colorslearning-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.amusement-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.libiitech.candyhospital-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.cateater.stopmotionstudio-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.pgatourlive.pga-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.ips.monstertrucksimulator-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kakzapletat-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.beatronik.djstudiodemo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lego.legotv-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.fridaylab.deeper-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.gamonaut.blockyDemolitionDerby-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.intretech.bounty-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.RMB.Chamar_Gato-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.pikpark.PikParkApp-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.discovery.tlcgo-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.planner5d.planner5d-screens/screenshot_3.png",
                            GEMMA_PREFIX + "bug.smash.matchthree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.doodlejoy.studio.kidsdoojoy-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.selfhealing.candle11-screens/screenshot_1.png",
                            GEMMA_PREFIX + "thelusca.com.toddersactivitiestest-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lamudi.android-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.phil.tv.view-screens/screenshot_2.png",
                            GEMMA_PREFIX + "luzheng.cam.govue-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.laura.fashiondesign-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.mybabygames.nurseryrhythms-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.samrendra.Bowler-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.korrisoft.ringtone.maker-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.pawsinc.garfielddaily-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.company.aquariumvr-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pescapps.game4kids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.inertiasoftware.jigsawworld-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.disney.disneymoviesanywhere_goo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ea.game.maddenmobile15_row-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tutotoons.app.peachfriendspajamafun.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.medicaljoyworks.clinicalsense-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.tonymagdy.PortablePiano-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kidstatic.kidsmusicinstruments-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tomdxs.skytech-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.jcbsystems.dollhouse1-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.disney.disneymoviesanywhere_goo-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.bytestorm.artflow-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.edubuzzkids.surpriseeggs-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kmb.lh-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.tomdxs.symago-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.samrendra.Bowler-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.carpart.r4932-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.asus.aicam-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.pg.jetskidrivingsimulator-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.liverpoolsol.new_nail_art_designs-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.korrisoft.ringtone.maker-screens/screenshot_5.png",
                            GEMMA_PREFIX + "jp.nagoya_studio.shakocho-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.lego.nexoknights.merlok-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.lego.duplo.trains-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.libiitech.littlepetdoctorpuppy-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hasbro.FurbyWorldAPPSTORE-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ocigrup.dottodot-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.toongoggles.a-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.charlieaffs.lotto-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.orionsmason.knockknockjokesfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.midasapps.roomcreator-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.nascar.nascarmobile-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.YOZHStudio.Balloons-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.CaillouSearchAndCount-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ionicframework.gravity-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.sinyee.babybus.dining-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.babbel.mobile.android.eng-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.emisoradominicana.tv-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.onteca.powertools-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cadTouch.androidTrial-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.kidsgames.superheroprincessgameforkids-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pawsinc.garfielddaily-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.msgi.msggo-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.qrcodescanner.barcodescanner-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.tomdxs.symago-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.imayi.trainbuilderfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ohnineline.sas.kids-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.developandroid.android.kidspiano-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.fxnetworks.fxnow-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.phil.tv.view-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.ats.etrackcertified-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.topoflearning.free.vibering.medical.advanced.biology.exam.review-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.cubicfrog.edukittyFree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ImagineLearning.IL-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tutotoons.app.ponysistershairsalon2.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.adi.remote.phone-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cateater.stopmotionstudio-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.uc.iflow-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.PSVGamestudio.HippoSchoolBus-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.ea.gp.nbamobile-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.budgestudios.ChuggingtonReadyToBuild-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.hasbro.mlpcoreAPPSTORE-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mindcandy.mmm-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nchsoftware.pocketwavepad_free-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.Baraban.NewtonCr-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pranktent.kollpang.gengkol-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.hasbro.HeroDroid-screens/screenshot_2.png",

                            # end of landscape/blank/foreign language screens
                            # that were previously a part of the dataset but then were taken out by combing
                            # through every image in the dataset
                    
                            GEMMA_PREFIX + "com.hornet.android-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.hornet.android-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.idw.transformers.reader-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.pescapps.kidspaint-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.rutaett.HeroesMovie-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kidstatic.kidsanimalpiano-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.tutotoons.app.christmasanimalhairsalon2.free-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.cadTouch.androidTrial-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.flexgames.stickypixels-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.burabura.FlyingLogo-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.learntomaster.vtlts-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.kevinbradford.games.barnyardgamesfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.pg.chess.player2-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.edubuzzkids.learntospellforchildren-screens/screenshot_1.png",
                            GEMMA_PREFIX + "holoduke.soccer_gen-screens/screenshot_1.png",
                            GEMMA_PREFIX + "appinventor.ai_freebies_freesamples_coupons.StoreCoupons-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.quagem.autoexpreso-screens/screenshot_2.png",
                            GEMMA_PREFIX + "net.metapps.meditationsounds-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.bigfans.crchesttracker-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.iversecomics.archie.android-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.magicv.airbrush-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.mark.CartoonImage-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.tmarki.comics-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nbcuni.telemundostations.puertorico-screens/screenshot_3.png",
                            GEMMA_PREFIX + "marrygold.logomaker3d-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.lamudi.android-screens/screenshot_3.png",
                            GEMMA_PREFIX + "co.com.fincaraiz.app-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.rexton.bxsmartremote-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.atistudios.italk.ja-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.norwegian.travelassistant-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.gogoair.ife-screens/screenshot_1.png",
                            GEMMA_PREFIX + "appinventor.ai_app_jcstudio.FodaSe-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.messageloud-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.nbcuni.telemundostation.telemundony-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.docusign.ink-screens/screenshot_1.png",
                            GEMMA_PREFIX + "net.hondash.hondash-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.hcceg.veg.compassionfree-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.amitech.allevents-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.weathernowapp.weathernow-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.norwegian.travelassistant-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.dexcom.follow-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.eogames.babysounds-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.itcode.reader-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.mhd.flasher.n54-screens/screenshot_4.png",
                            GEMMA_PREFIX + "arab.ringwe.com-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.atlogis.northamerica.free-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.preethu.motors-screens/screenshot_6.png",
                            GEMMA_PREFIX + "com.droid27.senseflipclockweather-screens/screenshot_2.png",
                            GEMMA_PREFIX + "com.ae.ae-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.earny.android-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.citizensbank.androidapp-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.bergman.fusiblebeads-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.kauf.sticker.funfacechangerproeffects-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.earny.android-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.gau.go.launcherex.theme.flowers.baohan-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.univision.noticias-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.escapistgames.starchart-screens/screenshot_1.png",
                            GEMMA_PREFIX + "ru.loveplanet.app-screens/screenshot_2.png",
                            GEMMA_PREFIX + "pl.nenter.app.flashlightgalaxys5-screens/screenshot_1.png",
                            GEMMA_PREFIX + "com.california.cyber.developers.gps.speedometer.tripmeter-screens/screenshot_4.png",
                            GEMMA_PREFIX + "com.spreadsong.freebooks-screens/screenshot_5.png",
                            GEMMA_PREFIX + "com.Mensajes.buena.manana.tarde.noche-screens/screenshot_3.png",
                            GEMMA_PREFIX + "com.touchbaseinc.seattletalent-screens/screenshot_1.png"
                            ]
                            

    sampled_screenshots = [] # list of all screenshots in new_batch.csv (to make sure we dont sample something that is currently pending)
    
    if os.path.isfile("new_batch.csv"):
        sampled_screenshots = open("new_batch.csv", "r").read().splitlines()
        del sampled_screenshots[0] # get rid of the header
    else:
        print("\nWarning: ./new_batch.csv does not exist, so if there is a batch out, the sent out files will be marked as unused in the master list.")
        
        
    empty_desc = {} # dictionary containing all entries that have empty descriptions

    description_list_duplicates = {} #dictionary mapping a string combining all descriptions to a list of corresponding filenames
                                     #with this dictionary, we will find any screens that have duplicated descriptions


    for f in files:
        if f[len(f)-4:len(f)] == ".csv": #if it's a .csv file (not a directory)
            #read in the csv as 2D array
            
            csvfile = open(directory + "/" + f, "r")
            
            lines = csvfile.read().splitlines()
            
            csvfile.close()
                                   
            for row in csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if len(row) == 33: # i.e. if it is not a header row (because header rows have 35 elements)
                    if (row[16].lower() == "approved") and ([row[0], row[15], row[27]] not in HIT_blacklist): # row[16] is the AssignmentStatus ("Approved" or "Rejected")
                                                      # we only consider rows that have been approved manually by someone in the lab; also exclude ones that we rejected later manually (part of HIT blacklist)
                        fn = row[27] #row[27] is the filename  
                        
                        garbage = [] # screens we want to blacklist as a list of GEMMA urls
                        
                        blacklist_row = [row[0], row[15], row[27]]
                        
                        if fn in garbage:
                            print(blacklist_row)
                        
                        if not fn in screenshot_blacklist: #make sure that the screenshot is not blacklisted before further considering it

                            descriptions = [row[28], row[29], row[30], row[31], row[32]] #descriptions is a row of the descriptions. In order: [HighLevel, LowLevel1, LowLevel2, LowLevel3, LowLevel4]

  
                            #if fn == GEMMA_PREFIX + "com.zizmos.equake-screens/screenshot_4.png":
                                #print(descriptions)

                            #if fn == GEMMA_PREFIX + "com.peter.lovestickers-screens/screenshot_3.png":
                                #print(descriptions)
                            
                            
                            
                            # Note: (row[15] == "A1USYYS8TPQ31A") is for the one worker who puts multiple sentences in his low level descriptions.
                            all_empty = clean_descriptions(descriptions, (row[15] == "A1USYYS8TPQ31A"), blacklist_row) #whether all the descriptions in the cleaned list are empty
                                                                         #clean_descriptions also casts all descriptions to lowercase, removes punctuation, and removes unicode
                                                                         
                                                                         #Basically, if passing 'descriptions' into 'clean_descriptions' results in a list of empty strings,
                                                                         #then the descriptions list is garbage, so we don't add it to filenames to be written to unique.csv
                            

                                
                            if all_empty == False: # i.e. if the descriptions list has at least one non empty string
                                
                                description_list_key = "High: " + descriptions[0] + "\nLow1: " + descriptions[1] + "\nLow2: " + descriptions[2] + "\nLow3: " + descriptions[3] + "\nLow4: " + descriptions[4] #key for the description_list_duplicates
                                
                                if description_list_key in description_list_duplicates: # we have a problematic case; completely copied descriptions
                                    description_list_duplicates[description_list_key].append(fn) # append the filename to the duplicate description list
                                    
                                    #Since this description list has been seen before, we have a duplicate description list on our hands
                                    #For every filename with this duplicated description list, if it's in the filenames dictionary, remove any matching description lists in the filenames dictionary
                                    
                                    #(Essentially getting rid of the duplicate description lists associated with each filename)
                                    
                                    
                                    for bad_fn in description_list_duplicates[description_list_key]:
                                        if bad_fn in filenames: # if the bad filename is in the filenames dictionary,
                                            
                                            to_remove = [] # list of indices to remove from filenames[bad_fn]
                                            
                                            
                                            for i in range(len(filenames[bad_fn])): # for every description list associated with this filename
                                                desc_list = filenames[bad_fn][i]
                                                
                                                if desc_list == descriptions: # if this desc_list equals the garbage duplicate description list,
                                                    to_remove.append(i)       # then add the index of this desc_list to the list of items to remove from filenames[bad_fn]
                                        
                                        
                                            for index in sorted(to_remove, reverse=True): # remove indices from to_remove in reverse order
                                                del filenames[bad_fn][index]
                                            
                                            
                                            
                                else:
                                    description_list_duplicates[description_list_key] = [fn] # make a new list containing only the filename
                                    
                                    #Since this is the first time we've seen this description list, we assume it's a non-duplicate and add its description list to the filenames dictionary
                                    
                                    if fn in filenames: #if the key already exists, it must be a duplicate, but we don't worry about it for now and we just add its descriptions
                                        filenames[fn].append(descriptions)
                                    else: #if the key does not exist, this is the first time we've seen this filename, so we make a new key and add the descriptions
                                        filenames[fn] = [descriptions]
                                

                            else: # happens when ALL the descriptions are the empty string
                                empty_desc[fn] = 1
                            
                elif row != std_header: #i.e. the row doesn't have 33 elements (because it failed the first if statement) AND is not the standard header (which is in reference to this if statement)
                    print("Malformed row in " + f + ": \"" + row + "\"")
    
    
    #We just created the filenames dictionary, but we need to get rid of filenames associated with an empty list; see below for when this will happen
    #note: the length of the description_list will be 0 if the filename ONLY had completely duplicated descriptions associated with it; in this case, we do not want it in the master list so remove it from the filenames dictionary
    
    for fname in list(filenames): #iterate through all the keys of dictionary
        if len(filenames[fname]) == 0:
            del filenames[fname] #delete filenames[fname] from the dictionary because it has an empty descriptions list
    
    if verbose == 1:
        for key in description_list_duplicates:
            if len(description_list_duplicates[key]) > 1: # i.e. a description duplicate is anything that has more than one filename with the corresponding description
                
                print(("=" * 65) + "Duplicate Descriptions" + ("=" * 65))
                print(key + "\n")
                
                for f in description_list_duplicates[key]:
                    print(f)
                    
                print("(x" + str(len(description_list_duplicates[key])) + ")\n")
                
                print(("=" * 150) + "\n\n")
    
    header_str = "\"Filename\"," + "\"High\"," + "\"Low1\"," + "\"Low2\"," + "\"Low3\"," + "\"Low4\"," + "\"Split\"\n"

    duplicate_str = header_str #string to hold the duplicate csv
    unique_str = header_str #string to hold the unique csv


    def count_words(descriptions):
        
        nwords = 0
        
        for desc in descriptions:
            words = desc.strip().split(' ')
            if words != ['']: #i.e. the empty string case
                nwords += len(words)

        return nwords

    def format_row(row_list): #formats a row to append to a csv string 
                                                  #this is a method to prevent code repetition
        
        name = row_list[0]
        descriptions = row_list[1]
        split = row_list[2]
        
        return "\"" + name + "\"," + "\"" + descriptions[0] + "\"," + "\"" + descriptions[1] + "\","  + "\"" + descriptions[2] + "\","  + "\"" + descriptions[3] + "\","  + "\"" + descriptions[4] + "\",\"" + split + "\"\n"



    print("\nWriting main csv files...")

    # Now given that we have a dictionary with all the used filenames,
    # create master-list.csv from Clarity-images-SQL.csv
    # the master list will have a "Used" column with 0 for unused, and 1 for used
    # the master list also has a "Blacklisted" column with 0 for blacklisted, and 1 for not blacklisted (blacklisted screens are landscape, blank, or in a foreign language) - the full list is screenshot_blacklist

    master_list_path = os.path.join(master_list_dir, "master-list.csv")

    master_str = "Filename,Used,Blacklisted\n"



    sql_csv = open(sql_path, "r")

    sql_lines = sql_csv.read().splitlines()

    sql_csv.close()

    del sql_lines[0] #get rid of the header row (first row)

    # read sql csv row by row

    nsqlrows = 0.0

    nused = 0.0

    master_entries = {}

    for row in csv.reader(sql_lines, delimiter=',', skipinitialspace=True):
        
        fname = "http://173.255.245.197:8080/GEMMA-CP/" + row[1] # row[1] is the file name without the server url prefix (except "Clarity")
         
        used = ""; # whether the screenshot is used
        blacklisted = ""; # whether the screenshot is blacklisted (landscape, blank, foreign language screens)
        #(fname in screenshot_blacklist) or
        if (fname in filenames) or (fname in sampled_screenshots): # if the file name is in the dictionary OR the filename is a blacklisted screenshot (b/c we dont want to sample landscape/blacklisted screens again) then put "1" in the used column and "0" in the unused column
            master_entries[fname] = 1                                                                 # or the file name is in the sampled screenshots
            used = "1"
            nused += 1
        else:
            used = "0"
        
        
        if (fname in screenshot_blacklist): # if the screenshot is blacklisted
            blacklisted = "1" # mark it as blacklisted
        else:
            blacklisted = "0" # otherwise don't mark it as blacklisted
        
        master_str += fname + "," + used + "," + blacklisted + "\n"
        
        nsqlrows += 1

    master_list = open(master_list_path, "w")
    master_list.write(master_str)
    master_list.close()

    print("Wrote " + str(int(nsqlrows)) + " entries to " + master_list_path + " (" + str(int(nused)) + " used or " + str(int((nused/nsqlrows)*1000)/10.0) + "%)")





    #Now write unique.csv and duplicate.csv

    nduplicates = 0
    
    unique_entries = [] # FINAL LIST of unique captions paired with their screenshot, and their split; i.e. [GEMMA_PREFIX + "com.psvn.traumatologist-screens/screenshot_1.png", ["a","b","c","d"], "val"]
    
    
    
    for name in filenames:
        
        if name not in master_entries: # i.e. if the filename is NOT in the sql csv (which is the case for 34 screenshots that should have never been tagged)
            
            if verbose == 1:
                print("excluded because not in sql csv: " + name)
                
            continue;
            
        if name in empty_desc: # i.e. if a filename got put in empty_desc which coincidentally had a duplicate that wasn't empty,
            empty_desc[name] = 0 # mark the empty_desc filename as "saved"; i.e. there was an instance of this filename that had nonempty descriptions
                                 # (since filenames with nonempty descriptions get put into the `filenames` dictionary in the first place)
                                 # this only matters for printing (when verbose == 1)
                                 
        description_list = filenames[name]
            

        if len(description_list) == 1: # case 1, we have a unique entry!
            
            unique_entries.append([name, description_list[0]]) # add the entry to unique_entries for later use (for seq2seq and neuraltalk2)
                
        elif len(description_list) > 1: #case 2, we have a duplicate entry!
                
            # it's a duplicate, so append one of its instances to unique_entries, and append the rest to duplicate_str
                
            # we choose to write to the unique entry the one with the most words (with the assumption that more words = richer description)
                
                
            most_words = 0 #max number of words of any description list so far
            most_words_index = 0 #the index in description_list containing the descriptions with the max words
                
            for i in range(len(description_list)):
                nwords = count_words(description_list[i])
                    
                if nwords > most_words:
                    most_words = nwords
                    most_words_index = i
                        
                

            unique_entries.append([name, description_list[most_words_index]]) # add the entry to unique_entries for later use (for seq2seq and neuraltalk2)
                
            
            for i in range(len(description_list)):
                if i != most_words_index:
                    duplicate_str += format_row([name, description_list[i], "N/A"]) # pass N/A for the split since this entry is never going to be used
                    nduplicates += 1
    
    
    
    
    # assign train, val, and test split
    
    nunique = len(unique_entries)
    
    #print(str(nunique) + " unique entries.")
    
    val_entries = [] # holds unique entries assigned to val
    test_entries = [] # holds unique entries assigned to test
    train_entries = [] # holds unique entries assigned to train
    
    for i in range(int(nunique * perc_val)): # do val split
        
        rand_index = random.randint(0, len(unique_entries)-1)
        
        random_entry = unique_entries[rand_index]
        random_entry.append("val") # brand the random entry with the "val" split tag
        
        val_entries.append(random_entry) # append random_entry to val_entries
        
        del unique_entries[rand_index]
    
    
    #print(str(len(val_entries)) + " val entries.")
    
    
    
    for i in range(int(nunique * perc_test)): # do test split
        
        rand_index = random.randint(0, len(unique_entries)-1)
        
        random_entry = unique_entries[rand_index]
        random_entry.append("test") # brand the random entry with the "test" split tag
        
        test_entries.append(random_entry) # append random_entry to test_entries
        
        del unique_entries[rand_index]
        
        
    #print(str(len(test_entries)) + " test entries.")
    
    
    
    for i in range(len(unique_entries)): # do train split (i.e. all the remaining entries in unique_entries since val and test have been removed out
        unique_entries[i].append("train")
        train_entries.append(unique_entries[i]) # all the remaining entries in unique_entries get put in train_entries
    
    
    #print(str(len(train_entries)) + " train entries.")

    
    # rebuild unique_entries from the val, test, and train lists
    
    unique_entries = [] # first empty unique_entries
    
    for entry in train_entries: # append all train entries
        unique_entries.append(entry)
    
    for entry in val_entries: # append all val entries
        unique_entries.append(entry)
    
    for entry in test_entries: # append all test entries
        unique_entries.append(entry)
    
    
    for entry in unique_entries: #build the unique str for writing out to unique.csv
        unique_str += format_row(entry)
    
    #print(str(len(unique_entries)) + " unique entries.")
    
    
    if verbose == 1:
        for k in empty_desc:
            if empty_desc[k] == 1: # if there was only ever one instance of this screenshot, and no duplicate was able to save it,
                print("excluded because empty description list: " + k)
    
        #Also print common typos
        
        ntypos = 0
        
        for key in typos_leaderboard:
            ntypos += typos_leaderboard[key]
        
        print("Most common typos (corrected " + str(ntypos) + " typos total): ")

        for key, value in sorted(typos_leaderboard.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
            print "%s (%s): %s" % (key, typos[key], value)

    
    
    #Write unique and duplicate csv
    
    processed_csv_dir = "../../processed-csv-data"
    
    unique_path = os.path.join(processed_csv_dir, "unique.csv") 
    unique = open(unique_path,"w")
    unique.write(unique_str)
    unique.close()
         
    print("Wrote " + str(nunique) + " unique entries to " + unique_path + "")

    duplicate_path = os.path.join(processed_csv_dir, "duplicate.csv") 
    duplicate = open(duplicate_path,"w")
    duplicate.write(duplicate_str)
    duplicate.close()

    print("Wrote " + str(nduplicates) + " duplicate entries to " + duplicate_path + "")





    ##########################################################  Neuraltalk2  ##########################################################

    #Note: a neuraltalk2 json input file looks like the following:
    '''[
    {
        "captions": [
            "On the top-half of the screen there is a list of most common providers that you can pick to watch ESPN online",
            "On the bottom-half there is a full list of providers you can pick to watch ESPN online"
        ],
        "file_path": "/scratch/ayachnes/Data/ClarityJpegs/air.WatchESPN-screens/screenshot_6.jpg"
    },
    {
        "captions": [
            "On the top-right of the screen there is a picker to select the language of the application",
            "On the center of the screen there is a list of buttons to register a new user, log in for existing users, and log in with Facebook"
        ],
        "file_path": "/scratch/ayachnes/Data/ClarityJpegs/co.zodiacmatch.freedating-screens/screenshot_1.jpg"
    }'''

    print("Done.\n\nWriting neuraltalk2 files...")
    
    processed_neuraltalk2_dir = "../../processed-neuraltalk2-data"
    
    
    
    data_both = open(os.path.join(processed_neuraltalk2_dir, "data-both.json"), "w")
    data_low = open(os.path.join(processed_neuraltalk2_dir, "data-low.json"), "w")
    data_high = open(os.path.join(processed_neuraltalk2_dir, "data-high.json"), "w")
    
    both_json = []
    low_json = []
    high_json = []
    
    for entry in unique_entries: # build the three json files for neuraltalk2
        
        # note: entry is of the form [filename, description_list, split_type]
        
        
        
        jpeg_path = entry[0][len(GEMMA_PREFIX):len(entry[0])] # i.e. turn http://173.255.245.197:8080/GEMMA-CP/Clarity/com.citc.aag-screens/screenshot_4.png
                                                              # into /scratch/ayachnes/Clarity-Data/ClarityJpegs/com.citc.aag-screens/screenshot_4.png
                                                   
        jpeg_path = (jpeg_path[0:len(jpeg_path)-4] + ".jpg") # get rid of .png and replace it with .jpg
        
        append_low = (not (entry[1][1] == "" and entry[1][2] == "" and entry[1][3] == "" and entry[1][4] == "")) # whether to append the low dictionary (based on if all low descriptions are empty or not)
        append_high = (entry[1][0] != "") # whether to append the high dictionary (based on if the high description is empty or not)
        
        both_dict = {}
        low_dict = {}
        high_dict = {}
        
        # give each their respective captions
        both_dict["captions"] = [] #entry[1] # because entry[1] is the full list of captions
        low_dict["captions"] = [] #[entry[1][1], entry[1][2], entry[1][3], entry[1][4]] # because entry[1][1] to entry[1][4] are the low level descriptions
        high_dict["captions"] = [] #[entry[1][0]] # because entry[1][0] is the high level description
        
        for cap in entry[1]: # only append nonempty captions to both_dict
            if cap != "":
                both_dict["captions"].append(cap)
                

                
        
        # give each the jpeg path
        both_dict["file_path"] = jpeg_path
        low_dict["file_path"] = jpeg_path
        high_dict["file_path"] = jpeg_path
        
        # give each their split
        
        both_dict["split"] = entry[2]
        low_dict["split"] = entry[2]
        high_dict["split"] = entry[2]
        
        
        if (len(both_dict["captions"]) > 0): # i.e. if there was at least one nonempty caption appended to both_dict (which is definitely the case since sets of all empty captions are filtered out beforehand)
            both_json.append(both_dict)
        
        if append_low:
            
            for cap in [entry[1][1], entry[1][2], entry[1][3], entry[1][4]]: # only append nonempty captions to both_dict
                if cap != "":
                    low_dict["captions"].append(cap)
                    
            low_json.append(low_dict)
        
        if append_high:
            high_dict["captions"].append(entry[1][0]) # append the nonempty high level caption
            high_json.append(high_dict)
    
    
    
    # dump the three json files
    
    json.dump(both_json, data_both)
    print("Wrote " + data_both.name)
    
    json.dump(low_json, data_low)
    print("Wrote " + data_low.name)
    
    json.dump(high_json, data_high)
    print("Wrote " + data_high.name)
    
    data_both.close()
    data_low.close()
    data_high.close()
    
    
    
    
    ##########################################################  im2txt  ##########################################################
    
    print("Done.\n\nWriting im2txt files...")
    
    
    processed_im2txt_dir = "../../processed-im2txt-data"

    #counter keeps track of the image id
    counter = 1
    
    #caption_count keeps track of the caption id
    caption_count_both = 0
    
    im2txt_train_json_both = {}
    im2txt_val_json_both = {}
    im2txt_test_json_both = {}

    train_image_lst_both = []
    val_image_lst_both = []
    test_image_lst_both = []

    train_caption_lst_both = []
    val_caption_lst_both = []
    test_caption_lst_both = []
    
    
    
    #caption_count keeps track of the caption id
    caption_count_high = 0
    
    im2txt_train_json_high = {}
    im2txt_val_json_high = {}
    im2txt_test_json_high = {}

    train_image_lst_high = []
    val_image_lst_high = []
    test_image_lst_high = []

    train_caption_lst_high = []
    val_caption_lst_high = []
    test_caption_lst_high = []
    
    
    
    
    #caption_count keeps track of the caption id
    caption_count_low = 0
    
    im2txt_train_json_low = {}
    im2txt_val_json_low = {}
    im2txt_test_json_low = {}

    train_image_lst_low = []
    val_image_lst_low = []
    test_image_lst_low = []

    train_caption_lst_low = []
    val_caption_lst_low = []
    test_caption_lst_low = []
    
    

    def make_image_json_entry(img_id, height, width):
        """Return a dictionary that can be stored as a json

        Arguments:
        img_id -- image id     -> string
        height -- image height -> string
        width  -- image width  -> string
        """
        image_entry = {}
        image_entry["file_name"] = "image_id_" + img_id + ".jpg"
        image_entry["height"] = height
        image_entry["width"] = width
        image_entry["id"] = img_id
        return image_entry


    def make_caption_json_entry(img_id, caption_id, caption):
        """Return a dictionary that can be stored as a json

        Arguments:
        img_id     -- the associated image id  -> string
        caption_id -- the id of the caption    -> string
        caption    -- the caption to be stored -> string
        """
        caption_entry = {}
        caption_entry["image_id"] = img_id
        caption_entry["id"] = caption_id
        caption_entry["caption"] = caption
        return caption_entry


    def append_to_caption_lst(caption_lst, img_id, captions, caption_count):
        """Append caption entry to the caption list
        
        Arguments:
        img_id      -- id of the associated image --string
        caption_id  -- id of the caption          --string
        img_captions -- captions for an image     --string
        """ 
        
        # local count of how many captions were appended (i.e. 5 for both, 4 for low level, 1 for high level)
        
        local_caption_count = 0
        
        # captions is a list of the form [high, low1, low2, low3, low4]
        
        for curr_caption in captions:
            #loop through the list of captions
            if curr_caption != "": # only append nonempty captions
                caption_entry = make_caption_json_entry(img_id, caption_count + local_caption_count, curr_caption)
                caption_lst.append(caption_entry)
                local_caption_count += 1
        
        return local_caption_count


    
    # a tar dir for each machine (hudson, bg9, semeru tower two)
    tar_dirs = [["im2txt-split-bg9", BG9_JPG_PATH], 
                ["im2txt-split-hudson", HUDSON_JPG_PATH], 
                ["im2txt-split-semeru2", SEMERU2_JPG_PATH], 
                ["im2txt-split-sciclone", SCICLONE_JPG_PATH],
                ["im2txt-split-bg4", BG4_JPG_PATH]] 
    
    for tar in tar_dirs:
        
        tar_dir = os.path.join("./",tar[0]) # i.e. './im2txt-split-bg9'
        
        if not os.path.exists(tar_dir): # make the tar_dir if it doesn't exist
            os.system("mkdir " + tar_dir)
            
            
        split_folders = [os.path.join(tar_dir, "train"), os.path.join(tar_dir, "val"), os.path.join(tar_dir, "test")]
            
            
        for direc in split_folders: # make train, val, and test dirs
            
            if os.path.exists(direc):
                os.system("rm -R " + direc)

            os.system("mkdir " + direc)
        
        
        
    
    for entry in unique_entries: # build the json files for im2txt
        
        # note: entry is of the form [filename, description_list, split_type]
        
        screen_name = entry[0]
        
        captions_both = entry[1] # i.e. [high, low1, low2, low3, low4] for this single screenshot
        captions_high = [entry[1][0]] # i.e. [high]
        captions_low = [entry[1][1], entry[1][2], entry[1][3], entry[1][4]]  # i.e. [low1, low2, low3, low4]
        
        split = entry[2]

        img_id = str(counter).zfill(7)
        image_entry = make_image_json_entry(img_id, "1920", "1200")
        
        
        num_appended_both = 0 # number of captions added by 'append_to_caption_lst' for 'both'
        num_appended_high = 0 # number of captions added by 'append_to_caption_lst' for 'high'
        num_appended_low = 0 # number of captions added by 'append_to_caption_lst' for 'low'
        
        
        
        # append "both" level captions
        
        image_lst_both = [] # the chosen 'both' image list to append to (depending on the split)
        caption_lst_both = [] # the chosen 'both' caption list to append to (depending on the split)

        image_lst_high = [] # the chosen 'high' image list to append to (depending on the split)
        caption_lst_high = [] # the chosen 'high' caption list to append to (depending on the split)
        
        image_lst_low = [] # the chosen 'low' image list to append to (depending on the split)
        caption_lst_low = [] # the chosen 'low' caption list to append to (depending on the split)
        

        if split == "train":
            image_lst_both = train_image_lst_both
            caption_lst_both = train_caption_lst_both

            image_lst_high = train_image_lst_high
            caption_lst_high = train_caption_lst_high
            
            image_lst_low = train_image_lst_low
            caption_lst_low = train_caption_lst_low
            
        elif split == "val":
            image_lst_both = val_image_lst_both
            caption_lst_both = val_caption_lst_both

            image_lst_high = val_image_lst_high
            caption_lst_high = val_caption_lst_high
            
            image_lst_low = val_image_lst_low
            caption_lst_low = val_caption_lst_low

        elif split == "test":
            image_lst_both = test_image_lst_both
            caption_lst_both = test_caption_lst_both

            image_lst_high = test_image_lst_high
            caption_lst_high = test_caption_lst_high
            
            image_lst_low = test_image_lst_low
            caption_lst_low = test_caption_lst_low
            
        
        
        # append to the chosen lists (low, high, and both lists)
        
        
        
        # both
        
        if captions_both != ["","","","",""]: # if not all the 'both' level captions are blank (this is definitely the case)
            image_lst_both.append(image_entry)
            num_appended_both = append_to_caption_lst(caption_lst_both, img_id, captions_both, caption_count_both)
        
        
        
        # high
        
        if captions_high != [""]: # i.e. only append if the high caption isn't blank (a few in the dataset are)
            image_lst_high.append(image_entry)
            num_appended_high = append_to_caption_lst(caption_lst_high, img_id, captions_high, caption_count_high)
            
            
        
        # low
        
        if captions_low != ["", "", "", ""]: # i.e. only append if there is at least one nonempty low level caption
            image_lst_low.append(image_entry)
            num_appended_low = append_to_caption_lst(caption_lst_low, img_id, captions_low, caption_count_low)
            
            
        
        
        # create symlink for this image
            
        for tar in tar_dirs:
            
            tar_dir = tar_dir = os.path.join("./",tar[0]) # i.e. './im2txt-split-bg9'
            
            machine_path = tar[1]
            
            jpeg_path = os.path.join(machine_path,screen_name[len(GEMMA_PREFIX):len(entry[0])]) # i.e. turn http://173.255.245.197:8080/GEMMA-CP/Clarity/com.citc.aag-screens/screenshot_4.png
                                                                                                        # into ${CLARITYJPEGS}/com.citc.aag-screens/screenshot_4.png
                                                       
            jpeg_path = (jpeg_path[0:len(jpeg_path)-4] + ".jpg") # get rid of .png and replace it with .jpg
            
            #print(jpeg_path)
           
            os.symlink(jpeg_path, os.path.join(tar_dir, split, "image_id_" + img_id + ".jpg"))
        
        caption_count_both += num_appended_both
        caption_count_high += num_appended_high
        caption_count_low += num_appended_low
        counter += 1

    
    for tar in tar_dirs:
        tar_dir = os.path.join("./",tar[0]) # i.e. './im2txt-split-bg9'
        
        # Create tar file with symlinks for im2txt
        
        tar_name = os.path.join(processed_im2txt_dir, tar[0] + ".tar.gz")

        with tarfile.open(tar_name, "w:gz") as tar:
            tar.add(tar_dir, arcname=os.path.basename(tar_dir))

        os.system("rm -R " + tar_dir) # remove the directory after compression happens

        print("Wrote " + tar_name)



    # write out 'both' jsons (3)
    
    train_header_both = {"description": "This is the Clarity dataset train split with combined high and low level captions.", "version": "1.0", "year": "2018"} # info field for the json file
    val_header_both = {"description": "This is the Clarity dataset val split with combined high and low level captions.", "version": "1.0", "year": "2018"} # info field for the json file
    test_header_both = {"description": "This is the Clarity dataset test split with combined high and low level captions.", "version": "1.0", "year": "2018"} # info field for the json file
    
    im2txt_train_json_both = {"type":"captions","licenses": [],"info": train_header_both, "images":train_image_lst_both, "annotations":train_caption_lst_both}
    im2txt_val_json_both = {"type":"captions","licenses": [],"info": val_header_both, "images":val_image_lst_both, "annotations":val_caption_lst_both}
    im2txt_test_json_both = {"type":"captions","licenses": [],"info": test_header_both, "images":test_image_lst_both, "annotations":test_caption_lst_both}
    
    
    outfile = open(os.path.join(processed_im2txt_dir,"both", "captions_train.json"), "w")
    json.dump(im2txt_train_json_both, outfile)
    outfile.close()
    
    print("Wrote " + outfile.name)
    
    
    outfile = open(os.path.join(processed_im2txt_dir,"both", "captions_val.json"), "w")
    json.dump(im2txt_val_json_both, outfile)
    outfile.close()
    
    print("Wrote " + outfile.name)
    
    
    outfile = open(os.path.join(processed_im2txt_dir,"both", "captions_test.json"), "w")
    json.dump(im2txt_test_json_both, outfile)
    outfile.close()
    
    print("Wrote " + outfile.name)
    
    
    
    
    # write out 'high' jsons (3)
    
    train_header_high = {"description": "This is the Clarity dataset train split with only high level captions.", "version": "1.0", "year": "2018"} # info field for the json file
    val_header_high = {"description": "This is the Clarity dataset val split with only high level captions.", "version": "1.0", "year": "2018"} # info field for the json file
    test_header_high = {"description": "This is the Clarity dataset test split with only high level captions.", "version": "1.0", "year": "2018"} # info field for the json file
    
    im2txt_train_json_high = {"type":"captions","licenses": [], "info": train_header_high, "images":train_image_lst_high, "annotations":train_caption_lst_high}
    im2txt_val_json_high = {"type":"captions","licenses": [], "info": val_header_high, "images":val_image_lst_high, "annotations":val_caption_lst_high}
    im2txt_test_json_high = {"type":"captions","licenses": [], "info": test_header_high, "images":test_image_lst_high, "annotations":test_caption_lst_high}
    
    
    outfile = open(os.path.join(processed_im2txt_dir,"high", "captions_train.json"), "w")
    json.dump(im2txt_train_json_high, outfile)
    outfile.close()
    
    print("Wrote " + outfile.name)
    
    
    outfile = open(os.path.join(processed_im2txt_dir,"high", "captions_val.json"), "w")
    json.dump(im2txt_val_json_high, outfile)
    outfile.close()
    
    print("Wrote " + outfile.name)
    
    
    outfile = open(os.path.join(processed_im2txt_dir,"high", "captions_test.json"), "w")
    json.dump(im2txt_test_json_high, outfile)
    outfile.close()
    
    print("Wrote " + outfile.name)
    
    

    
    
    
    
    # write out 'low' jsons (3)
    
    train_header_low = {"description": "This is the Clarity dataset train split with only low level captions.", "version": "1.0", "year": "2018"} # info field for the json file
    val_header_low = {"description": "This is the Clarity dataset val split with only low level captions.", "version": "1.0", "year": "2018"} # info field for the json file
    test_header_low = {"description": "This is the Clarity dataset test split with only low level captions.", "version": "1.0", "year": "2018"} # info field for the json file
    
    im2txt_train_json_low = {"type":"captions","licenses": [], "info": train_header_low, "images":train_image_lst_low, "annotations":train_caption_lst_low}
    im2txt_val_json_low = {"type":"captions","licenses": [], "info": val_header_low, "images":val_image_lst_low, "annotations":val_caption_lst_low}
    im2txt_test_json_low = {"type":"captions","licenses": [], "info": test_header_low, "images":test_image_lst_low, "annotations":test_caption_lst_low}
    
    
    outfile = open(os.path.join(processed_im2txt_dir,"low", "captions_train.json"), "w")
    json.dump(im2txt_train_json_low, outfile)
    outfile.close()
    
    print("Wrote " + outfile.name)
    
    
    outfile = open(os.path.join(processed_im2txt_dir,"low", "captions_val.json"), "w")
    json.dump(im2txt_val_json_low, outfile)
    outfile.close()
    
    print("Wrote " + outfile.name)
    
    
    outfile = open(os.path.join(processed_im2txt_dir,"low", "captions_test.json"), "w")
    json.dump(im2txt_test_json_low, outfile)
    outfile.close()
    
    print("Wrote " + outfile.name)
    
    
    
    
    
    
    
    ##########################################################  seq2seq  ##########################################################
    
    
    
    print("Done.\n\nWriting seq2seq files...")
    
    #Write seq2seq files (high, low, both)
    
    for caption_type in ["high", "low", "both"]: # for each caption type
        
        print("\nWriting seq2seq " + caption_type + "...\n")

        
        '''The purpose of this code is to create parallel text files from three different representations of 
        xml files and captions describing their content'''
        
        # this does train, val, and test splits

        seq2seq_base = "../../processed-seq2seq-uiautomator-data"
        
        key_csv_in = os.path.join(seq2seq_base, "readonly", "key.csv")
        text_csv_in = os.path.join(seq2seq_base, "readonly", "text.csv")
        type_csv_in = os.path.join(seq2seq_base, "readonly", "type.csv")
        type_text_csv_in = os.path.join(seq2seq_base, "readonly", "type_text.csv")
        type_text_loc_csv_in = os.path.join(seq2seq_base, "readonly", "type_text_loc.csv")


        #Files to be aligned

        key_csv_out_train = os.path.join(seq2seq_base, caption_type, "train", "key.csv")
        text_csv_out_train = os.path.join(seq2seq_base, caption_type, "train", "text.csv")
        type_csv_out_train = os.path.join(seq2seq_base, caption_type, "train", "type.csv") 
        type_text_csv_out_train = os.path.join(seq2seq_base, caption_type, "train", "type_text.csv") 
        type_text_loc_csv_out_train = os.path.join(seq2seq_base, caption_type, "train", "type_text_loc.csv") 
        caption_csv_out_train = os.path.join(seq2seq_base, caption_type, "train", "caption.csv")
        
        key_csv_out_val = os.path.join(seq2seq_base, caption_type, "val", "key.csv")
        text_csv_out_val = os.path.join(seq2seq_base, caption_type, "val", "text.csv")
        type_csv_out_val = os.path.join(seq2seq_base, caption_type, "val", "type.csv") 
        type_text_csv_out_val = os.path.join(seq2seq_base, caption_type, "val", "type_text.csv") 
        type_text_loc_csv_out_val = os.path.join(seq2seq_base, caption_type, "val", "type_text_loc.csv") 
        caption_csv_out_val = os.path.join(seq2seq_base, caption_type, "val", "caption.csv")
        
        key_csv_out_test = os.path.join(seq2seq_base, caption_type, "test", "key.csv")
        text_csv_out_test = os.path.join(seq2seq_base, caption_type, "test", "text.csv")
        type_csv_out_test = os.path.join(seq2seq_base, caption_type, "test", "type.csv") 
        type_text_csv_out_test = os.path.join(seq2seq_base, caption_type, "test", "type_text.csv") 
        type_text_loc_csv_out_test = os.path.join(seq2seq_base, caption_type, "test", "type_text_loc.csv") 
        caption_csv_out_test = os.path.join(seq2seq_base, caption_type,  "test", "caption.csv")
        

        
        
        key_dict = {} # dictionary mapping the xml filename to the line number; i.e. "a2dp.Vol-screens/hierarchy_1.xml" -> 1
        
        with open(key_csv_in, "r") as f:
            lines = f.read().splitlines()
            
            ln = 0
            
            for l in lines:
                key_dict[l] = ln
                ln += 1
        
        
        seq2seq_files_train = {} # dictionary mapping filenames to their contents read in and their contents to write out (train split)
                                 # seq2seq_files[key] = [read_in, write_out]
        seq2seq_files_val = {} # same as above, but val split
        
        seq2seq_files_test = {} # same as above, but test split
        
        
        file_pairings = [ # list with "in_file : [out_train, out_val, out_test]
                        [key_csv_in,  [key_csv_out_train, key_csv_out_val, key_csv_out_test]], 
                        [text_csv_in, [text_csv_out_train, text_csv_out_val, text_csv_out_test]], 
                        [type_csv_in, [type_csv_out_train, type_csv_out_val, type_csv_out_test]], 
                        [type_text_csv_in, [type_text_csv_out_train, type_text_csv_out_val, type_text_csv_out_test]], 
                        [type_text_loc_csv_in, [type_text_loc_csv_out_train, type_text_loc_csv_out_val, type_text_loc_csv_out_test]]
                        ] 


        #store the file content from the csv files in the dictionary for later usage
        for seqfn in file_pairings:
            with open(seqfn[0], "r") as f:
                lines = f.read().splitlines()
                
                
                # seqfn[1] is a list of train, val, and test output files
                
                seq2seq_files_train[seqfn[1][0]] = [lines, ""] # seqfn[1][0] is out_train
                
                seq2seq_files_val[seqfn[1][1]] = [lines, ""] # seqfn[1][1] is out_val
                
                seq2seq_files_test[seqfn[1][2]] = [lines, ""] # seqfn[1][2] is out_test


        FILENAME_BASE_LEN = len(GEMMA_PREFIX)
        FILENAME_OFFSET = len("-screens/screenshot_5.png")

        
        captions_str_train = "" # string to be written out to the captions file (train)
        captions_str_val = "" # string to be written out to the captions file (val)
        captions_str_test = "" # string to be written out to the captions file (val)
        
        
        for entry in unique_entries: # go through each entry that was appended to the unique csv
                                     # and arrange all files in the same order as unique.csv
            
            # note: entry is of the form [filename, description_list, split_type]
            
            package_name = entry[0][FILENAME_BASE_LEN:-FILENAME_OFFSET]
            screen_number = entry[0][-FILENAME_OFFSET:][-5] # get which number screen this is
            package_and_offset = package_name + "-screens/hierarchy_" + screen_number + ".xml"
            
            index = key_dict[package_and_offset]
            
            
            split = entry[2] # the "train", "val", or "test" split assigned to this entry
            
            seq2seq_files = None # pointer to which dictionary we are using to find the file to write to

            
            captions_append_str = ""
            
            if (caption_type == "high" and entry[1][0] == "") or (caption_type == "low" and ((entry[1][1] == "") and (entry[1][2] == "") and (entry[1][3] == "") and (entry[1][4] == ""))):
                continue; # continue b/c either the high is empty during a "high" run or all the lows are empty during a "low" run
            
            
            
            if caption_type == "high": 
                captions_append_str = "\"" + entry[1][0] + "\"\n"
            elif caption_type == "low":
                captions_append_str = "\"" + entry[1][1] + "\",\"" + entry[1][2] + "\",\"" + entry[1][3] + "\",\"" + entry[1][4] + "\"\n"
            elif caption_type == "both":
                captions_append_str = "\"" + entry[1][0] + "\",\"" + entry[1][1] + "\",\"" + entry[1][2] + "\",\"" + entry[1][3] + "\",\"" + entry[1][4] + "\"\n"
            
            if split == "train":
                seq2seq_files = seq2seq_files_train
                captions_str_train += captions_append_str
            elif split == "val":
                seq2seq_files = seq2seq_files_val
                captions_str_val += captions_append_str
            elif split == "test":
                seq2seq_files = seq2seq_files_test
                captions_str_test += captions_append_str
                
                
            
            for key in seq2seq_files: # go through each [read_in, write_out] pair in the seq2seq_files dictionary (which is either the train, val, or test split)
                read_write = seq2seq_files[key]
                read_write[1] += read_write[0][index] + "\n"
                

            
            
        
        for seq2seq_files in [seq2seq_files_train, seq2seq_files_val, seq2seq_files_test]: # go through each seq2seq_files dictionary (train, val, test)
            for key in seq2seq_files: # go through each [read_in, write_out] pair in the seq2seq_files dictionary and write out to files
                read_write = seq2seq_files[key]
                
                with open(key, "w") as f:
                    f.write(read_write[1])
                    print("Wrote " + key)
                
                    
        with open(caption_csv_out_train, "w") as f: # output captions.csv with ONLY the captions from unique.csv (train split)
            f.write(captions_str_train)
            print("Wrote " + caption_csv_out_train)
            
            
        with open(caption_csv_out_val, "w") as f: # output captions.csv with ONLY the captions from unique.csv (val split)
            f.write(captions_str_val)
            print("Wrote " + caption_csv_out_val)
            
            
        with open(caption_csv_out_test, "w") as f: # output captions.csv with ONLY the captions from unique.csv (test split)
            f.write(captions_str_test)
            print("Wrote " + caption_csv_out_test)
    
    
    elapsed = time.time() - start_time

    print("\n\nFinished preprocessing in " + str(int(elapsed*100)/100.0) + " seconds.")
    
if __name__ == "__main__":
    main(os.sys.argv)
