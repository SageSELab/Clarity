#Script to count the number of each type of UI in a directory and rename them all, removing Vertical/Horizontal ProgressBars and replacing them with ProgressBars

#python ui_counter.py <dir>











import os
import sys
import re

import hashlib #for hashing files

import random
import math

random.seed()

if len(os.sys.argv) < 2:
    print("\nCounts number of each UI component type in a directory and renames them by their SHA1 hash, renaming any Vertical/Horizontal ProgressBars to normal ProgressBars.\nusage: python " + __file__ + " <dir>")
    print("")
    exit()
    
direc = os.sys.argv[1]

files = os.listdir(direc)

num_files = len(files)

ndigits = math.ceil(math.log(num_files, 16)) # max number of digits needed to encode 'n', the file number

components = {} #dictionary of component and count

#We expect files to be in the following format (as an example): "81-android.widget.TextView.png"
pattern = re.compile("-android.widget." + r"[^\.]+")

n = 0 # keeps track of number of files we have processed so far

for f in files:
    
    if (n % (num_files/20) == 0):
        print("... " + str(int((float(n)/float(num_files))*100)) + "%")
    
    search = re.search(pattern, f)
    
    if search is not None:
        comp = search.group(0)[16:]
        
        # Replace any horizontal/vertical progress bar with just ProgressBar
        if comp == "ProgressBarVertical" or comp == "ProgressBarHorizontal":
            comp = "ProgressBar" 
        
        
        # add component to the components dictionary
        if not comp in components:
            components[comp] = 1
        else:
            components[comp] += 1
            
        
        # use the hash of the file to rename it to something like "ImageButton-d229f0abfc1c05292e4ee5de4f12e150010ada33.png" to avoid collisions
        sha1_hash = hashlib.sha1(open(os.path.join(direc, f), 'rb').read()).hexdigest()
        
        
        rd = "%09x" % (random.randint(1,9999999))
        nd = ("%0" + str(int(ndigits)) + "x") % (n)
        
        new_filename = comp + "-" + nd + "-" + rd + "-" + sha1_hash + f[f.rfind("."):len(f)] #f[f.find("."):len(f)] is the file extension
        
        os.rename(os.path.join(direc, f), os.path.join(direc, new_filename))
        
    else: # i.e. the file does not fall into the format "81-android.widget.TextView.png"
        print("? Invalid file: " + f)
    n = n + 1
for k in components:
    print(k + ": " + str(components[k]))



num_files_post = len(os.listdir(direc)) # the number of files in the directory AFTER all this renaming (it SHOULD be the same, otherwise we overwrote one or more files)
if num_files_post != num_files:
    print("Error: started with " + str(num_files) + " files and ended with " + str(num_files_post) + " files.")
else:
    print("Success: started with " + str(num_files) + " files and ended with " + str(num_files_post) + " files.")
