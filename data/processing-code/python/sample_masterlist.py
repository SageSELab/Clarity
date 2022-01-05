# file to draw a sample of unused screenshots from the master list and output them to new_batch.csv in the working directory

# should have a 'used' column, chooses N images that have never been used and outputs it to a csv (new_batch.csv); make an initial populating of the used column

# it writes out new_batch.csv in the following format:

'''
single column csv with the addresses of the images, header should say image_url. 

i.e:

image_url
Hit1_image_url_data
Hit2_image_url_data
Hit3_image_url_data

'''

# Note: this file updates all the csv files before continuing to make sure it is using an up to date master list

####################################################################################################################################################################################################


import os
import sys
import csv
import random
import update_csv


if len(os.sys.argv) < 4:
    print("\nDraws a sample of N unused screenshots from the master list and outputs them to 'new_batch.csv' in the working directory.\nIf there are less than N unused screenshots in the master list, it samples all of the unused screenshots.")
    print("This script automatically updates the master list from the raw data before sampling.")
    print("\nusage: python2.7 " + __file__ + " <number of screenshots to sample> <directory with mechanical turk csv files> <directory with master list files>")
    print("\nex: python2.7 " + __file__ + " 3000 ../../mechanical-turk-data/ ../../master-screen-list/\n")
    exit()

update_csv.main([None, os.sys.argv[2], os.sys.argv[3]]) # 2 is directory with mech turk csv files, 3 is directory with master list files

random.seed()

num_sample = int(os.sys.argv[1])

master_list_path = os.sys.argv[3] + "/master-list.csv"

if master_list_path[len(master_list_path)-4:len(master_list_path)] != ".csv":
    print("First argument should be a .csv file!")
    exit()
    
    
master_list = open(master_list_path, "r")

master_lines = master_list.read().splitlines()

master_list.close()

del master_lines[0] #get rid of the header row (first row)


unused_entries = [] #list of strings, each of which is a filename like 'http://173.255.245.197:8080/GEMMA-CP/Clarity/zok.android.shapes-screens/screenshot_1.png'

for row in csv.reader(master_lines, delimiter=',', skipinitialspace=True):
    
    filename = row[0]
    used = row[1]
    blacklisted = row[2] # not used for anything, but just note that each row has 3 columns
    
    if used == "0" and blacklisted == "0": # i.e. if this screenshot was not used and it isn't blacklisted (landscape, blank, etc.)
        unused_entries.append(filename)

    
    
new_batch_str = "image_url\n" # 'image_url' is the header for mechanical turk new batches we send





# take a sample of 'num_sample' unused screens and write them out to new_batch.csv

if len(unused_entries) < num_sample:
    print("\nWarning: There are only " + str(len(unused_entries)) + " unused entries in " + master_list_path + ". Sampling " + str(len(unused_entries)) + " entries instead of " + str(num_sample)+ ".\n")
    num_sample = len(unused_entries)
   


for i in range(num_sample): # at this point num_sample is guaranteed to be <= to len(unused_entries) (no bounds problems)
    
    rand_index = random.randint(0, len(unused_entries) - 1)
    
    new_batch_str += unused_entries[rand_index] +  "\n"
    
    del unused_entries[rand_index]




new_batch_path = "new_batch-" + str(num_sample) + ".csv" # new batch in the mechanical turk format

new_batch = open(new_batch_path, "w")
new_batch.write(new_batch_str)
new_batch.close()

print("Wrote " + str(num_sample) + " entries to " + new_batch_path)







