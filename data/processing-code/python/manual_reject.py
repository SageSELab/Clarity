# form of each entry is [HITId, WorkerId, Input.image_url]
#                   i.e [row[0], row[15], row[27]]


# this script reads in a mechanical turk csv and outputs all entries with a reject reason in the above manual reject format
# This script is used for batches that were auto approved but have been combed through manually later on to get rid of poor quality submissions


# Ali Yachnes
# ayachnes@email.wm.edu

import os
import sys
import csv



if len(os.sys.argv) < 2:
    print("Reads in a mechanical turk csv and outputs all entries with a reject reason in the manual reject format used by update_csv.py")
    print("\nusage: python2.7 " + __file__ + " <mturk csv file>")
    print("\ni.e. python2.7 " + __file__ + " batch.csv")
    exit()
    
mturk_file = os.sys.argv[1]

if mturk_file[len(mturk_file)-4:len(mturk_file)] != ".csv":
    print("Error: " + mturk_file + " is not a .csv file.")
    exit()
    

if not os.path.isfile(mturk_file):
    print("Error: could not open " + mturk_file)
    exit()


#what a header should look like; this is used to make sure that we are parsing each row correctly, since the script prints any row that is not 33 columns and isn't the standard header row
std_header = ['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate', 'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.image_url', 'Answer.HighLevel', 'Answer.LowLevel1', 'Answer.LowLevel2', 'Answer.LowLevel3', 'Answer.LowLevel4', 'Approve', 'Reject']


csvfile = open(mturk_file, "r")
            
lines = csvfile.read().splitlines()

for row in csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
    if row != std_header: # i.e. if it is not a header row
        if row[34] != "": # i.e. if the rejection column is not empty
            print("\"" + row[27] + "\",")
            #print("['" + row[0] + "', '" + row[15] + "', GEMMA_PREFIX + '" + row[27].replace("http://173.255.245.197:8080/GEMMA-CP/Clarity/","") + "'],")

    



