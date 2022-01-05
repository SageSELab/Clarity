# Script to sort a mechanical turk batch csv by submit time (so that we approve/reject the entries with the least time left FIRST)
# This does not write out anything that is "Approved" or "Rejected"

#Ali Yachnes
#ayachnes@email.wm.edu

import os
import sys
import csv
import re # for regular expression magic



if len(os.sys.argv) < 2:
    print("Sorts a mechanical turk batch csv by submit time (so that we approve/reject the entries with the least time left FIRST).")
    print("Note: this overwrites the original file")
    print("\nusage: python2.7 " + __file__ + " <mturk batch csv file>")
    print("\ni.e. python2.7 " + __file__ + " batch.csv")
    exit()
    
mturk_file = os.sys.argv[1]

if mturk_file[len(mturk_file)-4:len(mturk_file)] != ".csv":
    print("Error: " + mturk_file + " is not a .csv file.")
    exit()
    

if not os.path.isfile(mturk_file):
    print("Error: could not open " + mturk_file)
    exit()

def compare(a,b):
    if a[0] > b[0]:
        return 1
    elif a[0] < b[0]:
        return -1
    else:
        return 0

def format_row(row, is_header = False): #formats a row to output to csv
    out_str = ""
    
    for i in range(len(row)-1):
        out_str += "\"" + row[i].replace("\"","'") + "\","
    
    out_str +=  "\"" + row[len(row)-1].replace("\"","'") + "\""
    
    if not is_header:
        out_str += ",,"
    
    out_str += "\n"

    return out_str

#what a header should look like; this is used to make sure that we are parsing each row correctly, since the script prints any row that is not 33 columns and isn't the standard header row
std_header = ['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate', 'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.image_url', 'Answer.HighLevel', 'Answer.LowLevel1', 'Answer.LowLevel2', 'Answer.LowLevel3', 'Answer.LowLevel4', 'Approve', 'Reject']


csvfile = open(mturk_file, "r")
            
lines = csvfile.read().splitlines()
            
csvfile.close()
        
row_times = []# list of submit times mapped to rows

months = {"Jan" : 0, "Feb" : 1, "Mar" : 2, "Apr" : 3, "May" : 4, "Jun" :  5, "Jul" : 6, "Aug" : 7, "Sep" : 8, "Oct" : 9, "Nov" : 10, "Dec" : 11}


for row in csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
    if len(row) == 33: # i.e. if it is not a header row (because header rows have 35 elements)
        if row[16].lower() == "submitted": # we only consider things in the "submitted" status
            submit_time = row[18] # row[18] is SubmitTime (like "Wed Jul 11 11:41:07 PDT 2018")
            
            times = [int(n) for n in submit_time[11:19].split(":")]
            
            hour = times[0]
            minute = times[1]
            second = times[2]
            day = int(submit_time[8:10])
            month = months[submit_time[4:7]]
            
            timestamp = month * 30 * 24 * 60 * 60 + day * 24 * 60 * 60 + hour * 60 * 60 + minute * 60 + second

            row_times.append([timestamp, row])

    else:
        pass#print(row) # should only happen once (for the header row)

#Sort by times and output row

row_times.sort(cmp=compare)

csvfile = open(mturk_file, "w")

csv_str = ""

csv_str += format_row(std_header, True)

for entry in row_times:
    formatted = format_row(entry[1])
    csv_str += formatted


print("Wrote to " + mturk_file + " sorted by time.")
csvfile.write(csv_str)
csvfile.close()
