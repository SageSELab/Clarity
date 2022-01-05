#Reads in a csv, and writes out only the approved rows (this was for undoing the rejects)
import os
import sys
import csv
import re # for regular expression magic



if len(os.sys.argv) < 2:
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

def format_row(row): #formats a row to output to csv
    out_str = ""
    
    for i in range(len(row)-1):
        out_str += "\"" + row[i].replace("\"","'") + "\","
    
    out_str +=  "\"" + row[len(row)-1].replace("\"","'") + "\"\n"
    
    return out_str

#what a header should look like; this is used to make sure that we are parsing each row correctly, since the script prints any row that is not 33 columns and isn't the standard header row
std_header = ['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate', 'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.image_url', 'Answer.HighLevel', 'Answer.LowLevel1', 'Answer.LowLevel2', 'Answer.LowLevel3', 'Answer.LowLevel4', 'Approve', 'Reject']


csvfile = open(mturk_file, "r")
            
lines = csvfile.read().splitlines()
            
csvfile.close()

all_rows = []

for row in csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
    if len(row) == 36: # i.e. if it is not a header row (because header rows have 35 elements)
        del row[35]
        
        if row[33].lower() == "x":
            all_rows.append(row)
    else:
        pass

#Sort by times and output row
out_name = mturk_file + "-dup"
csvfile = open(out_name, "w")

csv_str = ""

csv_str += format_row(std_header)

for entry in all_rows:
    formatted = format_row(entry)
    csv_str += formatted


print("Wrote to " + out_name + ".")
csvfile.write(csv_str)
csvfile.close()
